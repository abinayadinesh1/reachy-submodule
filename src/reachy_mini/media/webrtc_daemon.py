"""WebRTC daemon.

Starts a gstreamer webrtc pipeline to stream video and audio.

This module provides a WebRTC server implementation using GStreamer that can
stream video and audio from the Reachy Mini robot to WebRTC clients. It's
designed to run as a daemon process on the robot and handle multiple client
connections for telepresence and remote monitoring applications.

The WebRTC daemon supports:
- Real-time video streaming from the robot's camera
- Real-time audio streaming from the robot's microphone
- Multiple client connections
- Automatic camera detection and configuration

Example usage:
    >>> from reachy_mini.media.webrtc_daemon import GstWebRTC
    >>>
    >>> # Create and start WebRTC daemon
    >>> webrtc_daemon = GstWebRTC(log_level="INFO")
    >>> # The daemon will automatically start streaming when initialized
    >>>
    >>> # Run until interrupted
    >>> try:
    ...     while True:
    ...         pass  # Keep the daemon running
    ... except KeyboardInterrupt:
    ...     pass  # Cleanup would be handled automatically
"""

import logging
import shutil
import subprocess
from threading import Thread
from typing import Optional, Tuple, cast

import gi
import numpy as np
import numpy.typing as npt
from reachy_mini.media.camera_constants import (
    ArducamSpecs,
    CameraSpecs,
    ReachyMiniLiteCamSpecs,
    ReachyMiniWirelessCamSpecs,
)

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import GLib, Gst, GstApp  # noqa: E402


class GstWebRTC:
    """WebRTC pipeline using GStreamer.

    This class implements a WebRTC server using GStreamer that streams video
    and audio from the Reachy Mini robot to connected WebRTC clients. It's
    designed to run as a daemon process and handle the complete WebRTC
    signaling and media streaming pipeline.

    Attributes:
        _logger (logging.Logger): Logger instance for WebRTC daemon operations.
        _loop (GLib.MainLoop): GLib main loop for handling GStreamer events.
        camera_specs (CameraSpecs): Specifications of the detected camera.
        _resolution (CameraResolution): Current streaming resolution.
        resized_K (npt.NDArray[np.float64]): Camera intrinsic matrix for current resolution.

    """

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the GStreamer WebRTC pipeline.

        Args:
            log_level (str): Logging level for WebRTC daemon operations.
                          Default: 'INFO'.

        Note:
            This constructor initializes the GStreamer environment, detects the
            available camera, and sets up the WebRTC streaming pipeline. The
            pipeline automatically starts streaming when initialized.

        Raises:
            RuntimeError: If no camera is detected or camera specifications cannot
                        be determined.

        Example:
            >>> # Initialize WebRTC daemon with debug logging
            >>> webrtc_daemon = GstWebRTC(log_level="DEBUG")
            >>> # The daemon is now streaming and ready to accept client connections

        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._rpicam_proc: Optional[subprocess.Popen] = None
        self._appsink: Optional[Gst.Element] = None

        Gst.init(None)
        self._loop = GLib.MainLoop()
        self._thread_bus_calls = Thread(target=lambda: self._loop.run(), daemon=True)
        self._thread_bus_calls.start()

        cam_path, self.camera_specs = self._get_video_device()

        if self.camera_specs is None:
            raise RuntimeError("Camera specs not set")
        self._resolution = self.camera_specs.default_resolution
        self.resized_K = self.camera_specs.K

        if self._resolution is None:
            raise RuntimeError("Failed to get default camera resolution.")

        self._pipeline_sender = Gst.Pipeline.new("reachymini_webrtc_sender")
        self._bus_sender = self._pipeline_sender.get_bus()
        self._bus_sender.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )

        webrtcsink = self._configure_webrtc(self._pipeline_sender)

        self._configure_video(cam_path, self._pipeline_sender, webrtcsink)
        self._configure_audio(self._pipeline_sender, webrtcsink)

        self._pipeline_receiver = Gst.Pipeline.new("reachymini_webrtc_receiver")
        self._bus_receiver = self._pipeline_receiver.get_bus()
        self._bus_receiver.add_watch(
            GLib.PRIORITY_DEFAULT, self._on_bus_message, self._loop
        )
        self._configure_receiver(self._pipeline_receiver)

    def _cleanup_rpicam(self) -> None:
        """Terminate the rpicam-vid subprocess if running."""
        if self._rpicam_proc is not None and self._rpicam_proc.poll() is None:
            self._logger.info("Terminating rpicam-vid subprocess")
            self._rpicam_proc.terminate()
            try:
                self._rpicam_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._logger.warning("rpicam-vid didn't exit, killing")
                self._rpicam_proc.kill()
                self._rpicam_proc.wait()
            self._rpicam_proc = None

    def __del__(self) -> None:
        """Destructor to ensure gstreamer resources are released."""
        self._logger.debug("Cleaning up GstWebRTC")
        self._cleanup_rpicam()
        self._loop.quit()
        self._bus_sender.remove_watch()
        self._bus_receiver.remove_watch()
        # Enable if need to dump logs
        # Gst.deinit()

    def _dump_latency(self) -> None:
        query = Gst.Query.new_latency()
        self._pipeline_sender.query(query)
        self._logger.info(f"Pipeline latency {query.parse_latency()}")

    def _configure_webrtc(self, pipeline: Gst.Pipeline) -> Gst.Element:
        self._logger.debug("Configuring WebRTC")
        webrtcsink = Gst.ElementFactory.make("webrtcsink")
        if not webrtcsink:
            raise RuntimeError(
                "Failed to create webrtcsink element. Is the GStreamer webrtc rust plugin installed?"
            )

        meta_structure = Gst.Structure.new_empty("meta")
        meta_structure.set_value("name", "reachymini")
        webrtcsink.set_property("meta", meta_structure)
        webrtcsink.set_property("run-signalling-server", True)

        # Disable FEC and retransmission — both add latency waiting for
        # lost-packet recovery. On Tailscale/LAN the loss rate is near zero;
        # the latency cost outweighs the reliability gain.
        webrtcsink.set_property("do-fec", False)
        webrtcsink.set_property("do-retransmission", False)
        # Disable GCC congestion control — it paces/holds packets to probe bandwidth,
        # adding 50–150ms on a predictable LAN/Tailscale link where bandwidth is not constrained.
        webrtcsink.set_property("congestion-control", 0)  # 0 = disabled

        webrtcsink.connect("consumer-added", self._consumer_added)

        pipeline.add(webrtcsink)

        return webrtcsink

    def _consumer_added(
        self, webrtcbin: Gst.Bin, arg1: Gst.Element, udata: bytes
    ) -> None:
        self._logger.info("consumer added")

        Gst.debug_bin_to_dot_file(
            self._pipeline_sender, Gst.DebugGraphDetails.ALL, "pipeline_full"
        )

        GLib.timeout_add_seconds(5, self._dump_latency)

    def _configure_receiver(self, pipeline: Gst.Pipeline) -> None:
        udpsrc = Gst.ElementFactory.make("udpsrc")
        udpsrc.set_property("port", 5000)
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=audio,encoding-name=OPUS,payload=96"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)
        rtpjitterbuffer = Gst.ElementFactory.make("rtpjitterbuffer")
        rtpjitterbuffer.set_property(
            "latency", 50
        )  # 50ms is sufficient on Tailscale/LAN; was 200ms
        rtpopusdepay = Gst.ElementFactory.make("rtpopusdepay")
        opusdec = Gst.ElementFactory.make("opusdec")
        queue = Gst.ElementFactory.make("queue")
        audioconvert = Gst.ElementFactory.make("audioconvert")
        audioresample = Gst.ElementFactory.make("audioresample")
        alsasink = Gst.ElementFactory.make("alsasink")
        alsasink.set_property(
            "device", "reachymini_audio_sink"
        )  # f"hw:{self._id_audio_card},0")
        alsasink.set_property("sync", False)

        pipeline.add(udpsrc)
        pipeline.add(capsfilter)
        pipeline.add(rtpjitterbuffer)
        pipeline.add(rtpopusdepay)
        pipeline.add(opusdec)
        pipeline.add(queue)
        pipeline.add(audioconvert)
        pipeline.add(audioresample)
        pipeline.add(alsasink)

        udpsrc.link(capsfilter)
        capsfilter.link(rtpjitterbuffer)
        rtpjitterbuffer.link(rtpopusdepay)
        rtpopusdepay.link(opusdec)
        opusdec.link(queue)
        queue.link(audioconvert)
        audioconvert.link(audioresample)
        audioresample.link(alsasink)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the current camera resolution as a tuple (width, height)."""
        return (self._resolution.value[0], self._resolution.value[1])

    @property
    def framerate(self) -> int:
        """Get the current camera framerate."""
        return self._resolution.value[2]

    def _configure_video(
        self, cam_path: str, pipeline: Gst.Pipeline, webrtcsink: Gst.Element
    ) -> None:
        self._logger.debug(f"Configuring video {cam_path}")
        if cam_path == "imx708":
            self._configure_video_rpicam(pipeline, webrtcsink)
        else:
            self._configure_video_v4l2(cam_path, pipeline, webrtcsink)

    def _configure_video_rpicam(
        self, pipeline: Gst.Pipeline, webrtcsink: Gst.Element
    ) -> None:
        """Configure video using rpicam-vid subprocess for IMX708 camera.

        Bypasses the broken v4l2h264enc GStreamer element by using rpicam-vid
        which accesses the hardware H.264 encoder directly via libcamera.
        The H.264 output is tee'd three ways: WebRTC, MPEG-TS TCP, and
        decoded raw frames for the unix socket (MJPEG HTTP).
        """
        if not shutil.which("rpicam-vid"):
            raise RuntimeError(
                "rpicam-vid not found. Required for IMX708 camera on this hardware."
            )

        width, height = self.resolution
        fps = self.framerate
        cmd = [
            "rpicam-vid",
            "-t", "0",                    # run forever
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--codec", "h264",
            "--profile", "baseline",       # constrained baseline for Safari/WebKit
            "--inline",                    # repeat SPS/PPS (matches repeat_sequence_header=1)
            "--low-latency",               # reduces encoder delay from ~8 frames to ~1 frame (~230ms win)
            "--bitrate", "5000000",        # 5 Mbps (matches v4l2h264enc video_bitrate)
            "--intra", "15",               # IDR every 15 frames (0.5s at 30fps) — faster connect/recovery
            "-o", "-",                     # output to stdout
        ]
        self._logger.info(f"Starting rpicam-vid: {' '.join(cmd)}")
        self._rpicam_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )

        # --- GStreamer elements ---
        fdsrc = Gst.ElementFactory.make("fdsrc")
        fdsrc.set_property("fd", self._rpicam_proc.stdout.fileno())

        h264parse = Gst.ElementFactory.make("h264parse")
        h264parse.set_property("config-interval", -1)  # send SPS/PPS with every IDR — faster browser decode start

        # Caps filter so webrtcsink knows it's receiving pre-encoded H.264
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter", "capsfilter_h264")
        capsfilter_h264.set_property("caps", caps_h264)

        h264_tee = Gst.ElementFactory.make("tee", "h264_tee")

        # Branch 1: WebRTC — leaky so old frames are dropped rather than buffered
        queue_webrtc = Gst.ElementFactory.make("queue", "queue_webrtc")
        queue_webrtc.set_property("max-size-buffers", 1)
        queue_webrtc.set_property("max-size-bytes", 0)
        queue_webrtc.set_property("max-size-time", 0)
        queue_webrtc.set_property("leaky", 2)  # drop oldest (downstream)

        # Branch 2: MPEG-TS over TCP for recording pipeline
        queue_tcp = Gst.ElementFactory.make("queue", "queue_tcp")
        h264parse_tcp = Gst.ElementFactory.make("h264parse", "h264parse_tcp")
        mpegtsmux = Gst.ElementFactory.make("mpegtsmux")
        tcpserversink = Gst.ElementFactory.make("tcpserversink")
        tcpserversink.set_property("host", "0.0.0.0")
        tcpserversink.set_property("port", 9001)
        tcpserversink.set_property("recover-policy", 3)  # keyframe
        tcpserversink.set_property("sync", False)  # don't let slow/absent clients backpressure the tee

        # Branch 3: Decode to raw BGR frames for MJPEG HTTP (via appsink)
        queue_decode = Gst.ElementFactory.make("queue", "queue_decode")
        avdec_h264 = Gst.ElementFactory.make("avdec_h264")
        if not avdec_h264:
            raise RuntimeError(
                "avdec_h264 not found. Install gstreamer1.0-libav: "
                "sudo apt install gstreamer1.0-libav"
            )
        videoconvert = Gst.ElementFactory.make("videoconvert")
        appsink = Gst.ElementFactory.make("appsink", "frame_appsink")
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        caps_bgr = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={width},height={height}"
        )
        appsink.set_property("caps", caps_bgr)
        self._appsink = appsink

        elements = [
            fdsrc, h264parse, capsfilter_h264, h264_tee,
            queue_webrtc,
            queue_tcp, h264parse_tcp, mpegtsmux, tcpserversink,
            queue_decode, avdec_h264, videoconvert, appsink,
        ]
        if not all(elements):
            self._cleanup_rpicam()
            raise RuntimeError("Failed to create GStreamer video elements for rpicam pipeline")

        for elem in elements:
            pipeline.add(elem)

        # Link: fdsrc → h264parse → capsfilter_h264 → h264_tee
        fdsrc.link(h264parse)
        h264parse.link(capsfilter_h264)
        capsfilter_h264.link(h264_tee)

        # Branch 1: WebRTC
        h264_tee.link(queue_webrtc)
        queue_webrtc.link(webrtcsink)

        # Branch 2: MPEG-TS over TCP
        h264_tee.link(queue_tcp)
        queue_tcp.link(h264parse_tcp)
        h264parse_tcp.link(mpegtsmux)
        mpegtsmux.link(tcpserversink)

        # Branch 3: Decode → raw BGR frames → appsink
        h264_tee.link(queue_decode)
        queue_decode.link(avdec_h264)
        avdec_h264.link(videoconvert)
        videoconvert.link(appsink)

        self._logger.info(
            "rpicam-vid pipeline configured: WebRTC + TCP:9001 + appsink"
        )

    def _configure_video_v4l2(
        self, cam_path: str, pipeline: Gst.Pipeline, webrtcsink: Gst.Element
    ) -> None:
        """Configure video using libcamerasrc + v4l2h264enc (original pipeline).

        Used for non-IMX708 cameras where v4l2h264enc works correctly.
        """
        camerasrc = Gst.ElementFactory.make("libcamerasrc")
        caps = Gst.Caps.from_string(
            f"video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.framerate}/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)
        tee = Gst.ElementFactory.make("tee")
        # Appsink branch: raw BGR frames for MJPEG HTTP
        appsink = Gst.ElementFactory.make("appsink", "frame_appsink")
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        caps_bgr = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.resolution[0]},height={self.resolution[1]}"
        )
        appsink.set_property("caps", caps_bgr)
        self._appsink = appsink
        queue_appsink = Gst.ElementFactory.make("queue", "queue_appsink")
        videoconvert_appsink = Gst.ElementFactory.make("videoconvert", "videoconvert_appsink")
        queue_encoder = Gst.ElementFactory.make("queue", "queue_encoder")
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        # doc: https://docs.qualcomm.com/doc/80-70014-50/topic/v4l2h264enc.html
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        extra_controls_structure.set_value("video_bitrate", 5_000_000)
        extra_controls_structure.set_value("h264_i_frame_period", 60)
        extra_controls_structure.set_value("video_gop_size", 256)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)
        # Use H264 Level 3.1 + Constrained Baseline for Safari/WebKit compatibility
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "level=(string)3.1,profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)

        # Tee the H.264 output: one branch to WebRTC, one to TCP for recording pipeline
        h264_tee = Gst.ElementFactory.make("tee", "h264_tee")
        queue_webrtc = Gst.ElementFactory.make("queue", "queue_webrtc")
        queue_webrtc.set_property("max-size-buffers", 1)
        queue_webrtc.set_property("max-size-bytes", 0)
        queue_webrtc.set_property("max-size-time", 0)
        queue_webrtc.set_property("leaky", 2)  # drop oldest (downstream)
        queue_tcp = Gst.ElementFactory.make("queue", "queue_tcp")
        h264parse = Gst.ElementFactory.make("h264parse")
        mpegtsmux = Gst.ElementFactory.make("mpegtsmux")
        tcpserversink = Gst.ElementFactory.make("tcpserversink")
        tcpserversink.set_property("host", "0.0.0.0")
        tcpserversink.set_property("port", 9001)
        # Recover from disconnected clients by starting from the next keyframe
        tcpserversink.set_property("recover-policy", 3)  # keyframe
        tcpserversink.set_property("sync", False)  # don't let slow/absent clients backpressure the tee

        if not all(
            [
                camerasrc,
                capsfilter,
                tee,
                queue_appsink,
                videoconvert_appsink,
                appsink,
                queue_encoder,
                v4l2h264enc,
                capsfilter_h264,
                h264_tee,
                queue_webrtc,
                queue_tcp,
                h264parse,
                mpegtsmux,
                tcpserversink,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer video elements")

        pipeline.add(camerasrc)
        pipeline.add(capsfilter)
        pipeline.add(tee)
        pipeline.add(queue_appsink)
        pipeline.add(videoconvert_appsink)
        pipeline.add(appsink)
        pipeline.add(queue_encoder)
        pipeline.add(v4l2h264enc)
        pipeline.add(capsfilter_h264)
        pipeline.add(h264_tee)
        pipeline.add(queue_webrtc)
        pipeline.add(queue_tcp)
        pipeline.add(h264parse)
        pipeline.add(mpegtsmux)
        pipeline.add(tcpserversink)

        camerasrc.link(capsfilter)
        capsfilter.link(tee)
        tee.link(queue_appsink)
        queue_appsink.link(videoconvert_appsink)
        videoconvert_appsink.link(appsink)
        tee.link(queue_encoder)
        queue_encoder.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(h264_tee)
        # Branch 1: WebRTC (existing)
        h264_tee.link(queue_webrtc)
        queue_webrtc.link(webrtcsink)
        # Branch 2: MPEG-TS over TCP for recording pipeline
        h264_tee.link(queue_tcp)
        queue_tcp.link(h264parse)
        h264parse.link(mpegtsmux)
        mpegtsmux.link(tcpserversink)

    def _configure_audio(self, pipeline: Gst.Pipeline, webrtcsink: Gst.Element) -> None:
        self._logger.debug("Configuring audio")

        alsasrc = Gst.ElementFactory.make("alsasrc")
        alsasrc.set_property("device", "reachymini_audio_src")
        # Reduce ALSA capture buffer from default (~1s) to 20ms for lower audio latency
        alsasrc.set_property("buffer-time", 20000)   # microseconds
        alsasrc.set_property("latency-time", 10000)  # microseconds

        if not all([alsasrc]):
            raise RuntimeError("Failed to create GStreamer audio elements")

        pipeline.add(alsasrc)
        alsasrc.link(webrtcsink)

    def _get_audio_input_device(self) -> Optional[str]:
        """Use Gst.DeviceMonitor to find the pipewire audio card.

        Returns the device ID of the found audio card, None if not.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Audio/Source")
        monitor.start()

        snd_card_name = "Reachy Mini Audio"

        devices = monitor.get_devices()
        for device in devices:
            name = device.get_display_name()
            device_props = device.get_properties()

            if snd_card_name in name:
                if device_props and device_props.has_field("object.serial"):
                    serial = device_props.get_string("object.serial")
                    self._logger.debug(f"Found audio input device with serial {serial}")
                    monitor.stop()
                    return str(serial)

        monitor.stop()
        self._logger.warning("No source audio card found.")
        return None

    def _get_video_device(self) -> Tuple[str, Optional[CameraSpecs]]:
        """Use Gst.DeviceMonitor to find the unix camera path /dev/videoX.

        Returns the device path (e.g., '/dev/video2'), or '' if not found.
        """
        monitor = Gst.DeviceMonitor()
        monitor.add_filter("Video/Source")
        monitor.start()

        cam_names = ["Reachy", "Arducam_12MP", "imx708"]

        devices = monitor.get_devices()
        for cam_name in cam_names:
            for device in devices:
                name = device.get_display_name()
                device_props = device.get_properties()

                if cam_name in name:
                    if device_props and device_props.has_field("api.v4l2.path"):
                        device_path = device_props.get_string("api.v4l2.path")
                        camera_specs = (
                            cast(CameraSpecs, ArducamSpecs)
                            if cam_name == "Arducam_12MP"
                            else cast(CameraSpecs, ReachyMiniLiteCamSpecs)
                        )
                        self._logger.debug(f"Found {cam_name} camera at {device_path}")
                        monitor.stop()
                        return str(device_path), camera_specs
                    elif cam_name == "imx708":
                        camera_specs = cast(CameraSpecs, ReachyMiniWirelessCamSpecs)
                        self._logger.debug(f"Found {cam_name} camera")
                        monitor.stop()
                        return cam_name, camera_specs
        monitor.stop()
        self._logger.warning("No camera found.")
        return "", None

    def _on_bus_message(self, bus: Gst.Bus, msg: Gst.Message, loop) -> bool:  # type: ignore[no-untyped-def]
        t = msg.type
        if t == Gst.MessageType.EOS:
            self._logger.warning("End-of-stream")
            return False

        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            self._logger.error(f"Error: {err} {debug}")
            return False

        else:
            # self._logger.warning(f"Unhandled message type: {t}")
            pass

        return True

    def start(self) -> None:
        """Start the WebRTC pipeline."""
        self._logger.debug("Starting WebRTC")
        self._pipeline_sender.set_state(Gst.State.PLAYING)
        self._pipeline_receiver.set_state(Gst.State.PLAYING)
        GLib.timeout_add_seconds(5, self._dump_latency)

    def pause(self) -> None:
        """Pause the WebRTC pipeline."""
        self._logger.debug("Pausing WebRTC")
        self._pipeline_sender.set_state(Gst.State.PAUSED)
        self._pipeline_receiver.set_state(Gst.State.PAUSED)

    def stop(self) -> None:
        """Stop the WebRTC pipeline."""
        self._logger.debug("Stopping WebRTC")

        self._pipeline_sender.set_state(Gst.State.NULL)
        self._pipeline_receiver.set_state(Gst.State.NULL)
        self._cleanup_rpicam()

    @property
    def is_running(self) -> bool:
        """Check if the sender pipeline is in PLAYING state."""
        _, state, _ = self._pipeline_sender.get_state(0)
        return state == Gst.State.PLAYING

    def read_frame(self) -> Optional[npt.NDArray[np.uint8]]:
        """Pull the latest video frame from the pipeline's appsink.

        Returns the frame as a BGR numpy array, or None if unavailable.
        Thread-safe: can be called from any thread while GLib MainLoop
        runs on another.
        """
        if self._appsink is None:
            return None

        sample = self._appsink.try_pull_sample(Gst.SECOND)  # 1s timeout
        if sample is None:
            return None

        buf = sample.get_buffer()
        if buf is None:
            return None

        data = buf.extract_dup(0, buf.get_size())
        width, height = self.resolution
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))


if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    webrtc = GstWebRTC(log_level="DEBUG")
    webrtc.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("User interrupted")
    finally:
        webrtc.stop()
        webrtc.__del__()
