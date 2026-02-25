"""GStreamer camera/media pipeline.

Captures video from the robot's camera and exposes it via:
- An appsink for raw BGR frames (consumed by aiortc WebRTC + MJPEG HTTP)
- MPEG-TS over TCP on port 9001 (consumed by the recording pipeline)
- A UDP receiver for incoming audio playback
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
    """GStreamer camera pipeline with appsink + TCP recording."""

    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
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

        self._configure_video(cam_path, self._pipeline_sender)

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
        self, cam_path: str, pipeline: Gst.Pipeline
    ) -> None:
        self._logger.debug(f"Configuring video {cam_path}")
        if cam_path == "imx708":
            self._configure_video_rpicam(pipeline)
        else:
            self._configure_video_v4l2(cam_path, pipeline)

    def _configure_video_rpicam(self, pipeline: Gst.Pipeline) -> None:
        """Configure video using rpicam-vid subprocess for IMX708 camera.

        Bypasses the broken v4l2h264enc GStreamer element by using rpicam-vid
        which accesses the hardware H.264 encoder directly via libcamera.
        The H.264 output is tee'd two ways: MPEG-TS TCP (recording) and
        decoded raw frames via appsink (WebRTC + MJPEG HTTP).
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
            "--bitrate", "5000000",        # 5 Mbps
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
        h264parse.set_property("config-interval", -1)  # send SPS/PPS with every IDR

        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter", "capsfilter_h264")
        capsfilter_h264.set_property("caps", caps_h264)

        h264_tee = Gst.ElementFactory.make("tee", "h264_tee")

        # Branch 1: MPEG-TS over TCP for recording pipeline
        queue_tcp = Gst.ElementFactory.make("queue", "queue_tcp")
        h264parse_tcp = Gst.ElementFactory.make("h264parse", "h264parse_tcp")
        mpegtsmux = Gst.ElementFactory.make("mpegtsmux")
        tcpserversink = Gst.ElementFactory.make("tcpserversink")
        tcpserversink.set_property("host", "0.0.0.0")
        tcpserversink.set_property("port", 9001)
        tcpserversink.set_property("recover-policy", 3)  # keyframe
        tcpserversink.set_property("sync", False)

        # Branch 2: Decode to raw BGR frames for appsink (WebRTC + MJPEG HTTP)
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

        # Branch 1: MPEG-TS over TCP
        h264_tee.link(queue_tcp)
        queue_tcp.link(h264parse_tcp)
        h264parse_tcp.link(mpegtsmux)
        mpegtsmux.link(tcpserversink)

        # Branch 2: Decode → raw BGR frames → appsink
        h264_tee.link(queue_decode)
        queue_decode.link(avdec_h264)
        avdec_h264.link(videoconvert)
        videoconvert.link(appsink)

        self._logger.info(
            "rpicam-vid pipeline configured: TCP:9001 + appsink"
        )

    def _configure_video_v4l2(
        self, cam_path: str, pipeline: Gst.Pipeline
    ) -> None:
        """Configure video using libcamerasrc + v4l2h264enc.

        Used for non-IMX708 cameras where v4l2h264enc works correctly.
        Raw YUY2 is tee'd: one branch to appsink (WebRTC + MJPEG HTTP),
        one branch to H.264 encode → MPEG-TS TCP (recording).
        """
        camerasrc = Gst.ElementFactory.make("libcamerasrc")
        caps = Gst.Caps.from_string(
            f"video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.framerate}/1,format=YUY2,colorimetry=bt709,interlace-mode=progressive"
        )
        capsfilter = Gst.ElementFactory.make("capsfilter")
        capsfilter.set_property("caps", caps)
        tee = Gst.ElementFactory.make("tee")

        # Branch 1: Appsink — raw BGR frames for WebRTC (aiortc) + MJPEG HTTP
        queue_appsink = Gst.ElementFactory.make("queue", "queue_appsink")
        videoconvert_appsink = Gst.ElementFactory.make("videoconvert", "videoconvert_appsink")
        appsink = Gst.ElementFactory.make("appsink", "frame_appsink")
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)
        caps_bgr = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={self.resolution[0]},height={self.resolution[1]}"
        )
        appsink.set_property("caps", caps_bgr)
        self._appsink = appsink

        # Branch 2: H.264 encode → MPEG-TS over TCP for recording pipeline
        queue_encoder = Gst.ElementFactory.make("queue", "queue_encoder")
        v4l2h264enc = Gst.ElementFactory.make("v4l2h264enc")
        extra_controls_structure = Gst.Structure.new_empty("extra-controls")
        extra_controls_structure.set_value("repeat_sequence_header", 1)
        extra_controls_structure.set_value("video_bitrate", 5_000_000)
        extra_controls_structure.set_value("h264_i_frame_period", 60)
        extra_controls_structure.set_value("video_gop_size", 256)
        v4l2h264enc.set_property("extra-controls", extra_controls_structure)
        caps_h264 = Gst.Caps.from_string(
            "video/x-h264,stream-format=byte-stream,alignment=au,"
            "level=(string)3.1,profile=(string)constrained-baseline"
        )
        capsfilter_h264 = Gst.ElementFactory.make("capsfilter")
        capsfilter_h264.set_property("caps", caps_h264)
        h264parse = Gst.ElementFactory.make("h264parse")
        mpegtsmux = Gst.ElementFactory.make("mpegtsmux")
        tcpserversink = Gst.ElementFactory.make("tcpserversink")
        tcpserversink.set_property("host", "0.0.0.0")
        tcpserversink.set_property("port", 9001)
        tcpserversink.set_property("recover-policy", 3)  # keyframe
        tcpserversink.set_property("sync", False)

        if not all(
            [
                camerasrc, capsfilter, tee,
                queue_appsink, videoconvert_appsink, appsink,
                queue_encoder, v4l2h264enc, capsfilter_h264,
                h264parse, mpegtsmux, tcpserversink,
            ]
        ):
            raise RuntimeError("Failed to create GStreamer video elements")

        for elem in [
            camerasrc, capsfilter, tee,
            queue_appsink, videoconvert_appsink, appsink,
            queue_encoder, v4l2h264enc, capsfilter_h264,
            h264parse, mpegtsmux, tcpserversink,
        ]:
            pipeline.add(elem)

        camerasrc.link(capsfilter)
        capsfilter.link(tee)

        # Branch 1: raw BGR → appsink
        tee.link(queue_appsink)
        queue_appsink.link(videoconvert_appsink)
        videoconvert_appsink.link(appsink)

        # Branch 2: H.264 → MPEG-TS over TCP
        tee.link(queue_encoder)
        queue_encoder.link(v4l2h264enc)
        v4l2h264enc.link(capsfilter_h264)
        capsfilter_h264.link(h264parse)
        h264parse.link(mpegtsmux)
        mpegtsmux.link(tcpserversink)

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
