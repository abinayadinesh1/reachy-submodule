"""Minimal aiortc-based WebRTC video endpoint.

Browser POSTs an SDP offer to /api/webrtc/offer, gets back an SDP answer.
No WebSocket signaling, no STUN/TURN, no GStreamer webrtcsink.

Test it at http://<robot-ip>:8000/api/webrtc/test
"""

import asyncio
import logging
from typing import Set

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import VideoFrame
from aiortc.mediastreams import VideoStreamTrack
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(prefix="/webrtc")
logger = logging.getLogger(__name__)

# Track active peer connections for cleanup
peer_connections: Set[RTCPeerConnection] = set()


class CameraTrack(VideoStreamTrack):
    """VideoStreamTrack that reads frames from the GStreamer appsink."""

    kind = "video"

    def __init__(self, request: Request, fps: int = 30) -> None:
        super().__init__()
        self._request = request
        self._fps = fps
        self._period = 1.0 / fps

    async def recv(self) -> VideoFrame:
        # Pace frames at target fps
        pts, time_base = await self.next_timestamp()

        # Read frame in executor to avoid blocking asyncio loop
        loop = asyncio.get_event_loop()
        frame = await loop.run_in_executor(
            None, self._read_frame
        )

        if frame is None:
            # Return black frame if camera unavailable
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def _read_frame(self) -> np.ndarray | None:
        daemon = self._request.app.state.daemon
        if daemon._webrtc is None:
            return None
        return daemon._webrtc.read_frame()


@router.post("/offer")
async def webrtc_offer(request: Request) -> JSONResponse:
    """Handle WebRTC SDP offer and return answer.

    Browser sends: {"sdp": "...", "type": "offer"}
    Server returns: {"sdp": "...", "type": "answer"}
    """
    params = await request.json()

    pc = RTCPeerConnection()
    peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        logger.info("WebRTC connection state: %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            peer_connections.discard(pc)

    # Add video track
    pc.addTrack(CameraTrack(request))

    # Set remote offer and create answer
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return JSONResponse(
        content={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


@router.get("/test")
async def webrtc_test_page() -> HTMLResponse:
    """Minimal test page for the aiortc WebRTC stream."""
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html>
<head>
  <title>WebRTC Test</title>
  <style>
    body { margin: 0; background: #111; display: flex; flex-direction: column;
           align-items: center; justify-content: center; height: 100vh; color: #eee;
           font-family: system-ui; }
    video { max-width: 95vw; max-height: 85vh; background: #000; border-radius: 8px; }
    #status { margin: 12px 0; font-size: 14px; opacity: 0.7; }
    button { padding: 8px 20px; font-size: 14px; cursor: pointer; border-radius: 4px;
             border: 1px solid #555; background: #333; color: #eee; }
    button:hover { background: #444; }
  </style>
</head>
<body>
  <div id="status">Click Connect to start</div>
  <video id="video" autoplay playsinline muted></video>
  <br>
  <button onclick="connect()">Connect</button>
  <script>
    const video = document.getElementById('video');
    const status = document.getElementById('status');

    async function connect() {
      status.textContent = 'Connecting...';
      try {
        const pc = new RTCPeerConnection();

        pc.ontrack = (e) => {
          video.srcObject = e.streams[0];
          status.textContent = 'Live';
          // Minimize jitter buffer for low latency (Chrome 123+)
          e.receiver.jitterBufferTarget = 0;
        };

        pc.onconnectionstatechange = () => {
          status.textContent = 'State: ' + pc.connectionState;
          if (pc.connectionState === 'failed') {
            status.textContent = 'Connection failed — click Connect to retry';
          }
        };

        pc.addTransceiver('video', { direction: 'recvonly' });

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const resp = await fetch('/api/webrtc/offer', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
        });
        const answer = await resp.json();
        await pc.setRemoteDescription(answer);
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
        console.error(err);
      }
    }
  </script>
</body>
</html>"""
    )


async def cleanup_peer_connections() -> None:
    """Close all active peer connections. Called during shutdown."""
    coros = [pc.close() for pc in peer_connections]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)
    peer_connections.clear()
