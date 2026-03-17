"""Camera-related API routes.

Provides HTTP endpoints for grabbing camera frames and streaming
MJPEG video from the Reachy Mini's camera. Reads directly from
the appsink in the WebRTC daemon's GStreamer pipeline.
"""

import asyncio
import logging
import threading
import time
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/camera")

logger = logging.getLogger(__name__)

# Cache the most recent JPEG encode per quality level so multiple clients
# viewing the same stream don't each trigger a redundant cv2.imencode call.
_jpeg_cache_lock = threading.Lock()
_jpeg_cache: dict[int, tuple[float, bytes]] = {}  # quality -> (timestamp, jpeg_bytes)
_JPEG_CACHE_MAX_AGE = 0.030  # 30ms — serve cached frame if younger than this


def _read_frame(request: Request) -> Optional[npt.NDArray[np.uint8]]:
    """Read a single BGR frame from the WebRTC daemon's appsink."""
    daemon = request.app.state.daemon
    if daemon._webrtc is None:
        return None
    return daemon._webrtc.read_frame()


def _encode_frame(
    frame: npt.NDArray[np.uint8], quality: int
) -> Optional[bytes]:
    """JPEG-encode a frame, returning cached bytes when possible."""
    now = time.monotonic()

    with _jpeg_cache_lock:
        cached = _jpeg_cache.get(quality)
        if cached is not None and (now - cached[0]) < _JPEG_CACHE_MAX_AGE:
            return cached[1]

    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        return None
    jpeg_bytes = jpeg.tobytes()

    with _jpeg_cache_lock:
        _jpeg_cache[quality] = (now, jpeg_bytes)

    return jpeg_bytes


@router.get("/status")
async def get_camera_status(request: Request) -> dict:
    """Check camera availability."""
    daemon = request.app.state.daemon
    webrtc = daemon._webrtc
    webrtc_running = webrtc is not None and webrtc.is_running
    mjpeg_available = webrtc is not None and webrtc._appsink is not None
    return {
        "webrtc_available": webrtc_running,
        "mjpeg_available": mjpeg_available and webrtc_running,
    }


@router.get("/frame")
async def get_camera_frame(request: Request, quality: int = 80) -> Response:
    """Grab a single JPEG frame from the camera.

    Args:
        quality: JPEG compression quality (1-100). Default: 80.
    """
    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _read_frame, request)
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="No frame available. Is the daemon running with --wireless-version?",
        )

    quality = max(1, min(100, quality))
    jpeg_bytes = await loop.run_in_executor(None, _encode_frame, frame, quality)
    if jpeg_bytes is None:
        raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG")

    return Response(content=jpeg_bytes, media_type="image/jpeg")


async def _mjpeg_generator(
    request: Request, quality: int, fps: float
) -> AsyncGenerator[bytes, None]:
    """Generate MJPEG frames for streaming.

    Blocking calls (_read_frame, _encode_frame) are offloaded to the
    default thread-pool executor so multiple generators can run
    concurrently without serialising on the asyncio event loop.
    """
    period = 1.0 / fps
    loop = asyncio.get_event_loop()
    while True:
        frame = await loop.run_in_executor(None, _read_frame, request)
        if frame is not None:
            jpeg_bytes = await loop.run_in_executor(
                None, _encode_frame, frame, quality
            )
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
        await asyncio.sleep(period)


@router.get("/stream")
async def get_camera_stream(
    request: Request, quality: int = 60, fps: float = 10.0
) -> StreamingResponse:
    """Stream MJPEG video from the camera.

    Viewable directly in a browser. Use an <img> tag or open the URL directly.

    Args:
        quality: JPEG compression quality (1-100). Default: 60.
        fps: Target frames per second. Default: 10.
    """
    fps = max(1.0, min(30.0, fps))
    quality = max(1, min(100, quality))
    return StreamingResponse(
        _mjpeg_generator(request, quality, fps),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Accel-Buffering": "no",
        },
    )
