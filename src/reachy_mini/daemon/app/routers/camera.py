"""Camera-related API routes.

Provides HTTP endpoints for grabbing camera frames and streaming
MJPEG video from the Reachy Mini's camera. Reads directly from
the appsink in the WebRTC daemon's GStreamer pipeline.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/camera")

logger = logging.getLogger(__name__)


def _read_frame(request: Request) -> Optional[npt.NDArray[np.uint8]]:
    """Read a single BGR frame from the WebRTC daemon's appsink."""
    daemon = request.app.state.daemon
    if daemon._webrtc is None:
        return None
    return daemon._webrtc.read_frame()


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
    frame = _read_frame(request)
    if frame is None:
        raise HTTPException(
            status_code=503,
            detail="No frame available. Is the daemon running with --wireless-version?",
        )

    quality = max(1, min(100, quality))
    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG")

    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


async def _mjpeg_generator(
    request: Request, quality: int, fps: float
) -> AsyncGenerator[bytes, None]:
    """Generate MJPEG frames for streaming."""
    period = 1.0 / fps
    while True:
        frame = _read_frame(request)
        if frame is not None:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
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
