"""Camera-related API routes.

Provides HTTP endpoints for grabbing camera frames and streaming
MJPEG video from the Reachy Mini's camera. Works by reading from
the unix socket that the WebRTC daemon creates at /tmp/reachymini_camera_socket.
"""

import asyncio
import logging
import os
import threading
import time
from typing import AsyncGenerator

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from reachy_mini.daemon.utils import CAMERA_SOCKET_PATH

router = APIRouter(prefix="/camera")

logger = logging.getLogger(__name__)
_camera_lock = threading.Lock()


def _get_camera(request: Request):
    """Get or lazily create the GStreamerCamera singleton."""
    if not hasattr(request.app.state, "_camera_reader") or request.app.state._camera_reader is None:
        with _camera_lock:
            if not hasattr(request.app.state, "_camera_reader") or request.app.state._camera_reader is None:
                if not os.path.exists(CAMERA_SOCKET_PATH):
                    raise HTTPException(
                        status_code=503,
                        detail=f"Camera socket not found at {CAMERA_SOCKET_PATH}. "
                        "Is the daemon running with --wireless-version and has WebRTC started?",
                    )
                from reachy_mini.media.camera_gstreamer import GStreamerCamera

                cam = GStreamerCamera()
                cam.open()
                # Warmup: give pipeline time to produce first frame
                for i in range(10):
                    if cam.read() is not None:
                        logger.info("Camera warmed up after %d reads", i + 1)
                        break
                    time.sleep(0.1)
                request.app.state._camera_reader = cam
                logger.info("Camera reader initialized from unix socket")
    return request.app.state._camera_reader


@router.get("/status")
async def get_camera_status() -> dict:
    """Check camera availability and configuration."""
    socket_exists = os.path.exists(CAMERA_SOCKET_PATH)
    return {
        "available": socket_exists,
        "socket_path": CAMERA_SOCKET_PATH,
        "socket_exists": socket_exists,
    }


@router.get("/frame")
async def get_camera_frame(request: Request, quality: int = 80) -> Response:
    """Grab a single JPEG frame from the camera.

    Args:
        quality: JPEG compression quality (1-100). Default: 80.
    """
    camera = _get_camera(request)
    frame = camera.read()
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available from camera")

    quality = max(1, min(100, quality))
    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame as JPEG")

    return Response(content=jpeg.tobytes(), media_type="image/jpeg")


async def _mjpeg_generator(
    request: Request, quality: int, fps: float
) -> AsyncGenerator[bytes, None]:
    """Generate MJPEG frames for streaming."""
    camera = _get_camera(request)
    period = 1.0 / fps
    while True:
        frame = camera.read()
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
    )
