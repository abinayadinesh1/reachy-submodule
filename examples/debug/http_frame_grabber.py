"""Grab camera frames from Reachy Mini via HTTP.

Works on any platform -- no GStreamer required.
The daemon must be running with --wireless-version on the Pi,
and the backend must be started (WebRTC active).

Usage:
    python http_frame_grabber.py --host reachy-mini.local
    python http_frame_grabber.py --host reachy-mini.local --save snapshot.jpg
    python http_frame_grabber.py --host reachy-mini.local --stream
    python http_frame_grabber.py --host reachy-mini.local --status
"""

import argparse
import time

import cv2
import numpy as np
import requests


def check_status(host: str, port: int = 8000) -> dict:
    """Check camera and daemon status."""
    print(f"Checking {host}:{port}...")

    try:
        r = requests.get(f"http://{host}:{port}/api/daemon/status", timeout=5)
        r.raise_for_status()
        daemon = r.json()
        print(f"  Daemon state: {daemon['state']}")
        print(f"  Wireless version: {daemon['wireless_version']}")
        print(f"  Version: {daemon.get('version', 'unknown')}")
        if daemon.get("backend_status"):
            bs = daemon["backend_status"]
            print(f"  Backend ready: {bs.get('ready')}")
            print(f"  Motor mode: {bs.get('motor_control_mode')}")
            freq = bs.get("control_loop_stats", {}).get("mean_control_loop_frequency")
            if freq:
                print(f"  Control loop: {freq:.1f} Hz")
        if daemon.get("error"):
            print(f"  ERROR: {daemon['error']}")
    except Exception as e:
        print(f"  Daemon status failed: {e}")
        return {}

    try:
        r = requests.get(f"http://{host}:{port}/api/camera/status", timeout=5)
        r.raise_for_status()
        cam = r.json()
        print(f"  Camera available: {cam['available']}")
        print(f"  Camera socket exists: {cam['socket_exists']}")
        return cam
    except Exception as e:
        print(f"  Camera status failed: {e}")
        return {}


def grab_frame(host: str, port: int = 8000, quality: int = 80) -> np.ndarray | None:
    """Grab a single JPEG frame and return as numpy array (BGR)."""
    url = f"http://{host}:{port}/api/camera/frame?quality={quality}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def stream_frames(host: str, port: int = 8000, quality: int = 60, fps: float = 10) -> None:
    """Stream MJPEG frames and display with OpenCV."""
    url = f"http://{host}:{port}/api/camera/stream?quality={quality}&fps={fps}"
    print(f"Streaming from {url}  (press 'q' to quit)")

    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()

    buf = b""
    for chunk in resp.iter_content(chunk_size=8192):
        buf += chunk
        # Find JPEG start (FFD8) and end (FFD9) markers
        start = buf.find(b"\xff\xd8")
        end = buf.find(b"\xff\xd9")
        if start != -1 and end != -1 and end > start:
            jpg = buf[start : end + 2]
            buf = buf[end + 2 :]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("Reachy Mini Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP camera frame grabber for Reachy Mini")
    parser.add_argument("--host", default="reachy-mini.local", help="Robot hostname or IP")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI port (default: 8000)")
    parser.add_argument("--save", type=str, default=None, help="Save single frame to file path")
    parser.add_argument("--stream", action="store_true", help="Stream and display frames live")
    parser.add_argument("--status", action="store_true", help="Check daemon and camera status")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS for streaming (default: 10)")
    args = parser.parse_args()

    if args.status:
        check_status(args.host, args.port)
    elif args.stream:
        stream_frames(args.host, args.port, args.quality, args.fps)
    else:
        frame = grab_frame(args.host, args.port, args.quality)
        if frame is not None:
            filename = args.save or "reachy_mini_frame.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame ({frame.shape[1]}x{frame.shape[0]}) to {filename}")
        else:
            print("Failed to decode frame")


if __name__ == "__main__":
    main()
