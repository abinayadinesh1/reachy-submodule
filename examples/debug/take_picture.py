"""Take a picture using Reachy Mini's camera.

Connects to the robot, grabs a single frame, and saves it as a JPEG file.
Falls back to a local camera if the robot's camera is unavailable
(e.g. WebRTC producer not running on a wireless robot).

Note: The daemon must be running before executing this script.
"""

import argparse
import time

import cv2

from reachy_mini import ReachyMini


def main(backend: str) -> None:
    """Get a frame and take a picture."""
    try:
        reachy_mini = ReachyMini(media_backend=backend)
    except KeyError as e:
        if "Producer" in str(e):
            print(f"WebRTC error: {e}")
            print("The robot's camera WebRTC producer is not available.")
            print("Falling back to local camera...")
            _capture_local_fallback()
            return
        raise
    except Exception as e:
        print(f"Failed to connect to Reachy Mini: {e}")
        raise

    with reachy_mini:
        frame = reachy_mini.media.get_frame()
        start_time = time.time()
        while frame is None:
            if time.time() - start_time > 20:
                print("Timeout: Failed to grab frame within 20 seconds.")
                exit(1)
            print("Failed to grab frame. Retrying...")
            frame = reachy_mini.media.get_frame()
            time.sleep(1)

        cv2.imwrite("reachy_mini_picture.jpg", frame)
        print("Saved frame as reachy_mini_picture.jpg")


def _capture_local_fallback() -> None:
    """Try to capture a frame from a local camera as fallback."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No local camera available either. Cannot take a picture.")
        return
    # Allow camera to warm up
    time.sleep(0.5)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("reachy_mini_picture.jpg", frame)
        print("Saved frame from local camera as reachy_mini_picture.jpg")
    else:
        print("Failed to capture frame from local camera.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display Reachy Mini's camera feed and make it look at clicked points."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["default", "gstreamer", "webrtc"],
        default="default",
        help="Media backend to use.",
    )

    args = parser.parse_args()
    main(backend=args.backend)
