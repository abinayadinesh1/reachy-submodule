from reachy_mini import ReachyMini
import cv2
with ReachyMini(media_backend="default") as reachy_mini:
    frame = reachy_mini.media.get_frame()
    cv2.imwrite(frame, "pic.png")

