import video_lktrackHSV
# import matplotlib.pyplot as plt
import cv2

imnames = 'DJI_001.mp4'

# Track using the LKTracker generator
lkt = video_lktrackHSV.LKTracker(imnames)
for i, f in lkt.track():
    continue

cv2.destroyAllWindows()
video_lktrackHSV.LKTracker.capture.release()
video_lktrackHSV.LKTracker.writer.release()
