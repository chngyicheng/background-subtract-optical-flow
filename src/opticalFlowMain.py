#!/usr/bin/env python
# import video_lktrackHSV

import MovingTracker
# import matplotlib.pyplot as plt
import cv2

imnames = 'DJI_005.mp4'

# Track using the LKTracker generator
lkt = MovingTracker.Detect(imnames)
for i, f in lkt.track():
    continue

cv2.destroyAllWindows()
# video_lktrackHSV.LKTracker.capture.release()
# video_lktrackHSV.LKTracker.writer.release()
MovingTracker.Detect.capture.release()
MovingTracker.Detect.writer.release()
