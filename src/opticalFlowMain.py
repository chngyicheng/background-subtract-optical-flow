#!/usr/bin/env python
# import video_lktrackHSV

import MovingTracker
# import matplotlib.pyplot as plt
import cv2

imnames = 'DJI_004.mp4'

# Track using the LKTracker generator
lkt = MovingTracker.LKDenseTracker(imnames)
for i, f in lkt.track():
    continue

cv2.destroyAllWindows()
# video_lktrackHSV.LKTracker.capture.release()
# video_lktrackHSV.LKTracker.writer.release()
MovingTracker.LKDenseTracker.capture.release()
MovingTracker.LKDenseTracker.writer.release()
