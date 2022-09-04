# Moving Object Detection with a Moving Monocular Camera using Optical Flow

This project was my Final Year Project in the National University of Singapore. Many functionalities are still not present and are to be added in the future. A legacy file of using the Lucas-Kanade Optical Flow has also been left in the `src/` directory for future reference and experimentation.

 
## Overview
With the current background subtraction techniques, a static camera is required for the usual background subtraction techniques. For a moving camera, a new method had to be explored.


This repository attempts to utilise OpenCV's Farneback Optical Flow in this new way: background subtraction and foreground object detection, performed in post-processing.


This works by the following methods:

1. The video input is split into a set of 5 x 5 boxes, then assuming the top two rows of the video feed being the sky and ignoring it to increase computational speed.

2. The average optical flow of each box is then calculated and assumed to be the background. 

3. Each pixels' optical flow magnitudes and directions are then compared to this average, and if the pixels fall outside of a certain threshold value, it is taken to be the foreground (because it is NOT the background).

4. The background flow directions are then extrapolated to obtain the vanishing point (camera's moving direction). This was done for future development's sake of obtaining the camera's heading.

## Usage
Make sure you have OpenCV installed. Then, follow the following steps:

1. Place the video you want to run this method into the `src/` directory. Then in `src/opticalFlowMain.py`, edit the following lines to input the video title as input:

```bash
imnames = 'VIDEO_INPUT_NAME_HERE.mp4'
```

2. Next, within `src/MovingTracker.py`, edit the output video names accordingly (videos will be saved as these titles). For example:

``` bash
self.writer1    = cv2.VideoWriter('VIDEO_OUTPUT_NAME_HERE.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (self.width, self.height))         # Displays the video output
self.writer2    = cv2.VideoWriter('FARNEBACK_VIDEO_NAME.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (self.width, int(self.height))))     # Displays the farneback output
```

3. Once the setup is done, to run the script, enter the following command into your terminal:
```bash
$ python3 opticalFlowMain.py
```


The background subtraction will now happen frame by frame and once finished, will output the subtracted video with the object detected shown with a bounding box around it.
## Sample Video Output
The GIF is shows the object detection, background subtraction, and heading estimation performed for a video taken from a moving Unmanned Aerial Vehicle (UAV) following another UAV.
<p align="center">
  <img src="demo/OpFlow_Segment_Output.gif" height=300>
</p>  
