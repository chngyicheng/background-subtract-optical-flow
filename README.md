# Foreground and Background Segmentation using Farneback Optical Flow

This project was my Final Year Project in the National University of Singapore. Many functionalities are still not present and are to be added in the future. A legacy file of using the Lucas-Kanade Optical Flow has also been left in the `src/` directory for future reference and experimentation.

 
## Overview
This repository attempts to segment the foreground from the background in post-processing using OpenCV's Farneback Optical flow. This method works by:

1. Splitting the video input into a set of 5 x 5 boxes, whilst ignoring the top two rows by taking the assumption of it being the sky.

2. The average optical flow of each box is then calculated and assumed to be the background. The pixels' optical flow magnitudes and directions are then compared to this average, and if the pixels fall outside of a certain threshold value, it is taken to be the foreground (because it is NOT the background).

3. For future implementations' sake, the background flow directions are then extrapolated to obtain the vanishing point (camera's moving direction). This could prove useful for obtaining heading direction in the future.

## Usage
Make sure you have OpenCV installed. Then, follow the following steps:

1. Place the video you want to run this method into the `src/` directory. 

```bash
imnames = 'VIDEO_INPUT_NAME_HERE.mp4'
```

2. Then, within MovingTracker.py, edit the output video names accordingly. For example:

``` bash
self.writer1    = cv2.VideoWriter('VIDEO_OUTPUT_NAME_HERE.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (self.width, self.height))         # Displays the video output
self.writer2    = cv2.VideoWriter('FARNEBACK_VIDEO_NAME.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (self.width, int(self.height))))     # Displays the farneback output
```

3. Once the setup is done, run the following command in your terminal:
```bash
$ python3 opticalFlowMain.py
```

## Sample Video Output
The GIF shows the segmentation and heading estimation after video post-processing.
<p align="center">
  <img src="demo/OpFlow_Segment_Output.gif" height=300>
</p>  
