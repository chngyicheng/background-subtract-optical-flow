import cv2
import numpy as np
import copy

# Some constraints and default parameters

# Lucas-Kanade optical flow params
lk_params = dict(winSize=(15,15), maxLevel=2,
	criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

subpix_params = dict(zeroZone=(-1,-1), winSize=(10,10),
	criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))

# Shi-Tomasi corner detection params
feature_params = dict(maxCorners=50, qualityLevel=0.001, minDistance=50, blockSize=20)

# Create some random colors
color = np.random.randint(0,255,(100,3))

class LKTracker(object):
	""" Class for Lucas-Kanade tracking with
	pyramidal optical flow"""

	def __init__(self, imnames):
		# Initialise with a list of image names
		
		self.imnames         = imnames
		self.features        = []
		self.prev_features   = []
		self.tracks          = []
		self.current_frame   = 0
		self.detect_interval = 5
		self.current_x       = 0
		self.current_y       = 0

		## For thresholding
		self.threshold = 60  # BINARY threshold
		self.blurValue = 5  # GaussianBlur parameter
		self.bgSubThreshold = 50
		self.learningRate = 0.007
		self.isBgCaptured = 0   # whether the background captured
		self.bgModel = None
		
	def detectPoints(self):
		"""Detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""

		# Load image and threshold image
		self.capture = cv2.VideoCapture(self.imnames)
		width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
		# self.writer = cv2.VideoWriter('drone_track_2.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (width, height))
		__, self.frame = self.capture.read()
		for i in range(5):
			__, self.frame = self.capture.read()
			self.bgThreshold()

		# Search for good points
		self.prev_features = cv2.goodFeaturesToTrack(self.drawing, **feature_params)

		# Refine the corner locations
		cv2.cornerSubPix(self.drawing, self.prev_features, **subpix_params)

		self.features = self.prev_features
		self.tracks   = [[p] for p in self.prev_features.reshape((-1,2))]
		
		self.prev_thresh = self.thresh
		self.prev_gray   = self.gray
		self.prev_drawing = self.drawing
		# cv2.imshow('test', self.prev_drawing)
		
		self.mask        = np.zeros_like(self.frame)
		
		self.draw()
		


	def bgThreshold(self):

		#  Main operation
		self.img = cv2.bilateralFilter(self.frame, 5, 50, 100)  # smoothening filter
		if self.isBgCaptured == 0:  # this part wont run until background captured
			self.bgModel = cv2.createBackgroundSubtractorMOG2(600, self.bgSubThreshold, False)
			self.isBgCaptured = 1
			print( 'Background Captured')
		# cv2.rectangle(self.img, (int(1 * self.img.shape[1]), (int(0.5 * self.img.shape[1]))),
        #          (self.img.shape[1], int(1 * self.img.shape[0])), (255, 0, 0), 2) #drawing ROI
		# cv2.imshow('original', self.img)
		self.img = self.removeBG(self.img)
		# img = img[0:int(0.5 * 50),
		# 	0:self.frame.shape[1]]  # clip the ROI
		

		# cv2.imshow('mask', img)

		# convert the image into binary image
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.blur = cv2.GaussianBlur(self.gray, (self.blurValue, self.blurValue), 0)
		# cv2.imshow('blur', self.blur)
		ret, self.thresh = cv2.threshold(self.blur, self.threshold, 255, cv2.THRESH_BINARY) #thresholding the frame
		cv2.imshow('thresh', self.thresh)
		

		# get the contours
		self.thresh1 = copy.deepcopy(self.thresh)
		self.contours, hierarchy = cv2.findContours(self.thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detecting contours
		length = len(self.contours)
		maxArea = -1
		if length > 0:
			for i in range(length):  # find the biggest contour (according to area)
				temp = self.contours[i]
				area = cv2.contourArea(temp)
				if area > maxArea:
					maxArea = area
					ci = i

			res = self.contours[ci]
			drawing = np.zeros_like(self.frame)
			cv2.drawContours(drawing, [res], 0, (255, 255, 255), thickness=-1) #drawing contours
			self.drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
			# cv2.imshow('draw', self.drawing)
			

	def removeBG(self, frame): #Subtracting the background
		fgmask = self.bgModel.apply(frame,learningRate=self.learningRate)

		kernel = np.ones((3, 3), np.uint8)
		fgmask = cv2.erode(fgmask, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=fgmask)
		return res
	
	def checkPoints(self):
		"""Re-detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""
		
		# # Search for good points
		self.features = cv2.goodFeaturesToTrack(self.drawing, **feature_params)

		# Refine the corner locations
		cv2.cornerSubPix(self.drawing, self.features, **subpix_params)

		# self.tracks = [[p] for p in self.features.reshape((-1,2))]
		

	def trackPoints(self):
		"""Track the detected features"""

		# Load the image and create grayscale
		__, self.frame = self.capture.read()
		self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		# self.bgThreshold()

		"""To be used later, re-checks detection every few frames"""
		# if self.current_frame % self.detect_interval == 0:
		# 	self.checkPoints()

		

		# Reshape to fit input format
		tmp = np.float32(self.features).reshape(-1, 1, 2)

		self.prev_features = self.features
		# Calculate optical flow
		# self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_thresh, self.thresh, tmp, None, **lk_params)
		if self.current_frame < 10:
			self.bgThreshold()
			print(self.features)
			self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_drawing, self.drawing, tmp, None, **lk_params)

		elif self.current_frame == 10:
			self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_drawing, self.gray, tmp, None, **lk_params)
			
		elif self.current_frame > 10:
			self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, tmp, None, **lk_params)
			# self.draw()

		# flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.gray,None,0.5,3,15,3,5,1.2,0)

		self.current_x = (self.features[0][0][0] - self.prev_features[0][0][0]) / (1/self.fps)
		self.current_y = -1*(self.features[0][0][1] - self.prev_features[0][0][1]) / (1/self.fps)
		self.draw()
		self.prev_thresh = self.thresh
		self.prev_gray   = self.gray
		self.prev_drawing = self.drawing
		self.current_frame += 1
		

	def draw(self):
		"""Draw the current image with points using OpenCV's
		own drawing functions. Press any key to close window"""

		# Draw points as green circles
		# for point in self.features:
		# 	self.out = cv2.circle(self.frame, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)

		for i,(new,old) in enumerate(zip(self.features,self.prev_features)):
			a,b = new.ravel()
			c,d = old.ravel()
			self.mask  = cv2.line(self.mask, (a,b),(c,d), color[i].tolist(), 2)
			self.frame = cv2.circle(self.frame,(a,b),5,color[i].tolist(),-1)

		self.out = cv2.add(self.frame,self.mask)
		cv2.putText(self.out, 'x = '+ str(self.current_x), (int(self.features[0][0][0]+20), int(self.features[0][0][1]-15)),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 'y = ' + str(self.current_y), (int(self.features[0][0][0]+20), int(self.features[0][0][1])),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 'z = ?', (int(self.features[0][0][0]+20), int(self.features[0][0][1]+15)),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 't = ' + str(self.current_frame/self.fps), (int(self.features[0][0][0]+20), int(self.features[0][0][1]+30)),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		# self.writer.write(self.out)
		# cv2.imshow('LKtrack', self.out)
		print(self.fps)

		
		

	def track(self):
		"""Generator for stepping through a sequence"""

		while(1):
			if self.features == []:
				self.detectPoints()
			else:
				self.trackPoints()
			
			self.current_frame += 1
			# cv2.imshow('LKtrack', self.out)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				print(self.tracks)
				break

# Unfinished code for background velocity calculation
# flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.gray,None,0.5,3,15,3,5,1.2,0)
# length =  sqrt(dx**2 + dy**2)
# totalLength = length(prev1[y,x]) + length(prev2[y+prev1[y,x][1], prev2[x+prev1[y,x]][0]]) ....
# disp = (x,y) + prev1[y,x] + prev2[y,x] ...
# speed = disp / t