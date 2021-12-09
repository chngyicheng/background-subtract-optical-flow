import cv2
import numpy as np
import copy


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
		self.detect_interval = 3
		self.current_x       = 0
		self.current_y       = 0

		# Some constraints and default parameters
		## Create some random colors
		self.color = np.random.randint(0,255,(100,3))

		## Lucas-Kanade optical flow params
		self.lk_params = dict(winSize=(15,15), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
		self.subpix_params = dict(zeroZone=(-1,-1), winSize=(10,10), criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))
		
		## Shi-Tomasi corner detection params
		self.feature_params = dict(maxCorners=50, qualityLevel=0.001, minDistance=50, blockSize=20)


		# For thresholding
		self.threshold      = 60  # BINARY threshold
		self.blurValue      = 5   # GaussianBlur parameter
		self.bgSubThreshold = 50
		self.learningRate   = 0.003
		self.isBgCaptured   = False   # whether the background captured
		self.bgModel        = None
		self.objectDetected = False
		
	def detectPoints(self):
		"""Detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""

		# Load image and threshold image
		self.capture   = cv2.VideoCapture(self.imnames)
		width          = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height         = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.fps       = int(self.capture.get(cv2.CAP_PROP_FPS))
		self.writer    = cv2.VideoWriter('drone3_update_track.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (width, height))
		__, self.frame = self.capture.read()

		for i in range(5):
			__, self.frame = self.capture.read()
			self.bgThreshold()

		# Search for good points
		self.prev_features = cv2.goodFeaturesToTrack(self.drawing, **self.feature_params)

		# Refine the corner locations
		cv2.cornerSubPix(self.drawing, self.prev_features, **self.subpix_params)

		self.features = self.prev_features
		print(self.features)
		self.objectDetected = True
		self.tracks   = [[p] for p in self.prev_features.reshape((-1,2))]
		
		self.prev_drawing = self.drawing
		# cv2.imshow('test', self.prev_drawing)
		
		self.mask = np.zeros_like(self.frame)
		
		
		self.draw()

	def bgThreshold(self):
		"""Main operation that runs thresholding
		using HSV values and background subtraction"""

		#  Apply smoothening filter over the frame
		self.img = cv2.bilateralFilter(self.frame, 5, 50, 100)

		# If background not captured, run background subtractor MOG2
		if not self.isBgCaptured:
			self.bgModel = cv2.createBackgroundSubtractorMOG2(600, self.bgSubThreshold, False)
			self.isBgCaptured = True    # Updates that background has been captured
			print( 'Background Captured')

		# Set the HSV values
		h_low  = 0
		s_low  = 0
		v_low  = 103
		h_high = 61
		s_high = 65
		v_high = 255

		# If object (drone) has been detected, create a mask to focus
		# on object with square of side (85 x 85) pixels, then creates
		# new image of frame in HSV with clearer image (parameters of 
		# thresholding and removes BG
		if self.objectDetected:
			self.object_mask = np.zeros_like(self.frame)
			x1 = self.x1 - 85
			y1 = self.y1 - 85
			x2 = self.x2 + 85
			y2 = self.y2 + 85
			cv2.rectangle(self.object_mask, (x1, y1), (x2, y2), (255,0,0), -1)
			
			# Update HSV value if object found
			h_low  = 0
			s_low  = 0
			v_low  = 0
			h_high = 180
			s_high = 76
			v_high = 255
		
		# If object not detected, 
		img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		img_blur = cv2.GaussianBlur(img_hsv, (self.blurValue, self.blurValue), 0)
		
		frame_threshold = cv2.inRange(img_blur, (h_low, s_low, v_low), (h_high, s_high, v_high))
		frame_threshold = self.removeBG(frame_threshold)

		# cv2.imshow('th', frame_threshold)

		drawing = self.img.copy()
		drawing[frame_threshold == 0] = 0
		# cv2.drawContours(drawing, [res], 0, (255, 255, 255), thickness=-1)
		self.drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
			

	def removeBG(self, frame): #Subtracting the background
		fgmask = self.bgModel.apply(frame,learningRate=self.learningRate)

		kernel = np.ones((3, 3), np.uint8)
		fgmask = cv2.erode(fgmask, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=fgmask)
		return res
	
	def checkPoints(self):
		"""Re-detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""
		self.color = np.random.randint(0,255,(100,3))

		# # Search for good points
		if not self.objectDetected:
			self.features = cv2.goodFeaturesToTrack(self.drawing, **self.feature_params, mask=self.mask)

			# Refine the corner locations
			cv2.cornerSubPix(self.drawing, self.features, **self.subpix_params)

		# self.tracks = [[p] for p in self.features.reshape((-1,2))]
		

	def trackPoints(self):
		"""Track the detected features"""

		# Load the image and create grayscale
		__, self.frame = self.capture.read()
		self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		# self.bgThreshold()

		"""To be used later, re-checks detection every few frames"""
		if self.current_frame % self.detect_interval == 0:
			self.checkPoints()

		

		# Reshape to fit input format
		tmp = np.float32(self.features).reshape(-1, 1, 2)

		self.prev_features = self.features
		# Calculate optical flow
		self.bgThreshold()
		self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_drawing, self.drawing, tmp, None, **self.lk_params)

		# flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.gray,None,0.5,3,15,3,5,1.2,0)

		self.current_x = (self.features[0][0][0] - self.prev_features[0][0][0]) / (1/self.fps)
		self.current_y = -1*(self.features[0][0][1] - self.prev_features[0][0][1]) / (1/self.fps)
		self.draw()
		self.prev_drawing = self.drawing
		self.current_frame += 1
		cv2.imshow("input", self.drawing)
		# cv2.imshow("mask", self.object_mask)
		

	def draw(self):
		"""Draw the current image with points using OpenCV's
		own drawing functions. Press any key to close window"""

		# Draw points as green circles
		# for point in self.features:
		self.x1 = 0
		self.y1 = 0
		self.x2 = 0
		self.y2 = 0
		# 	self.out = cv2.circle(self.frame, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
		for i,(new,old) in enumerate(zip(self.features,self.prev_features)):
			a,b = new.ravel()
			c,d = old.ravel()
			self.x1 += a
			self.y1 += b
			self.x2 += c
			self.y2 += d
		if i > 0:
			self.x1 = int(self.x1 / (i+1))
			self.y1 = int(self.y1 / (i+1))
			self.x2 = int(self.x2 / (i+1))
			self.y2 = int(self.y2 / (i+1))
		else:
			self.x1 = int(self.x1)
			self.y1 = int(self.y1)
			self.x2 = int(self.x2)
			self.y2 = int(self.y2)

		self.mask  = cv2.line(self.mask, (self.x1,self.y1),(self.x2,self.y2), self.color[i].tolist(), 2)
		self.frame = cv2.circle(self.frame,(self.x1,self.y1),5,self.color[i].tolist(),-1)

		self.out = cv2.add(self.frame,self.mask)
		cv2.putText(self.out, 'x = '+ str(self.current_x), (self.x1+20, self.y1-15),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 'y = ' + str(self.current_y), (self.x1+20, self.y1),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 'z = ?', (self.x1+20, self.y1+15),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 't = ' + str(self.current_frame/self.fps), (self.x1+20, self.y1+30),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		self.writer.write(self.out)
		cv2.imshow('LKtrack', self.out)
		# print(self.fps)

		
		

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