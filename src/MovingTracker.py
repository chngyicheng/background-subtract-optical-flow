import cv2
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class LKDenseTracker(object):
	""" Class for Lucas-Kanade tracking with
	dense optical flow"""

	def __init__(self, imnames):
		# Initialise with a list of image names
		self.imnames         = imnames
		self.features        = []
		self.prev_features   = []
		self.tracks          = []
		self.vel_orient      = []
		self.current_frame   = 0
		self.detect_interval = 3
		self.dx              = 0
		self.dy              = 0
		self.velocity        = [0, 0, 0]
		self.area            = 0
		self.prev_area       = 0
		self.x1 = 0
		self.y1 = 0
		self.x2 = 0
		self.y2 = 0
		self.mask_x1 = 0
		self.mask_y1 = 0

		# Some constraints and default parameters
		## Create some random colors
		self.color = np.random.randint(0,255,(100,3))

		## Lucas-Kanade optical flow params
		self.lk_params     = dict(winSize=(15,15), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
		self.subpix_params = dict(zeroZone=(-1,-1), winSize=(10,10), criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,20,0.03))
		
		## Shi-Tomasi corner detection params
		self.feature_params = dict(maxCorners=5, qualityLevel=0.001, minDistance=50, blockSize=20)

		# For thresholding
		self.threshold         = 60  # BINARY threshold
		self.blurValue         = 7   # GaussianBlur parameter
		self.bgSubThreshold    = 50
		self.learningRate      = 0.003
		self.bgModel           = None
		self.min_detect_frames = 10

		# Initialise flags
		self.isBgCaptured   = False   # whether the background captured
		self.objectDetected = False
		self.bgFlowInit     = False
		

	def detectPoints(self):
		"""Detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""

		# Load image and threshold image
		self.capture    = cv2.VideoCapture(self.imnames)
		self.width      = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height     = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

		self.fps        = int(self.capture.get(cv2.CAP_PROP_FPS))
		self.writer1    = cv2.VideoWriter('track_w_vanish_point.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (self.width, self.height))
		self.writer2    = cv2.VideoWriter('bg_flow.mp4', cv2.VideoWriter_fourcc(*'XVID'),25, (self.width, int(self.height)))#/4*3)))
		__, self.frame  = self.capture.read()

		# for i in range(self.min_detect_frames):
		# 	__, self.frame = self.capture.read()
		# 	self.bgThreshold()

		# # Search for good points
		# self.prev_features = cv2.goodFeaturesToTrack(self.drawing, **self.feature_params)

		# # Refine the corner locations
		# cv2.cornerSubPix(self.drawing, self.prev_features, **self.subpix_params)

		# self.features = self.prev_features
		# # print(self.features)
		# self.frame_two = self.frame
		# self.objectDetected = True
		# self.tracks   = [[p] for p in self.prev_features.reshape((-1,2))]
		
		# self.prev_drawing = self.drawing
		# self.frame_one    = self.frame_two
		# cv2.imshow('test', self.prev_drawing)
		
		self.mask  = np.zeros_like(self.frame)
		self.arrow = np.zeros_like(self.frame)

		self.hsv_mask = np.zeros_like(self.frame)
		self.hsv_mask[..., 1] = 255

		self.prev_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

		# self.draw()


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

		# Set the HSV values (DJI Video 900+)
		# h_low  = 0
		# s_low  = 0
		# v_low  = 103
		# h_high = 61
		# s_high = 65
		# v_high = 255

		# Set the HSV values (DJI Video 001)
		h_low  = 0
		s_low  = 18
		v_low  = 103
		h_high = 53
		s_high = 60
		v_high = 255

		# If object (drone) has been detected, create a rectangle mask
		# to focus on object, then creates new image of frame in HSV 
		# with clearer image (parameters of thresholding) and removes BG
		if self.objectDetected:
			# Create rectangular mask to track object
			self.FGMaskUpdate()
			
			# Update HSV value if object found
			h_low  = 0
			s_low  = 0
			v_low  = 136
			h_high = 180
			s_high = 76
			v_high = 255
		
		img_hsv  = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
		img_blur = cv2.GaussianBlur(img_hsv, (self.blurValue, self.blurValue), 0)
		# h, s, v  = cv2.split(img_blur)
		
		frame_threshold = cv2.inRange(img_blur, (h_low, s_low, v_low), (h_high, s_high, v_high))
		frame_threshold = self.removeBG(frame_threshold)
		# cv2.imshow('thresh',frame_threshold)

		
		if self.objectDetected:
			# con_img = frame_threshold[x1:x2,y1:y2]
			con_img = img_blur.copy()
			con_img[self.object_mask == 0] = 0

			# self.detect_img = self.img.copy()
			# self.detect_img[self.object_mask == 0] = 0

			con_img = cv2.inRange(img_blur, (h_low, s_low, v_low), (h_high, s_high, v_high))
			# con_img = self.removeBG(con_img)
			
			# get contours
			contours, hierarchy = cv2.findContours(con_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detecting contours

			self.area = cv2.contourArea(contours[0])

		# cv2.imshow('th', frame_threshold)

		drawing = self.img.copy()
		drawing[frame_threshold == 0] = 0
		self.drawing = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)


	def FGMaskUpdate(self):
		"""Update foreground mask to track object"""

		# Create empty mask to track object
		self.object_mask = [[0]*self.width]*self.height
		self.object_mask = np.asarray(self.object_mask)
		self.object_mask = self.object_mask.astype(np.uint8)

		# Draw rectangle of opposing corners of (x1, y1), (x2, y2)
		# if self.x1 > self.oldx1:
		# 	x_diff = self
		x1 = self.mask_x1 - int(abs(self.mask_x1-self.old_mask_x1)*2) - 25
		y1 = self.mask_y1 - int(abs(self.mask_y1-self.old_mask_y1)*2) - 15
		x2 = self.old_mask_x1 + int(abs(self.mask_x1-self.old_mask_x1)*2) + 25
		y2 = self.old_mask_y1 + int(abs(self.mask_y1-self.old_mask_y1)*2) + 15
		self.object_mask[y1:y2, x1:x2]  = 255

	
	def BGMaskUpdate(self):
		"""Update background mask to track object"""

		# Create mask of ones ignoring drone for background flow calculation
		self.bg_flow_mask = [[1]*self.width]*self.height
		self.bg_flow_mask = np.asarray(self.bg_flow_mask)
		self.bg_flow_mask = self.bg_flow_mask.astype(np.uint8)

		# Draw rectangle of opposing corners of (x1, y1), (x2, y2) and equate it
		# to 0
		# x1 = self.x1 - 55
		# y1 = self.y1 - 35
		# x2 = self.oldx1 + 55
		# y2 = self.oldy1 + 35

		x1 = 0
		y1 = int(self.height)
		x2 = int(self.width)
		y2 = int(self.height/2)
		self.bg_flow_mask[y1:y2, x1:x2] = 0

		# gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

		# self.bg_flow_masked = gray_frame.copy()
		# # self.bg_flow_mask[self.object_mask == 1] = 0
		# self.bg_flow_masked[bg_flow_mask == 0] = 0

		# cv2.imshow("bg flow", self.bg_flow_masked)


	def removeBG(self, frame):
		"""Subtracts the background"""

		fgmask = self.bgModel.apply(frame,learningRate=self.learningRate)

		kernel = np.ones((3, 3), np.uint8)
		fgmask = cv2.erode(fgmask, kernel, iterations=1)
		res = cv2.bitwise_and(frame, frame, mask=fgmask)
		return res

	
	def checkPoints(self):
		"""Re-detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""
		
		

		# # Search for good points
		if self.current_frame > 5 and self.objectDetected:
			self.color = np.random.randint(0,255,(100,3))
			self.features = cv2.goodFeaturesToTrack(self.drawing, **self.feature_params, mask=self.object_mask)

			# Refine the corner locations
			# if np.all(cv2.cornerSubPix(self.drawing, self.features, **self.subpix_params) > 0):
			# 	cv2.cornerSubPix(self.drawing, self.features, **self.subpix_params)
			# else:
			# 	self.features = self.prev_features
			try:
				cv2.cornerSubPix(self.drawing, self.features, **self.subpix_params)
			except:
				self.features = self.prev_features

			cv2.imshow("drawing", self.drawing)

		
	def trackPointsLK(self):
		"""Track the detected features"""

		# Load the image and create grayscale
		__, self.frame = self.capture.read()
		self.frame_two = self.frame
		# self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		self.bgThreshold()

		# Re-checks detection every few frames
		if self.current_frame % self.detect_interval == 0:
			self.checkPoints()

		# Reshape to fit input format
		tmp = np.float32(self.features).reshape(-1, 1, 2)

		self.prev_features = self.features
		# Calculate optical flow
		# self.bgThreshold()
		self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_drawing, self.drawing, tmp, None, **self.lk_params)

		self.FlowTrackFB()

		self.dx += (self.features[0][0][0] - self.prev_features[0][0][0]) #/ (1/self.fps)
		self.dy -= (self.features[0][0][1] - self.prev_features[0][0][1])# / (1/self.fps)
		self.draw()

		# Update all prev to current
		self.prev_area    = self.area
		self.prev_drawing = self.drawing
		self.frame_one    = self.frame_two

		self.current_frame += 1
		# cv2.imshow("input", self.drawing)
		cv2.imshow("mask", self.object_mask)
	

	"""New code begins here"""

	def FlowTrackFB(self):
		"""Track the scene flow with Farneback Optical Flow"""

		# Update mask used for background
		# self.BGMaskUpdate()

		__, self.frame  = self.capture.read()
		self.line_list = []
		# image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		# pixel_values = image.reshape((-1, 3))
		# print(pixel_values)
		# return

		self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

		# Set the ROI for bg flow
		# x1 = 0
		# y1 = 0 #int(self.height/4)
		# x2 = int(self.width)
		# y2 = int(self.height)

		# Farneback
		# current_frame = cv2.cvtColor(self.frame_two, cv2.COLOR_BGR2GRAY)
		# prev_frame    = cv2.cvtColor(self.frame_one, cv2.COLOR_BGR2GRAY)

		# current_frame = current_frame[y1:y2, x1:x2]
		# prev_frame    = prev_frame[y1:y2, x1:x2]

		# LK Tracker
		# current_frame = self.frame_two[y1:y2, x1:x2]
		# prev_frame    = self.frame_one[y1:y2, x1:x2]

		## Test code
		# self.flow = cv2.optflow.calcOpticalFlowSparseToDense(prev_frame, current_frame, None)
		self.flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.gray,None,0.5,3,15,3,5,1.2,0)
		# self.flow = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, None, None, **self.lk_params)
		focal_length = 0.00449
		height = 20
		ang_vel = 0

		self.frame_out = self.frame
		self.foregroundDetect()
		self.getFlowLines()

		if self.line_list:
			# If lines exist, get vanishing point
			self.min_x, self.min_y = self.getVanishingPoint(self.line_list)
			self.frame_out = cv2.circle(self.frame_out, (int(self.min_x), int(self.min_y)),5,(255,255,0),-1)

		# self.detectKCluster()


		
		## Old
		# self.flow = cv2.calcOpticalFlowFarneback(self.prev_drawing,self.drawing,None,0.5,3,15,3,5,1.2,0)
		# Compute magnitude and angle of 2D vector
		self.mag, ang = cv2.cartToPolar(self.foreground_points[..., 0], self.foreground_points[..., 1])
		# Set image hue value according to the angle of optical flow
		self.hsv_mask[..., 0] = ang * 180 / math.pi / 2
		# Set value as per the normalized magnitude of optical flow
		self.hsv_mask[..., 2] = cv2.normalize(self.mag, None, 0, 255, cv2.NORM_MINMAX)
		# Convert to rgb
		self.rgb_representation = cv2.cvtColor(self.hsv_mask, cv2.COLOR_HSV2BGR)
		
		self.out = cv2.add(self.frame_out,self.flow_line_mask)
		self.blobDetector()
		cv2.imshow('flow', self.rgb_representation)
		cv2.imshow("frame", self.out)

		# # Get start and end coordinates of the optical flow
		# self.flow_start = np.stack(np.meshgrid(range(self.flow.shape[1]), range(self.flow.shape[0])), 2)
		# self.flow_end = (self.flow[self.flow_start[:,:,1],self.flow_start[:,:,0],:1]*3 + self.flow_start).astype(np.int32)

		# Calculate optical flow



		# Threshold values
		# norm = np.linalg.norm(self.flow_end - self.flow_start, axis=2)
		# norm[norm < 2] = 0

		# self.nz = np.nonzero(norm)
		
		# Update all prev to current
		self.prev_flow = self.flow
		self.prev_gray = self.gray
		# self.frame_one = self.frame_two
	
		# cv2.imshow('flow', rgb_representation)
		# cv2.imshow('frame', self.frame)

		self.writer1.write(self.out)
		self.writer2.write(self.rgb_representation)

	
	def foregroundDetect(self):
		print("Current Frame: ", self.current_frame)

		self.foreground_points = np.zeros_like(self.flow)
		self.background_points = self.flow

		# Get average flow of each box
		self.getAvgFlow()

		# Find foreground from each box
		self.splitFGAndBG()


	def getAvgFlow(self):
		
		# Average for bottom third fifth, left first fifth
		self.avg_x_third_quart_l_corner, self.avg_y_third_quart_l_corner = self.calculateFlowAtPoint(self.height/2-10, self.height/2+10, self.width/10-5, self.width/10+5)
		self.avg_x_third_quart_l_quart, self.avg_y_third_quart_l_quart   = self.calculateFlowAtPoint(self.height/2-10, self.height/2+10+10, self.width*3/10-5, self.width*3/10+5)
		self.avg_x_third_quart_centre, self.avg_y_third_quart_centre     = self.calculateFlowAtPoint(self.height/2-10, self.height/2+10+10, self.width/2-5, self.width/2+5)
		self.avg_x_third_quart_r_quart, self.avg_y_third_quart_r_quart   = self.calculateFlowAtPoint(self.height/2-10, self.height/2+10+10, self.width*7/10-5, self.width*7/10+5)
		self.avg_x_third_quart_r_corner, self.avg_y_third_quart_r_corner = self.calculateFlowAtPoint(self.height/2-10, self.height/2+10+10, self.width*9/10-5, self.width*9/10+5)

		# # Average for bottom fourth fifth, left first fifth
		self.avg_x_four_quart_l_corner, self.avg_y_four_quart_l_corner = self.calculateFlowAtPoint(self.height*7/10-10, self.height*7/10+10, self.width/10-5, self.width/10+5)
		self.avg_x_four_quart_l_quart, self.avg_y_four_quart_l_quart   = self.calculateFlowAtPoint(self.height*7/10-10, self.height*7/10+10, self.width*3/10-5, self.width*3/10+5)
		self.avg_x_four_quart_centre, self.avg_y_four_quart_centre     = self.calculateFlowAtPoint(self.height*7/10-10, self.height*7/10+10, self.width/2-5, self.width/2+5)
		self.avg_x_four_quart_r_quart, self.avg_y_four_quart_r_quart   = self.calculateFlowAtPoint(self.height*7/10-10, self.height*7/10+10, self.width*7/10-5, self.width*7/10+5)
		self.avg_x_four_quart_r_corner, self.avg_y_four_quart_r_corner = self.calculateFlowAtPoint(self.height*7/10-10, self.height*7/10+10, self.width*9/10-5, self.width*9/10+5)

		# Average for bottom last fifth
		self.avg_x_bot_l_corner, self.avg_y_bot_l_corner = self.calculateFlowAtPoint(self.height*9/10-10, self.height*9/10+10, self.width/10-5, self.width/10+5)
		self.avg_x_bot_l_quart, self.avg_y_bot_l_quart   = self.calculateFlowAtPoint(self.height*9/10-10, self.height*9/10+10, self.width*3/10-5, self.width*3/10+5)
		self.avg_x_bot_centre, self.avg_y_bot_centre     = self.calculateFlowAtPoint(self.height*9/10-10, self.height*9/10+10, self.width/2-5, self.width/2+5)
		self.avg_x_bot_r_quart, self.avg_y_bot_r_quart   = self.calculateFlowAtPoint(self.height*9/10-10, self.height*9/10+10, self.width*7/10-5, self.width*7/10+5)
		self.avg_x_bot_r_corner, self.avg_y_bot_r_corner = self.calculateFlowAtPoint(self.height*9/10-10, self.height*9/10+10, self.width*9/10-5, self.width*9/10+5)


	def calculateFlowAtPoint(self, y1, y2, x1, x2):
		avg_x = 0
		avg_y = 0
		count = 0
		for y in range(int(y1), int(y2)):
			for x in range(int(x1),int(x2)):
				avg_x += self.flow[y,x,0]
				avg_y += self.flow[y,x,1]
				count += 1
		avg_x /= count
		avg_y /= count
		return avg_x, avg_y


	def splitFGAndBG(self):

		# Criteria numbers
		third  = 1.5
		fourth = 2.5
		last   = 2.75

		# Bottom third row
		criteria = third
		# Bottom third fifth, left first fifth
		self.getForegroundObjectsAtBox(self.height*2/5, self.height*3/5, 1, self.width/5, self.avg_x_third_quart_l_corner, self.avg_y_third_quart_l_corner, criteria+0.5, criteria+0.5)
		# Bottom third fifth, left second fifth
		self.getForegroundObjectsAtBox(self.height*2/5, self.height*3/5, self.width/5, self.width*2/5, self.avg_x_third_quart_l_quart, self.avg_y_third_quart_l_quart, criteria+0.25, criteria+0.2)
		# Bottom third fifth, middle
		self.getForegroundObjectsAtBox(self.height*2/5, self.height*3/5, self.width*2/5, self.width*3/5, self.avg_x_third_quart_centre, self.avg_y_third_quart_centre, criteria, criteria)
		# Bottom third fifth, right third fifth
		self.getForegroundObjectsAtBox(self.height*2/5, self.height*3/5, self.width*3/5, self.width*4/5, self.avg_x_third_quart_r_quart, self.avg_y_third_quart_r_quart, criteria+0.25, criteria+0.2)
		# Bottom third fifth, right last fifth
		self.getForegroundObjectsAtBox(self.height*2/5, self.height*3/5, self.width*4/5, self.width, self.avg_x_third_quart_r_corner, self.avg_y_third_quart_r_corner, criteria+0.5, criteria+0.5)

		# Bottom fourth row
		criteria = fourth
		# # Bottom fourth fifth, left first fifth
		self.getForegroundObjectsAtBox(self.height*3/5, self.height*4/5, 1, self.width/5, self.avg_x_four_quart_l_corner, self.avg_y_four_quart_l_corner, criteria+0.5, criteria+0.5)
		# Bottom fourth fifth, left second fifth
		self.getForegroundObjectsAtBox(self.height*3/5, self.height*4/5, self.width/5, self.width*2/5, self.avg_x_four_quart_l_quart, self.avg_y_four_quart_l_quart, criteria+0.25, criteria+0.2)
		# Bottom fourth fifth, middle
		self.getForegroundObjectsAtBox(self.height*3/5, self.height*4/5, self.width*2/5, self.width*3/5, self.avg_x_four_quart_centre, self.avg_y_four_quart_centre, criteria, criteria)
		# Bottom fourth fifth, right third fifth
		self.getForegroundObjectsAtBox(self.height*3/5, self.height*4/5, self.width*3/5, self.width*4/5, self.avg_x_four_quart_r_quart, self.avg_y_four_quart_r_quart, criteria+0.25, criteria+0.2)
		# Bottom fourth fifth, right last fifth
		self.getForegroundObjectsAtBox(self.height*3/5, self.height*4/5, self.width*4/5, self.width, self.avg_x_four_quart_r_corner, self.avg_y_four_quart_r_corner, criteria+0.5, criteria+0.5)

		# Bottom fourth row
		criteria = last
		# Bottom last fifth, left first fifth
		self.getForegroundObjectsAtBox(self.height*4/5, self.height, 1, self.width/5, self.avg_x_bot_l_corner, self.avg_y_bot_l_corner, criteria+0.75, criteria+0.75)
		# Bottom last fifth, left second fifth
		self.getForegroundObjectsAtBox(self.height*4/5, self.height, self.width/5, self.width*2/5, self.avg_x_bot_l_quart, self.avg_y_bot_l_quart, criteria+0.25, criteria+0.25)
		# Bottom last fifth, middle
		self.getForegroundObjectsAtBox(self.height*4/5, self.height, self.width*2/5, self.width*3/5, self.avg_x_bot_centre, self.avg_y_bot_centre, criteria, criteria)
		# Bottom last fifth, right third fifth
		self.getForegroundObjectsAtBox(self.height*4/5, self.height, self.width*3/5, self.width*4/5, self.avg_x_bot_r_quart, self.avg_y_bot_r_quart, criteria+0.25, criteria+0.25)
		# Bottom last fifth, right last fifth
		self.getForegroundObjectsAtBox(self.height*4/5, self.height, self.width*4/5, self.width, self.avg_x_bot_r_corner, self.avg_y_bot_r_corner, criteria+0.75, criteria+0.75)


	def getForegroundObjectsAtBox(self, y1, y2, x1, x2, flow_x, flow_y, criteria_x, criteria_y):
		for y in range(int(y1), int(y2)):
			for x in range(int(x1), int(x2)):
				if abs(self.flow[y,x,0] - flow_x) < 0.5 and abs(self.flow[y,x,1] - flow_y) < 0.5:
					self.foreground_points[y,x,0] = -1
					self.foreground_points[y,x,1] = -1
				if abs(self.flow[y,x,0] - flow_x) > criteria_x or abs(self.flow[y,x,1] - flow_y) > criteria_y:
					if self.foreground_points[y,x,0] != -1 or self.foreground_points[y,x,1] != -1:
						# Get foreground points and add it to self.foreground_points
						self.foreground_points[y,x,0] = self.flow[y,x,0]
						self.foreground_points[y,x,1] = self.flow[y,x,1]
						# Remove foreground points from background points
						self.background_points[y,x,0] = 0
						self.background_points[y,x,1] = 0
				else:
					self.foreground_points[y,x,0] = 0
					self.foreground_points[y,x,1] = 0

	def getFlowLines(self):
		"""Plot the flow lines and output them onto frame if needed"""
		self.flow_line_mask  = np.zeros_like(self.frame)
		for x in range(100, self.width, 100):
			for y in range(100, self.height, 100):
				# print(y,x)
				new_x = x - self.background_points[y,x,0]
				new_y = y - self.background_points[y,x,1]
				
				if abs(self.background_points[y,x,0]) < 1 or abs(self.background_points[y,x,1]) < 1:
					new_x = 0
					new_y = 0
				
				if new_x != 0 or new_y != 0:
					#Comment out the next line if output of flow lines not needed
					self.flow_line_mask = cv2.line(self.flow_line_mask, (x,y),(int(new_x),int(new_y)), (255,255,0), 1)
					# Get the line equation and add to the list
					# y = mx + b
					m = self.background_points[y,x,1]/self.background_points[y,x,0]
					b = y - m*x
					self.line_list.append([m,b])


	def getVanishingPoint(self, line_equation):
		"""Obtained from KEDIARAHUL135 on GitHub (https://github.com/KEDIARAHUL135/VanishingPoint)"""
		# We will apply RANSAC inspired algorithm for this. We will take combination 
		# of 2 lines one by one, find their intersection point, and calculate the 
		# total error(loss) of that point. Error of the point means root of sum of 
		# squares of distance of that point from each line.
		vanishing_point = None
		min_error = 1000000000000

		for i in range(len(line_equation)):
			for j in range(i+1, len(line_equation)):
				m1, b1 = line_equation[i][0], line_equation[i][1]
				m2, b2 = line_equation[j][0], line_equation[j][1]

				if m1 != m2:
					x0 = (b1 - b2) / (m2 - m1)
					y0 = m1 * x0 + b1

					error = 0
					for k in range(len(line_equation)):
						m, b = line_equation[k][0], line_equation[k][1]
						m_ = (-1 / m)
						b_ = y0 - m_ * x0

						x_ = (b - b_) / (m_ - m)
						y_ = m_ * x_ + b_

						l = np.sqrt((y_ - y0)**2 + (x_ - x0)**2)

						error += l**2

					error = np.sqrt(error)

					if min_error > error:
						min_error = error
						vanishing_point = [x0, y0]			
		return vanishing_point


	def blobDetector(self):
		"""Draws box around detected target"""

		gray_image = self.gray.copy()
		gray_image[self.mag < 1] = 0

		# get contours
		# result = img.copy()
		(ret, thresh) = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
		contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0] if len(contours) == 2 else contours[1]
		num_target = 0
		for cntr in contours:
			rect = cv2.boundingRect(cntr)
			if rect[2] < 10 or rect[3] < 10: continue
			x,y,w,h = rect
			cv2.rectangle(self.out, (x, y), (x+w, y+h), (0, 0, 255), 2)
			num_target += 1

		text = "Number of targets: " + str(num_target)
		cv2.putText(self.out, text, (10, 50),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), 2)

		# show thresh and result    
		# cv2.imshow("bounding_box", self.frame_out)



		# detector = cv2.SimpleBlobDetector_create(params)
		# # gray_image = cv2.cvtColor(self.rgb_representation, cv2.COLOR_BGR2GRAY)
		# (ret, thresh) = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
		# cv2.imshow("thresh",thresh)

		# keypoints = detector.detect(thresh)

		# # blank = np.zeros((1, 1))
		# blank = np.array([])
		# self.blob = cv2.drawKeypoints(self.out, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# cv2.imshow("blob",self.blob)
		# number_of_blobs = len(keypoints)
		# text = "Number of targets: " + str(number_of_blobs)



	def detectKCluster(self):
		k = 3
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
		foreground = self.foreground_points.reshape((-1,3))
		_, labels, (centers) = cv2.kmeans(foreground, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

		# convert back to 8 bit values
		centers = np.uint8(centers)

		# flatten the labels array
		labels = labels.flatten()

		# convert all pixels to the color of the centroids
		segmented_image = centers[labels.flatten()]
		segmented_image = segmented_image.reshape(self.foreground_points.shape)


		""" PROGRESS TILL HERE"""
		# cv2.imshow("segment", segmented_image)
		# plt.imshow(segmented_image)
		# plt.show()

		# Compute magnitude and angle of 2D vector
		# mag, ang = cv2.cartToPolar(segmented_image[..., 0], segmented_image[..., 1])
		# # Set image hue value according to the angle of optical flow
		# self.hsv_mask[..., 0] = ang * 180 / math.pi / 2
		# # Set value as per the normalized magnitude of optical flow
		# self.hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		# # Convert to rgb
		# rgb_representation = cv2.cvtColor(self.hsv_mask, cv2.COLOR_HSV2BGR)
		# cv2.imshow('flow', rgb_representation)

		# disable only the cluster number 2 (turn the pixel into black)
		masked_image = np.copy(self.flow)
		# masked_image[..., 2] = self.current_frame/60

		# masked_image[..., 0] = segmented_image[..., 0]
		# masked_image[..., 1] = segmented_image[..., 1]
		# convert to the shape of a vector of pixel values
		masked_image = masked_image.reshape((-1, 3))
		# color (i.e cluster) to disable
		# cluster = 1
		# masked_image[labels != cluster] = 0
		cluster = 2
		masked_image[labels == cluster] = 255
		
		# masked_image[labels == cluster] = [0, 0, 0]
		# convert back to original shape
		masked_image = masked_image.reshape(self.flow.shape)

		# print(masked_image)

		# Compute magnitude and angle of 2D vector
		mag, ang = cv2.cartToPolar(self.foreground_points[..., 0], self.foreground_points[..., 1])
		# Set image hue value according to the angle of optical flow
		self.hsv_mask[..., 0] = ang * 180 / math.pi / 2
		# Set value as per the normalized magnitude of optical flow
		self.hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		# Convert to rgb
		rgb_representation = cv2.cvtColor(self.hsv_mask, cv2.COLOR_HSV2BGR)
		cv2.imshow('flow', rgb_representation)
		cv2.imshow("frame", self.frame)
		self.writer2.write(rgb_representation)


		print("it works", self.current_frame)
		# plt.imshow(masked_image)
		# ax = plt.axes(projection='3d')

		# Data for a three-dimensional line
		# zline = np.linspace(0, 15, 1000)
		# xline = np.sin(zline)
		# yline = np.cos(zline)
		# ax.plot3D(xline, yline, zline, 'gray')

		# Data for three-dimensional scattered points
		# zdata = 15 * np.random.random(100)
		# xdata = masked_image
		# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
		# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
		# plt.show()



	def draw(self):
		"""
		Draw the current image with points using OpenCV's
		own drawing functions. Press any key to close window
		"""

		# Draw points as green circles
		# for point in self.features:
		self.old_mask_x1 = self.x1
		self.old_mask_y1 = self.y1

		self.x1 = 0
		self.y1 = 0
		self.x2 = 0
		self.y2 = 0
		
		self.mask_x1 = 0
		self.mask_y1 = 0

		for i,(new,old) in enumerate(zip(self.features,self.prev_features)):
			a,b = new.ravel()
			c,d = old.ravel()
			self.mask  = cv2.line(self.mask, (int(a),int(b)),(int(c),int(d)), self.color[i].tolist(), 2)
			self.frame = cv2.circle(self.frame,(int(a),int(b)),5,self.color[i].tolist(),-1)
			self.x1 += a
			self.y1 += b
			self.x2 += c
			self.y2 += d

			if self.mask_x1 == 0 and self.mask_y1 == 0:
				self.mask_x1 = int(a)
				self.mask_y1 = int(b)

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


		self.out = cv2.add(self.frame,self.mask)
		
		self.update_3D_velocity_orientation()

		# float_dx = "{:.2f}".format(self.velocity[0])
		# float_dy = "{:.2f}".format(self.velocity[1])
		float_dx = "{:.2f}".format(self.dx)
		float_dy = "{:.2f}".format(self.dy)
		# float_dz = "{:.2f}".format(self.velocity[2])
		float_dz = 0
		float_time = "{:.2f}".format(self.current_frame/60)  ## or self.fps

		cv2.putText(self.out, 'dx = ' + str(float_dx), (self.x1+40, self.y1-15),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 'dy = ' + str(float_dy), (self.x1+40, self.y1),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 'Vz = ' + str(float_dz), (self.x1+40, self.y1+15),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		cv2.putText(self.out, 't  = ' + str(float_time), (self.x1+40, self.y1+30),
                cv2.FONT_HERSHEY_PLAIN, 1 , (0,0,0), thickness=2)
		
		self.writer1.write(self.out)
		cv2.imshow('LKtrack', self.out)
		# print(self.fps)

	# def calc3DPose(self):
	# 	self.dy = 0
		

	"""
	Calculate 3D position of object.
	Code from Seah Shao Xuan (@seahhorse on Github)
	"""
	def update_3D_velocity_orientation(self) :
		# checks if the data is from the current frame
		# if (frameNos_.size() > 15 && frameNos_.back() == frame_no):

		history = 15

		# declare camera lens parameters
		
		# Specs for the Creative Camera
		# FOV_X_ = 69.5
		# FOV_Y_ = 42.6

		# Specs for the GOPRO HERO 9 Camera
		# FOV_X_ = 87
		# FOV_Y_ = 56

		# Specs for Mavic Mini Camera
		FOV_X_ = 82
		FOV_Y_ = 39
		# self.features[0][0][0] - self.prev_features[0][0][0]
		
		# dx = xs_.end()[-1] - xs_.end()[-1-history];
		# double dy = ys_.end()[-1] - ys_.end()[-1-history];
		
		if (self.area == 0 or self.prev_area == 0):
			r = 1
		else:
			r = math.sqrt(self.area/self.prev_area)
		
		
		velocity_x = (self.dx / self.width) * 2 * math.tan(math.pi / 180.0 * FOV_X_ / 2.0)
		velocity_x += ((1.0 / r) - 1.0) / math.tan(((self.features[0][0][0] / self.width) - 0.5) * FOV_X_ * math.pi / 180.0)

		velocity_y = -1* (self.dy / self.height) * 2 * math.tan(math.pi / 180.0 * FOV_Y_ / 2.0)
		velocity_y -= ((1.0 / r) - 1.0) / math.tan(((self.features[0][0][1] / self.height) - 0.5) * FOV_Y_ * math.pi / 180.0)

		velocity_z = (1.0 / r) - 1.0

		self.velocity_length = math.sqrt(pow(velocity_x, 2) + pow(velocity_y, 2) + pow(velocity_z, 2))
		# print(self.velocity_length)

		if (self.velocity_length != 0 and self.velocity_length >= 0.05):
			self.velocity[0] = velocity_x / self.velocity_length; 
			self.velocity[1] = velocity_y / self.velocity_length; 
			self.velocity[2] = velocity_z / self.velocity_length; 
		else:
			self.velocity[0] = 0.0
			self.velocity[1] = 0.0
			self.velocity[2] = 0.0
		

		self.vel_orient.append(self.velocity)


	def track(self):
		"""Generator for stepping through a sequence"""

		while(1):
			if self.current_frame == 0:
				self.detectPoints()
			else:
				self.FlowTrackFB()
			
			self.current_frame += 1
			# cv2.imshow('LKtrack', self.out)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				print(self.tracks)
				break
