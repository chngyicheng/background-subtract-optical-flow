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


	def detectPoints(self):
		"""Detect 'Good features to track' (corners)
		in current frame using sub-pixel accuracy"""

		# Load image and threshold image
		self.capture   = cv2.VideoCapture(self.imnames)
		# self.capture.set(3, 960)
		# self.capture.set(4, 540)
		width          = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height         = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		# print('width')
		# print(width)
		# print(height)
		# print('height')
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
		"""
		Draw the current image with points using OpenCV's
		own drawing functions. Press any key to close window
		"""

		# Draw points as green circles
		# for point in self.features:
		self.x1 = 0
		self.y1 = 0
		self.x2 = 0
		self.y2 = 0

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

	"""
	Calculate 3D position of object.
	Code from Seah Shao Xuan (@seahhorse on Github)
	"""
	# def calculate_3D(self):

	# 	fx = 1454.6
	# 	cx = 960.9
	# 	fy = 1450.3
	# 	cy = 543.7
	# 	B = 1.5
	# 	epsilon = 7

	# 	for (auto & matched_track : matched_tracks_) {

	# 		if (matched_track.second[NUM_OF_CAMERAS_] < 2) continue;

	# 		int matched_id = matched_track.first;
	# 		int first_cam = -1;
	# 		int second_cam = -1;

	# 		for (int cam_idx = 0; cam_idx < NUM_OF_CAMERAS_; cam_idx++) {
	# 			if (first_cam == -1) {
	# 				if (matched_track.second[cam_idx]) first_cam = cam_idx;
	# 			} else if (second_cam == -1) {
	# 				if (matched_track.second[cam_idx]) second_cam = cam_idx;
	# 			} else {
	# 				break;
	# 			}
	# 		}

	# 		auto track_plot_a = cumulative_tracks_[first_cam]->track_plots_[matched_id];
	# 		auto track_plot_b = cumulative_tracks_[second_cam]->track_plots_[matched_id];

	# 		if (track_plot_a->lastSeen_ != frame_count_ || track_plot_b->lastSeen_ != frame_count_) continue;

	# 		int x_L = track_plot_a->xs_.back();
	# 		int y_L = track_plot_a->ys_.back();
	# 		int x_R = track_plot_b->xs_.back();
	# 		int y_R = track_plot_b->ys_.back();

	# 		double alpha_L = atan2(x_L - cx, fx) / M_PI * 180;
	# 		double alpha_R = atan2(x_R - cx, fx) / M_PI * 180;

	# 		double Z = B / (tan((alpha_L + epsilon / 2) * (M_PI / 180)) - tan((alpha_L - epsilon / 2) * (M_PI / 180)));
	# 		double X = (Z * tan((alpha_L + epsilon / 2) * (M_PI / 180)) - B / 2
	# 					+ Z * tan((alpha_R - epsilon / 2) * (M_PI / 180)) + B / 2) / 2;
	# 		double Y = (Z * - (y_L - cy) / fy + Z * - (y_R - cy) / fy) / 2;

	# 		double tilt = 10 * M_PI / 180;
	# 		Eigen::Matrix3d R;
	# 		R << 1, 0, 0,
	# 			0, cos(tilt), sin(tilt),
	# 			0, -sin(tilt), cos(tilt);
	# 		Eigen::Vector3d XYZ_original;
	# 		XYZ_original << X, Y, Z;
	# 		auto XYZ = R * XYZ_original;
	# 		X = XYZ(0);
	# 		Y = XYZ(1);
	# 		Z = XYZ(2);

	# 		Y += 1;

	# 		X = (std::round(X*100))/100;
	# 		Y = (std::round(Y*100))/100;
	# 		Z = (std::round(Z*100))/100;

	# 		track_plot_a->xyz_ = {X, Y, Z};
	# 		track_plot_b->xyz_ = {X, Y, Z};

	# 		log_3D(track_plot_a, track_plot_b);


# Unfinished code for background velocity calculation
# flow = cv2.calcOpticalFlowFarneback(self.prev_gray,self.gray,None,0.5,3,15,3,5,1.2,0)
# length =  sqrt(dx**2 + dy**2)
# totalLength = length(prev1[y,x]) + length(prev2[y+prev1[y,x][1], prev2[x+prev1[y,x]][0]]) ....
# disp = (x,y) + prev1[y,x] + prev2[y,x] ...
# speed = disp / t