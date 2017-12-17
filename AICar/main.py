import cv2
import numpy as np
from sklearn.externals import joblib
import bluetooth

def pre_treatment(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	img = cv2.Canny(img, 20, 50)
	return img
	
def extract_feature(img):
	keypoints = cv2.xfeatures2d.StarDetector_create().detect(img)
	keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img, keypoints)
	return descriptors

def normalize(input_data):
	sum_input = np.sum(input_data)
	if sum_input > 0:
		return input_data / sum_input
	else:
		return input_data
		
def construct_feature(img, kmeans, centroids):
	img = pre_treatment(img)
	feature_vectors = extract_feature(img)
	if feature_vectors is not None:
		labels = kmeans.predict(feature_vectors)
		feature_vector = np.zeros(15)
		for i, item in enumerate(feature_vectors):
			feature_vector[labels[i]] += 1
		feature_vector = np.reshape(feature_vector, (1, feature_vector.shape[0]))
		return normalize(feature_vector)
	else:
		return None


addr = "AB:A4:95:56:34:02"	# need to change
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((addr,1))
while True:
	ans = 0
	vid = cv2.VideoCapture(0)
	time.sleep(2)
	img = vid.read()[1]
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv = np.array(hsv[:, :, 0])
	for i in range(hsv.shape[0]):
		for j in range(hsv.shape[1]):
			if hsv[i][j] > 0 and hsv[i][j] < 10:
				ans = 5
			if hsv[i][j] > 156 and hsv[i][j] < 180:
				ans = 5
			if hsv[i][j] > 35 and hsv[i][j] < 77:
				ans = 1
	if ans != 0:
		# send ans to car
		sock.send(ans)
	else:
		# send 8 to car to get distance s
		s = sock.recv(8)
		if s < 50:
			kmeans = joblib.load("kmeans.model")
			centroids = joblib.load("centroids.model")
			classifier = joblib.load("svm.model")
			img = construct_feature(img, kmeans, centroids)
			if img is not None:
				y = classifier.predict(np.reshape(img, (1, -1)))
				ans = y[0]
			if ans == 1:
				# send 3 to car
				sock.send(3)
			else:
				# send 4 to car
				sock.send(4)
sock.close()
