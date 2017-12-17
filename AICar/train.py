import cv2
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

def pre_treatment(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img, (3, 3), 0)
	img = cv2.Canny(img, 20, 50)
	return img

def extract_feature(img):
	keypoints = cv2.xfeatures2d.StarDetector_create().detect(img)
	keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img, keypoints)
	return descriptors
	
def bow(input_file):
	keypoints_all = []
	with open(input_file, 'r') as f:
		for line in f.readlines():
			line = line.split(' ')
			img = cv2.imread("images/" + line[0] + ".jpg")
			img = pre_treatment(img)
			img = extract_feature(img)
			if img is not None:
				keypoints_all.extend(img)
			else:
				print(line[0])
	return cluster(keypoints_all) 

def cluster(datapoints):
	kmeans = KMeans(init = 'k-means++', n_clusters = 15, n_init = 1000)
	res = kmeans.fit(datapoints)
	centroids = res.cluster_centers_
	return kmeans, centroids
	
def normalize(input_data):
	sum_input = np.sum(input_data)
	if sum_input > 0:
		return input_data / sum_input
	else:
		return input_data
		
def load_data(input_file, kmeans, centroids):
	X = []
	y = []
	with open(input_file, 'r') as f:
		for line in f.readlines():
			line = line.split(' ')
			img = cv2.imread("images/" + line[0] + ".jpg")
			img = construct_feature(img, kmeans, centroids)
			if img is not None:
				X.extend(img)
				y.append(int(line[1]))
			else:
				print(line[0])
	X = np.array(X)
	y = np.array(y)
	print(X.shape)
	print(y.shape)
	return X, y
		
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

def search_best(X_train, y_train):
	parameter_grid = [
		{'kernel': ['linear'], 'class_weight': ['balanced'], 'gamma': [0.01, 0.001], 'C': [0.5, 1, 10, 50]},
		{'kernel': ['poly'], 'class_weight': ['balanced'], 'gamma': [0.01, 0.001], 'degree': [2, 3]},
		{'kernel': ['rbf'], 'class_weight': ['balanced'], 'gamma': [0.01, 0.001], 'C': [0.5, 1, 10, 50]},
	]
	print("\n#### Searching optimal hyperparameters ####\n")
	classifier = GridSearchCV(SVC(), parameter_grid, cv = 5)
	classifier.fit(X_train, y_train)
	cv_result = pd.DataFrame.from_dict(classifier.cv_results_)
	with open('cv_result.csv', 'w') as f:
		cv_result.to_csv(f)
	print("\nHighest scoring parameter set:")
	print(classifier.best_params_)
	return classifier

def test(classifier, X_train, X_test, y_train, y_test):
	target_names = ['Class-' + str(i) for i in set(y)]
	print("\n##############################\n")
	print("\nClassifier performance on training dataset\n")
	print(classification_report(y_train, classifier.predict(X_train), target_names = target_names))
	print("\n##############################\n")
	print("\n##############################\n")
	print("\nClassifier performance on testing dataset\n")
	print(classification_report(y_test, classifier.predict(X_test), target_names = target_names))
	print("\n##############################\n")

kmeans, centroids = bow("data.txt")	
X, y = load_data("data.txt", kmeans, centroids)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)	
classifier = search_best(X_train, y_train)
test(classifier, X_train, X_test, y_train, y_test)
joblib.dump(kmeans, "kmeans.model")
joblib.dump(centroids, "centroids.model")
joblib.dump(classifier, "svm.model")

