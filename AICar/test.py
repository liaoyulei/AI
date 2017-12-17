import cv2

def extract_feature(img):
	keypoints = cv2.xfeatures2d.StarDetector_create().detect(img)
	keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img, keypoints)
	return descriptors

img = cv2.imread("images/d-12.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#灰度图
img = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow('Gaussian', img)
cv2.waitKey()
img = cv2.Canny(img, 1, 255)#需调整 
cv2.imshow('Canny', img)
cv2.waitKey()
keypoints = cv2.xfeatures2d.StarDetector_create().detect(img)
keypoints, descriptors = cv2.xfeatures2d.SIFT_create().compute(img, keypoints)
print(keypoints)
print(descriptors)
print(descriptors.shape)
