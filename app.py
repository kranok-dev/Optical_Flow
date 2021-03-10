# Based on code from: https://github.com/khushboo-agarwal/Optical-Flow

from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import time
import math
import cv2

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def opticalFlow(img1,img2,frameIndex,totalFrames):
	h,w = img1.shape[:2]
	colorImage1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

	img1G = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	img2G = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	img1 = np.array(img1G)
	img2 = np.array(img2G)

	img1_smooth = cv2.GaussianBlur(img1,(3,3),0)
	img2_smooth = cv2.GaussianBlur(img2,(3,3),0)
		
	Ix = signal.convolve2d(img1_smooth,[[-0.25, 0.25],[-0.25, 0.25]],'same') + signal.convolve2d(img2_smooth,[[-0.25, 0.25],[-0.25, 0.25]],'same')
	Iy = signal.convolve2d(img1_smooth,[[-0.25,-0.25],[ 0.25, 0.25]],'same') + signal.convolve2d(img2_smooth,[[-0.25,-0.25],[ 0.25, 0.25]],'same')
	It = signal.convolve2d(img1_smooth,[[ 0.25, 0.25],[ 0.25, 0.25]],'same') + signal.convolve2d(img2_smooth,[[-0.25,-0.25],[-0.25,-0.25]],'same')

	features = cv2.goodFeaturesToTrack(img1_smooth,10000,0.01,10)	
	feature = np.int0(features)

	u = np.nan*np.ones((h,w))
	v = np.nan*np.ones((h,w))
	
	for l in feature:
		j,i = l.ravel()
		
		IX,IY,IT = [],[],[]
		
		if(i+2 < h and i-2 > 0 and j+2 < w and j-2 > 0):
			for b1 in range(-2,3):
				for b2 in range(-2,3):
					IX.append(Ix[i+b1,j+b2])
					IY.append(Iy[i+b1,j+b2])
					IT.append(It[i+b1,j+b2])
					
			LK = (IX,IY)
			LK = np.matrix(LK)
			LK_T = np.array(np.matrix(LK))
			LK = np.array(np.matrix.transpose(LK)) 
			
			A1 = np.dot(LK_T,LK)
			A2 = np.linalg.pinv(A1)
			A3 = np.dot(A2,LK_T)
			
			(u[i,j],v[i,j]) = np.dot(A3,IT)
	
	fig = plt.figure('')
	plt.subplot(1,1,1)
	plt.axis('off')
	plt.imshow(colorImage1, cmap = 'gray')
	for i in range(h):
		for j in range(w):
			if abs(u[i,j]) > t or abs(v[i,j]) > t:
				plt.arrow(j,i,1.5*(-1*u[i,j]),1.5*(-1*v[i,j]), head_width = 3, head_length = 3, color = 'red')
	
	print('\r({:4}/{:4}) - Time Elapsed: {:10.10} seconds'.format(frameIndex+1,totalFrames,time.time()-start), end='')

	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	plt.close()

	return img

#------------------------------------------------------------------------------------------------------------------------------------------------------------------

start = time.time()
t = 0.7
video = cv2.VideoCapture('video.mp4')
totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

_,image1 = video.read()
ret,image2 = video.read()
imageCounter = 0

videoName = 'optical_flow.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
first = True

while(ret):
	image = opticalFlow(image1,image2,imageCounter,totalFrames)
	
	if(not imageCounter):
		height,width = image.shape[:2]
		videoOut = cv2.VideoWriter(videoName, fourcc, fps, (width,height))

	videoOut.write(image)
	cv2.imshow("Optical Flow",image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	
	image1 = image2.copy()
	ret,image2 = video.read()
	imageCounter += 1

print("")
video.release()
videoOut.release()