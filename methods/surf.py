import cv2
import numpy as np
from imutils.video import FPS

MIN_MATCH_COUNT=1

detector = cv2.xfeatures2d.SURF_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("dataset2.png")
trainImg = cv2.resize(trainImg, (500, int(trainImg.shape[0] / trainImg.shape[1] * 500)))
trainKP,trainDesc=detector.detectAndCompute(trainImg, None)
ball_surf = cv2.drawKeypoints(trainImg,trainKP,None,(255,0,0),4)

cv2.imshow('rpi', ball_surf)
a = 0

fps = FPS()
cam=cv2.VideoCapture('dataset2_resized.mp4')
ret, QueryImgBGR=cam.read()
fps.start()
while ret:
	QueryImgBGR = cv2.resize(QueryImgBGR, (800, int(QueryImgBGR.shape[0] / QueryImgBGR.shape[1] * 800)))
	QueryImg=QueryImgBGR.copy()
	queryKP,queryDesc=detector.detectAndCompute(QueryImg, None)
	matches=flann.knnMatch(queryDesc,trainDesc,k=2)
	matchesMask = [[0,0] for i in range(len(matches))]
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.7*n.distance:
			matchesMask[i]=[1,0]

	draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

	img3 = cv2.drawMatchesKnn(QueryImg,queryKP,trainImg,trainKP,matches,None,**draw_params)

	cv2.imshow('result',img3)
	cv2.imwrite('./surf/rpi/img' + str(a).zfill(6) + '.jpg', img3)
	ret, QueryImgBGR=cam.read()
	fps.update()
	a += 1
	if cv2.waitKey(1)==ord('q'):
		break

fps.stop()
print('FPS:', fps.fps())

cam.release()
cv2.destroyAllWindows()
