import numpy as np
import cv2 as cv
from imutils.video import FPS

def label(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
        points.append((x, y))

points = []
cv.namedWindow('Output')
cv.setMouseCallback('Output', label)

cap = cv.VideoCapture('dataset2_resized.mp4')
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window

while True:
	cv.imshow('Output', frame)
	if cv.waitKey(1) == ord('q'):
		break
cv.destroyAllWindows()

(c, r), (x2, y2) = points
w, h = x2 - c, y2 - r
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

a = 1
fps = FPS()
fps.start()
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('img2',img2)
        #cv.imwrite('camshift/frame{0:03d}.jpg'.format(a), img2)
        fps.update()
        a += 1
        k = cv.waitKey(1) & 0xff
        if k == 27:
            break
    else:
        break
fps.stop()
print('FPS:', fps.fps())
cv.destroyAllWindows()
cap.release()
