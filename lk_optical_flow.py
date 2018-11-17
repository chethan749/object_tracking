import numpy as np
import cv2
from imutils.video import FPS

def label(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(old_frame, (x, y), 5, (0, 255, 0), -1)
        points.append([[x, y]])

points = []
cv2.namedWindow('Output')
cv2.setMouseCallback('Output', label)

cap = cv2.VideoCapture('ball.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
# cv2.imwrite('ball.jpg', old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

while True:
	cv2.imshow('Output', old_frame)
	if cv2.waitKey(1) == ord('q'):
		break
cv2.destroyAllWindows()

p0 = np.array(points, dtype = np.float32)
# p0 = np.array([[[np.float32(259), np.float32(172)]]])
# print(type(p0))
# print(p0)
# print(p0.shape)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
ret,frame = cap.read()
x = 1
fps = FPS()
fps.start()
while ret:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #print(p1, st, err)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    cv2.imwrite('optical_flow/frame{0:03d}.jpg'.format(x), img)
    ret,frame = cap.read()
    x += 1
    fps.update()
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

fps.stop()
print('FPS:', fps.fps())

cv2.destroyAllWindows()
cap.release()
