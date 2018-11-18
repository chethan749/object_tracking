import cv2 as cv
import numpy as np
from imutils.video import FPS

# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

#Performs required image processing to get ball coordinated in the video
class ProcessImage:

    def DetectObject(self):

        vid = cv.VideoCapture('ball.mp4')

        if(vid.isOpened() == False):
            print('Cannot open input video')
            return

        width = int(vid.get(3))
        height = int(vid.get(4))

        # Create Kalman Filter Object
        kfObj = KalmanFilter()
        predictedCoords = np.zeros((2, 1), np.float32)
        a = 1
        fps = FPS()
        fps.start()
        while(vid.isOpened()):
            rc, frame = vid.read()

            if(rc == True):
                [ballX, ballY], flag = self.DetectBall(frame)

                if flag == 1:
                    continue

                ballX = int(ballX)
                ballY = int(ballY)
                predictedCoords = kfObj.Estimate(ballX, ballY)

                # Draw Actual coords from segmentation
                cv.circle(frame, (ballX, ballY), 10, [0, 0, 255], 2, 8)

                # Draw Kalman Filter Predicted output
                cv.circle(frame, (predictedCoords[0], predictedCoords[1]), 10, [255, 0, 0], 2, 8)
                cv.imshow('Input', frame)
                #cv.imwrite('kalman/frame{0:03d}.jpg'.format(a), frame)
                a += 1
                fps.update()
                if (cv.waitKey(1) & 0xFF == ord('q')):
                    break

            else:
                break
        
        fps.stop()
        print('FPS:', fps.fps())
        vid.release()
        cv.destroyAllWindows()

    # Segment the green ball in a given frame
    def DetectBall(self, frame):
        flag = 0
        ballX = ballY = None
        # Set threshold to filter only green color & Filter it
        lowerBound = np.array([90,30,0], dtype = "uint8")
        upperBound = np.array([255,255,90], dtype = "uint8")
        greenMask = cv.inRange(frame, lowerBound, upperBound)

        # Dilate
        kernel = np.ones((5, 5), np.uint8)
        greenMaskDilated = cv.dilate(greenMask, kernel)
        cv.imshow('Thresholded', greenMaskDilated)

        # Find ball blob as it is the biggest green object in the frame
        [nLabels, labels, stats, centroids] = cv.connectedComponentsWithStats(greenMaskDilated, 8, cv.CV_32S)

        # First biggest contour is image border always, Remove it
        stats = np.delete(stats, (0), axis = 0)
        try:
            maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(stats.argmax(), stats.shape)
        except ValueError:
            flag = 1
            pass

        if flag != 1:
            # This is our ball coords that needs to be tracked
            ballX = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2]/2)
            ballY = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3]/2)

        return [ballX, ballY], flag


#Main Function
def main():
    processImg = ProcessImage()
    processImg.DetectObject()


if __name__ == "__main__":
    main()

print('Program Completed!')
