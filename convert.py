import cv2
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', help = 'Path to video file', type = str)
parser.add_argument('-f', '--frames', help = 'Path to folder containing frames', type = str)
parser.add_argument('output', help = 'Path to video file', type = str)
parser.add_argument('-c', '--codec', type = str, default = 'mp4v', help = 'codec of output video')
parser.add_argument('--fps', type = int, default = 24, help = 'Frame rate of video')
args = vars(parser.parse_args())

if args['video'] is not None:
	cap = cv2.VideoCapture(args['video'])
	
	ret, img = cap.read()
	
	video = None
	codec = fourcc = cv2.VideoWriter_fourcc(*args['codec'])
	
	while ret:
		cv2.imshow('Feed', img)
		if video is None:
		    video = cv2.VideoWriter(args['output'], codec, args['fps'], (img.shape[1], img.shape[0]), True)

		video.write(img)
		
		ret, img = cap.read()
	
	cap.release()
	cv2.destroyAllWindows()
	
elif args['frames'] is not None:
	path_split = args['frames'].split(os.path.sep)
	if path_split[-1] == '':
		del path_split[-1]

	path_split.append('*')
	paths = os.path.sep.join(path_split)
	paths = glob.glob(paths)
	paths = sorted(paths, key = lambda x: int(x.split(os.path.sep)[-1].split('.')[0][5: ]))

	video = None
	codec = fourcc = cv2.VideoWriter_fourcc(*args['codec'])
	for i in paths:
		img = cv2.imread(i)
		cv2.imshow('Feed', img)
		if video is None:
		    video = cv2.VideoWriter(args['output'], codec, args['fps'], (img.shape[1], img.shape[0]), True)

		video.write(img)

		if cv2.waitKey(1) == ord('q'):
		    break

	cv2.destroyAllWindows()

else:
	print('Provide path to either directory containing frames or video file')
