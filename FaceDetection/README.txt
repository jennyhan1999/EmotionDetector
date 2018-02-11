Hi!

This program detects Happy, Sad, Angry, Neutral, and Surprised facial expressions. It will first calibrate to the person’s neutral face. Then it matches the person’s movement to emotions and display’s them.

This program needs OpenCV and dlib in order to run on PyCharm using Python3. It must also have “haarcascade_mouth.xml”, “haarcascade_frontalface_default.xml”, and “shape_predictor_68_face_landmarks.dat” in the same folder as the facialDetect file.

To download OpenCV, type into terminal: 
	1. pip install opencv-python
 	2. import cv2

To download dlib, type into terminal:
	1. must have boost:
		brew install python
		brew install boost-python --with-python3 --without-python				brew install boost-python
		brew list | grep 'boost' (check)
			should print:
				boost
				boost-python
	2. must have cMake:
		https://cmake.org/download/
	3. pip3 install dlib

Link to video: https://www.youtube.com/watch?v=_wlW6pTVPlU

Good Luck!