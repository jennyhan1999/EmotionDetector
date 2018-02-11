import cv2
import dlib
import math
from EmoFunctions import *

import time
import numpy as np

# taken from https://github.com/opencv/opencv/tree/master/data/haarcascades
faceCas = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouthCas = cv2.CascadeClassifier('haarcascade_mouth.xml')

# taken from https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()

vidcap = cv2.VideoCapture(0)
ds_factor = 0.5

width = vidcap.get(3)  # float
height = vidcap.get(4)

factor = .5
adjust = int(1 / factor)


# Using OpenCV
def run_detection(data):
    _, frame = vidcap.read()
    # face detection on a smaller frame, faster
    loResFrame = cv2.resize(frame, dsize=(0, 0), fx=factor, fy=factor)
    pic = cv2.cvtColor(loResFrame, cv2.COLOR_BGR2GRAY)

    camera = cv2.VideoCapture(0)

    frame_dlib = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_dlib, width=120)

    detected = detector(frame_resized, 1)

    if len(detected) > 0:
        #using dlib
        for k, d in enumerate(detected):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(frame_resized, d)
            landmarkKey, coordKey, landmarks = shape_to_np(shape)
            calibrate(data, landmarkKey)
            averageCalibration(data)

            diffX, diffY = isMoved(data, landmarkKey)

            for i in landmarkKey:
                t = landmarkKey[i]
                x, y = t[0], t[1]
                newX, newY = x + diffX, y + diffY
                landmarkKey[i] = tuple([newX, newY])

            # calibration values
#            for i in data.calibrateDic:
#                x, y = data.calibrateDic[i]
#                cv2.putText(frame, str(i), (int(x / ratio), int(y / ratio)),
#                            cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (0, 0, 0))
            # emotions(data,landmarkKey,frame)
            emos = emotions(data, landmarkKey, frame)
            matching = matchingEmo(data, emos)

            # Using dlib
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in landmarks:
                landmarkVal = coordKey[(x, y)]
                #cv2.putText(frame, str(landmarkVal), (int(x / ratio), int(y / ratio)),
                #            cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255))
                cv2.circle(frame, (int(x / ratio), int(y / ratio)), 1, (255, 255, 255), -1)

    # modified from https://www.youtube.com/watch?v=88HdqNDQsEk
    faces = faceCas.detectMultiScale(pic, 1.3, 5)
    for (xf, yf, wf, hf) in faces:
        x, y, w, h = xf * adjust, yf * adjust, wf * adjust, hf * adjust
        # print rectangle on regular frame 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 50, 50), 2)
        drawEmoVals(data, x, y, w, h, matching, frame, emos)
        roi_gray = pic[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        break

    return frame


def isMoved(data, landmarkKey):
    calibrateDic = data.calibrateDic

    xTL, yTL = landmarkKey[21]
    xTR, yTR = landmarkKey[22]
    xB, yB = landmarkKey[8]
    CxTL, CyTL = calibrateDic[21]
    CxTR, CyTR = calibrateDic[22]
    CxB, CyB = calibrateDic[8]

    avgX = (xTL + xTR + xB) / 3
    avgY = (yTL + yTR + yB) / 3
    avgXC = (CxTL + CxTR + CxB) / 3
    avgYC = (CyTL + CyTR + CyB) / 3

    diffX = avgX - avgXC
    diffY = avgY - avgYC
    return diffX, diffY


'''
#what each T/F value in the tuple means in order
    1. eyebrow raised (up / 1 or down / 0)
    2. eyebrow together (yes/1 or no/0)
    3. eyes open (yes/1 or no/0)
    4. mouth open vertically small(yes / 1 or no / 0)
    5. mouth open vert Big (yes/1 or no/0)
    6. mouth open horizontally(yes / 1 or no / 0)
    7. mouth corners down (down/1 or not/0)
    8. one eyebrow up? (up/1 or no/0)
    9. brow slopes down (rly sloped /1 or neutral /0)
    10. brow slopes Up (up/1 or neutral /0)
'''


def emotionsDic(data):
    data.emotionsDic = dict()
    data.emotionsDic[tuple([False, False, True,
                            True, False, True,
                            False, False, False,
                            False])] = "Happy"
    data.emotionsDic[tuple([True, False, True,
                            False, True, True,
                            False, False, False,
                            False])] = "Suprised"
    data.emotionsDic[tuple([False, False, True,
                            False, False, False,
                            False, False, False,
                            False])] = "Neutral"
    data.emotionsDic[tuple([False, True, True,
                            False, True, True,
                            False, False, True,
                            False])] = "Anger"
    data.emotionsDic[tuple([False, True, False,
                            False, True, True,
                            True, False, False,
                            True])] = "Sad"
    print(data.emotionsDic)


def emotions(data, landmarkKey, frame):
    lEye, rEye = eyes(landmarkKey)
    eyesR = eyesOpen(data, rEye)
    eyesL = eyesOpen(data, lEye)
    if eyesR and eyesL:
        eye = True
        wink = False
    if (eyesR or eyesL) and not (eyesR and eyesL):
        eye = False
        wink = True
    if not (eyesR or eyesL):
        eye = False
        wink = False

    lBrow, rBrow = eyebrows(landmarkKey)
    browSlopeDR = eyebrowSlopeDown(data, rBrow)
    browSlopeDL = eyebrowSlopeDown(data, lBrow)

    if browSlopeDR and browSlopeDL:
        browD = True
    else:
        browD = False
    browSlopeUL = eyebrowSlopeUp(data, lBrow)
    browSlopeUR = eyebrowSlopeUp(data, rBrow)

    if browSlopeUL and browSlopeUR:
        browU = True
    else:
        browU = False

    browR = eyebrowsRaised(data, rBrow)
    browL = eyebrowsRaised(data, lBrow)

    if browR and browL:
        browUp = True
        oneBrowUp = False
    if (browR or browL) and not (browR and browL):
        browUp = False
        oneBrowUp = True
    if not (browR or browL):
        browUp = False
        oneBrowUp = False

    eyebrowT = eyebrowsTogether(data, lBrow, rBrow)

    m = mouth(landmarkKey)

    mouthOVS = mouthOpenVertSmall(data, m)
    mouthOVB = mouthOpenVertBig(data, m)
    mouthOH = mouthOpenHoriz(data, m)
    frown = mouthCornerDown(data, m)

    tpl = tuple([browUp, eyebrowT, eye, mouthOVS, mouthOVB, mouthOH, frown, oneBrowUp, browD, browU])
    return tpl


def matchingEmo(data, currentEmo):
    matchingDic = dict()
    count = 0
    for key in data.emotionsDic:
        emotion = data.emotionsDic[key]
        for i in range(len(key)):
            setEmo = key[i]
            curEmo = currentEmo[i]
            if setEmo == curEmo:
                count += 1
            else:
                continue
        matchingDic[emotion] = count
        count = 0
    return matchingDic


def highestEmo(emoDic, currentEmo):
    count = 0
    highestVal = 0
    highestEmo = ""
    for emotion in emoDic:
        val = emoDic[emotion]
        if val > highestVal:
            highestVal = val
            highestEmo = str(emotion)
        if emotion == highestEmo:
            if currentEmo[8] and currentEmo[4]:
                highestEmo = "Angry"
            if currentEmo[0]:
                highestEmo = "Suprised"
            if currentEmo[3]:
                highestEmo = "Happy"
            if currentEmo[3] == False and currentEmo[4] == False:
                highestEmo = "Neutral"
            if currentEmo[2] == False:
                highestEmo = "Sad"
    return highestEmo


# drawing emotion box ranges
def drawEmoVals(data, x, y, wF, wH, matching, img, emoDic):
    w = 100
    h = 10
    margin = 20
    moveL = 50
    moveR = wF + w + 2 * margin
    textM = 25
    count = 0
    highEmo = highestEmo(matching, emoDic)
    for i in matching:
        count += 1
        val = matching[i]
        totalLen = 10
        percent = (val / totalLen) * 100
        newW = int((w / totalLen) * val)
        font = cv2.FONT_HERSHEY_PLAIN

        if count % 2 == 1:  # left
            cv2.rectangle(img, (x - margin, y - margin + moveL * count),
                          (x - margin - w, y - margin + moveL * count - h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - margin - w, y - margin + moveL * count),
                          (x - margin + newW - w, y - margin + moveL * count - h), (0, 255, 0), -1)
            cv2.putText(img, "%0.2f" % (percent) + " % " + str(i),
                        (x - margin - newW, y - margin + moveL * count - h + textM), font, 1, (0, 255, 0), 1,
                        cv2.LINE_AA)

        else:  # right
            cv2.rectangle(img, (x - margin + moveR - w, y - margin + moveL * (count - 1) - h),
                          (x - margin + moveR, y - margin + moveL * (count - 1)), (0, 255, 0), 3)
            cv2.rectangle(img, (x - margin + moveR + newW - w, y - margin + moveL * (count - 1) - h),
                          (x - margin + moveR - w, y - margin + moveL * (count - 1)), (0, 255, 0), -1)
            cv2.putText(img, "%0.2f" % (percent) + " % " + str(i),
                        (x - margin + moveR - w, y - margin + moveL * (count - 1) - h + textM),
                        font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(img, "You are most likely feeling " + highEmo, (x - int(wF) // 3, y - margin),
                font, 2, (0, 255, 0), 2, cv2.LINE_AA)


# taken from https://gist.github.com/nikgens/da582d745fa2bf0ddd8f5f7480042291
def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


# modified from https://gist.github.com/nikgens/da582d745fa2bf0ddd8f5f7480042291
def shape_to_np(shape, dtype="int"):
    coordKey = dict()
    landmarkKey = dict()
    # initialize the list of (x, y)-coordinates
    landmarks = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        xVal = shape.part(i).x
        yVal = shape.part(i).y
        landmarks[i] = (xVal, yVal)
        #        print("i is ", i, "val is ", landmarkKey[i])
        #        print("xy val is ", xVal, yVal)
        landmarkKey[i] = (xVal, yVal)
        coordKey[(xVal, yVal)] = i
    # return the list of (x, y)-coordinates
    return landmarkKey, coordKey, landmarks


def init(data):
    data.gameScreen = 0
    data.timerCount = 0
    data.emoChange = 0
    data.cDic = dict()
    data.calibrateDic = dict()
    data.img = None
    data.cDic = dict()
    emotionsDic(data)

def firstScreenEmos(canvas,data):
    if data.emoChange % 5 == 0:
        canvas.create_oval(data.width//3,data.height//3,
                           data.width*2//3,data.height*2//3,  fill = "yellow")
        canvas.create_oval(data.width*4//9,data.height*4//9,
                           data.width*4.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_oval(data.width*5//9,data.height*4//9,
                           data.width*5.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_arc(data.width*4//9, data.height*4.5//9,
                          data.width*5.2//9 ,data.height*5.2//9,
                          start = 0, extent = -180,outline= "black",
                            width=5, style="arc")
    if data.emoChange %5 == 1:
        canvas.create_oval(data.width//3,data.height//3,
                           data.width*2//3,data.height*2//3,  fill = "blue")
        canvas.create_line(data.width*1.7//3, data.height*1.2//3, data.width*1.9//3,data.height*1.35//3, width = 5)
        canvas.create_line(data.width*1.3//3, data.height*1.35//3, data.width*1.5//3,data.height*1.2//3, width = 5)
        canvas.create_oval(data.width*4//9,data.height*4//9,
                           data.width*4.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_oval(data.width*5//9,data.height*4//9,
                           data.width*5.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_arc(data.width*4//9, data.height*5//9,
                          data.width*5.2//9 ,data.height*5.7//9,
                          start = 0, extent = 180,outline= "black",
                            width=5, style="arc")
    if data.emoChange %5 == 3:
        canvas.create_oval(data.width//3,data.height//3,
                           data.width*2//3,data.height*2//3,  fill = "red")
        canvas.create_line(data.width*1.3/3,data.height*1.2//3,data.width*1.5//3, data.height*1.35//3, width = 5)
        canvas.create_line(data.width*1.8//3,data.height*1.2//3,data.width*1.6//3, data.height*1.35//3,  width = 5)
        canvas.create_oval(data.width*4//9,data.height*4//9,
                           data.width*4.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_oval(data.width*5//9,data.height*4//9,
                           data.width*5.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_arc(data.width*4//9, data.height*5//9,
                          data.width*5.2//9 ,data.height*5.7//9,
                          start = 0, extent = 180,outline= "black",
                            width=5, style="arc")
    if data.emoChange %5 == 4:
        canvas.create_oval(data.width//3,data.height//3,
                           data.width*2//3,data.height*2//3,  fill = "yellow")
        canvas.create_line(data.width*1.7//3, data.height*1.2//3, data.width*1.9//3,data.height*1.35//3, width = 5)
        canvas.create_line(data.width*1.3//3, data.height*1.35//3, data.width*1.5//3,data.height*1.2//3, width = 5)
        canvas.create_oval(data.width*4//9,data.height*4//9,
                           data.width*4.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_oval(data.width*5//9,data.height*4//9,
                           data.width*5.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_oval(data.width*4//9, data.height*5//9,
                          data.width*5.2//9 ,data.height*5.7//9, fill = "black")
    if data.emoChange %5 == 2:
        canvas.create_oval(data.width//3,data.height//3,
                           data.width*2//3,data.height*2//3,  fill = "yellow")
        canvas.create_oval(data.width*4//9,data.height*4//9,
                           data.width*4.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_oval(data.width*5//9,data.height*4//9,
                           data.width*5.3//9,data.height*4.3//9,  fill = "black")
        canvas.create_line(data.width*4//9, data.height*5//9,
                          data.width*5.2//9 ,data.height*5//9, fill = "black", width = 5)

def firstScreen(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height, fill = "light blue")
    firstScreenEmos(canvas,data)
    canvas.create_text(data.width // 2, data.height // 5,
                       text="Welcome to the Emotion Detector!",
                       fill="Black", font="Times 30 bold")
    canvas.create_text(data.width // 2, data.height * 3 // 4,
                       text="Press 'spacebar' to start",
                       fill="Black", font="Times 25 bold")


def secondScreen(canvas, data):
    canvas.create_rectangle(0,0,data.width,data.height, fill = "light blue")

    canvas.create_text(data.width // 2, data.height // 4,
                       text="Come close to the camera, stay still, keep a neutral face",
                       fill="Black", font="Times 25 bold")
    canvas.create_text(data.width // 2, data.height * 1.2 // 4,
                       text="for calibration, then make expressions!",
                       fill="Black", font="Times 25 bold")
    canvas.create_text(data.width // 2, data.height * 2 // 4,
                       text="Press 'spacebar' to continue",
                       fill="Black", font="Times 25 bold")


# create dict of all average values that the persons resting face is on to compare
def calibrate(data, landmarkKey):
    if data.timerCount <= 10:
        data.calibrationText = True
        for lndmrk in landmarkKey:
            if lndmrk not in data.cDic:
                s = set()
                vals = landmarkKey[lndmrk]
                s.add(vals)
                data.cDic[lndmrk] = s
            else:
                s = set()
                s = data.cDic[lndmrk]  # adds what is already in calibrate dict
                t = landmarkKey[lndmrk]
                s.add(t)
                data.cDic[lndmrk] = s
    data.calibrationText = False
    return (data.cDic)


def averageCalibration(data):
    for i in data.cDic:
        tuples = data.cDic[i]
        xAll = 0
        yAll = 0
        count = 0
        for tpl in tuples:
            x, y = tpl[0], tpl[1]
            xAll += x
            yAll += y
            count += 1
        xAvg = xAll / count
        yAvg = yAll / count
        cTuple = tuple([xAvg, yAvg])
        data.calibrateDic[i] = cTuple
    return data.calibrateDic


#### TKINTER####

# framework from https://github.com/VasuAgrawal/112-opencv-tutorial/blob/master/opencvTkinterTemplate.py
# modified
import time
import sys
from tkinter import *

import numpy as np
import cv2
from PIL import Image, ImageTk


def opencvToTk(frame):
    """Convert an opencv image to a tkinter image, to display in canvas."""
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    tk_image = ImageTk.PhotoImage(image=pil_img)
    return tk_image


def mousePressed(event, data):
    pass


def keyPressed(event, data):
    if event.keysym == "q":
        data.root.destroy()
    if event.keysym == "space":
        data.gameScreen += 1
        data.timerCount = 0


def timerFired(data):
    if data.gameScreen == 0:
        data.timerCount += 1
        if data.gameScreen == 0:
            if data.timerCount % 15 == 0:
                data.emoChange += 1
        data.timerCount += 1
    if data.gameScreen > 1:
        data.timerCount += 1
        data.frame = run_detection(data)


def drawCamera(canvas, data):
    data.tk_image = opencvToTk(data.frame)
    canvas.create_image(data.width / 2, data.height / 2, image=data.tk_image)


def redrawAll(canvas, data):
    if data.gameScreen == 0:
        firstScreen(canvas, data)
    if data.gameScreen == 1:
        secondScreen(canvas, data)
    if data.gameScreen > 1:
        try:
            drawCamera(canvas, data)
        except:
            print("got here")


def run(width=300, height=300):
    class Struct(object): pass

    data = Struct()
    data.width = width
    data.height = height
    data.camera_index = 0

    data.timer_delay = 50  # ms
    data.redraw_delay = 50  # ms
    init(data)

    # Make tkinter window and canvas
    data.root = Tk()
    canvas = Canvas(data.root, width=data.width, height=data.height)
    canvas.pack()

    # Basic bindings. Note that only timer events will redraw.
    data.root.bind("<Button-1>", lambda event: mousePressed(event, data))
    data.root.bind("<Key>", lambda event: keyPressed(event, data))

    # Timer fired needs a wrapper. This is for periodic events.
    def timerFiredWrapper(data):
        # Ensuring that the code runs at roughly the right periodicity
        start = time.time()
        timerFired(data)
        redrawAllWrapper(canvas, data)
        end = time.time()
        diff_ms = (end - start) * 1000
        delay = int(max(data.timer_delay - diff_ms, 0))
        data.root.after(delay, lambda: timerFiredWrapper(data))

    # Wait a timer delay before beginning, to allow everything else to
    # initialize first.
    data.root.after(data.timer_delay,
                    lambda: timerFiredWrapper(data))

    def redrawAllWrapper(canvas, data):
        start = time.time()

        # Redrawing code
        canvas.delete(ALL)
        redrawAll(canvas, data)
        canvas.update()

        # Calculate delay accordingly
        end = time.time()
        diff_ms = (end - start) * 1000

        # Have at least a 5ms delay between redraw. Ideally higher is better.
        delay = int(max(data.redraw_delay - diff_ms, 5))

    # Start drawing immediately
    data.root.after(0, lambda: redrawAllWrapper(canvas, data))

    # Loop tkinter
    data.root.mainloop()

    # Once the loop is done, release the camera.
    print("Releasing camera!")
    #    data.camera.release()
    vidcap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run(800, 800)