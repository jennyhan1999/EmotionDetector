import math

def slope(x1, y1, x2, y2):
    try:
        return (y2-y1)/(x2-x1)
    except:
        pass

def distance(x1, y1, x2, y2):
    return (((x1+x2)**2)+ ((y1+y2)**2))**.5

def eyebrows(dic):
    rightBrow = dict()
    leftBrow = dict()
    for lndmrk in dic:
        if lndmrk >= 17 and lndmrk <= 21:
            leftBrow[lndmrk] = dic[lndmrk]
        if lndmrk >= 22 and lndmrk <= 26:
            rightBrow[lndmrk] = dic[lndmrk]
#    print(rightBrow)
    return leftBrow, rightBrow

def eyebrowsTogether(data,lBrow, rBrow):
    avgXL = 0
    avgXR = 0

    avgXLC = 0
    avgXRC = 0

    for i in lBrow:
        t = lBrow[i]
        t2 = data.calibrateDic[i]
        x,y = t[0],t[1]
        xC,yC = t2[0],t2[1]
        avgXL += x
        avgXLC += xC

    for i in rBrow:
        t = rBrow[i]
        t2 = data.calibrateDic[i]
        x,y = t[0],t[1]
        xC,yC = t2[0],t2[1]
        avgXR += x
        avgXRC += xC

    avgPointL = avgXL/5
    avgPointR = avgXR/5
    avgPointLC = avgXLC/5
    avgPointRC = avgXRC/5

    dist = avgPointR - avgPointL
    distC = avgPointRC - avgPointLC
    if dist + .75 < distC:
#        print("fustrated")
        return True
    else:
#        print("normal")
        return False

def eyebrowsRaised(data,brow):
    avgY = 0
    avgYC = 0
    for i in brow:
        t = brow[i]
        t2 = data.calibrateDic[i]
        x,y = t[0],t[1]
        xC,yC = t2[0],t2[1]
        avgY += y
        avgYC += yC
    avgLine = avgY/5
    avgLineC = avgYC/5
    if avgLine < avgLineC * .95:
#        print("raisedBrow")
        return True
    else:
#        print("normal")
        return False

def eyes(d):
    rightEye = dict()
    leftEye = dict()

    for lndmrk in d:
        if lndmrk >= 36 and lndmrk <= 41:
            leftEye[lndmrk] = d[lndmrk]
        if lndmrk >= 42 and lndmrk <= 47:
            rightEye[lndmrk] = d[lndmrk]
    eyes = tuple([leftEye, rightEye])
            #    print(rightBrow)
    return eyes

def eyesOpen(data,eye):
    count = 0
    top = 0
    bottom = 0
    topC = 0
    bottomC = 0
    for i in eye:
        tpl1 = eye[i]
        x1,y1 = tpl1
        tpl2 = data.calibrateDic[i]
        x2,y2 = tpl2
        count += 1
        if count == 2 or count == 3:
            top += y1
            topC += y2
        if count == 5 or count == 6:
            bottom += y1
            bottomC += y2
    eyeDist = bottom - top
    eyeRefDist = bottomC - topC
    if eyeDist <= eyeRefDist/2:
#        print("eyes closed")
        return False
    else:
#        print ("eyes open")
        return True

def eyebrowSlopeDown(data, brow): #brow is a dict
    midpoint = 0
    for i in brow:
        if i == 19:
            t1 = brow[i]
            x1,y1 = t1[0],t1[1]
            t1C = data.calibrateDic[i]
            x1C,y1C = t1C[0],t1C[1]
        if i == 21:
            t2 = brow[i]
            x2,y2 = t2[0],t2[1]
            t2C = data.calibrateDic[i]
            x2C,y2C = t2C[0],t2C[1]
        else:
            if i == 22:
                t1 = brow[i]
                x1, y1 = t1[0], t1[1]
                t1C = data.calibrateDic[i]
                x1C, y1C = t1C[0], t1C[1]
            if i ==24:
                t2 = brow[i]
                x2, y2 = t2[0], t2[1]
                t2C = data.calibrateDic[i]
                x2C, y2C = t2C[0], t2C[1]
    s = slope(x1,y1,x2,y2)
    sC = slope(x1C,y1C,x2C,y2C)

    if s > 0:
        if s > sC:
            return True
        else:
            return False
    if s < 0:
        if s < sC:
            return True
        else:
            return False

def eyebrowSlopeUp(data, brow): #brow is a dict
    midpoint = 0
    for i in brow:
        if i == 17:
            t1 = brow[i]
            x1,y1 = t1[0],t1[1]
            t1C = data.calibrateDic[i]
            x1C,y1C = t1C[0],t1C[1]
        if i == 21:
            t2 = brow[i]
            x2,y2 = t2[0],t2[1]
            t2C = data.calibrateDic[i]
            x2C,y2C = t2C[0],t2C[1]
        else:
            if i == 22:
                t1 = brow[i]
                x1, y1 = t1[0], t1[1]
                t1C = data.calibrateDic[i]
                x1C, y1C = t1C[0], t1C[1]
            if i ==26:
                t2 = brow[i]
                x2, y2 = t2[0], t2[1]
                t2C = data.calibrateDic[i]
                x2C, y2C = t2C[0], t2C[1]
    s = slope(x1,y1,x2,y2)
    sC = slope(x1C,y1C,x2C,y2C)

    if s > 0:
        if s > sC:
            return True
        else:
            return False
    if s < 0:
        if s < sC:
            return True
        else:
            return False

def nose(landmarkKey):
    nose = dict()
    for lndmrk in landmarkKey:
        if lndmrk >= 27 and lndmrk <= 35:
            nose[lndmrk] = landmarkKey[lndmrk]
    return nose

def noseScrunch(data, nose): # nose is a dict
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x1C = 0
    y1C = 0
    x2C = 0
    y2C = 0
    for i in nose:
        if i >= 30:
            x,y = nose[i]
            y1 += y
            xC,yC = data.calibrateDic[i]
            x1C += xC
            y1C += yC
    yAvg = y1/6
    yCAvg = y1C/6

    if yAvg *1.01< yCAvg:
#        print("Scrunch!")
        return True
    else:
#        print("normal!")
        return False

def mouth(d):
    m = dict()
    for lndmrk in d:
        if lndmrk >= 48 and lndmrk <= 67:
            m[lndmrk] = d[lndmrk]
    return m

def mouthOpenVertSmall(data, mouth):
    topLip = dict()
    avgYTL = 0 #Y top lip
    avgYTLC = 0 #Y Top Lip Calibrate

    bottomLip = dict()
    avgYBL = 0
    avgYBLC = 0
    for i in mouth:
        if i <= 54:
            topLip[i] = mouth[i]
        if i >= 54 and i <= 59 or i == 48:
            bottomLip[i] = mouth[i]
    for i in topLip:
        t = topLip[i]
        x,y = t[0],t[1]
        avgYTL += y
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgYTLC+= y2
    for i in bottomLip:
        t = bottomLip[i]
        x,y = t[0],t[1]
        avgYBL += y
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgYBLC+= y2

    YTL = avgYTL/7
    YBL = avgYBL/7

    YTLC = avgYTLC/7
    YBLC = avgYBLC/7

    dist = YBL - YTL
    distC = YBLC - YTLC

    if dist> distC and dist < distC * 2:
#        print("Opened")
        return True
    else:
#        print("Closed")
        return False

def mouthOpenVertBig(data, mouth):
    topLip = dict()
    avgYTL = 0 #Y top lip
    avgYTLC = 0 #Y Top Lip Calibrate

    bottomLip = dict()
    avgYBL = 0
    avgYBLC = 0
    for i in mouth:
        if i <= 54:
            topLip[i] = mouth[i]
        if i >= 54 and i <= 59 or i == 48:
            bottomLip[i] = mouth[i]
    for i in topLip:
        t = topLip[i]
        x,y = t[0],t[1]
        avgYTL += y
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgYTLC+= y2
    for i in bottomLip:
        t = bottomLip[i]
        x,y = t[0],t[1]
        avgYBL += y
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgYBLC+= y2

    YTL = avgYTL/7
    YBL = avgYBL/7

    YTLC = avgYTLC/7
    YBLC = avgYBLC/7

    dist = YBL - YTL
    distC = YBLC - YTLC

    if dist> distC*2:
#        print("Opened")
        return True
    else:
#        print("Closed")
        return False

def mouthOpenHoriz(data,mouth):
    L = dict()
    R = dict()

    avgL = 0
    avgLC = 0
    avgR = 0
    avgRC =0
    for i in mouth:
        if i == 48 or i == 49 or i == 59 or i == 60:
            L[i] = mouth[i]
        if i == 53 or i == 54 or i == 55 or i == 64:
            R[i] = mouth[i]
    for i in L:
        t = L[i]
        x,y = t[0],t[1]
        avgL += x
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgLC += x2
    for i in R:
        t = R[i]
        x,y = t[0],t[1]
        avgR += x
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgRC += x2
    XL = avgL/4
    XR = avgR/4
    XLC = avgLC / 4
    XRC = avgRC / 4

    dist = XR - XL
    distC = XRC - XLC

    if dist > distC:
#        print("mouth open Horiz")
        return True
    else:
#        print("closed")
        return False

def mouthCornerDown(data,mouth):
    L = dict()
    R = dict()
    avgL = 0
    avgLC = 0
    avgR = 0
    avgRC = 0
    for i in mouth:
        if i == 48 or i == 49 or i == 59 or i == 60:
            L[i] = mouth[i]
        if i == 53 or i == 54 or i == 55 or i == 64:
            R[i] = mouth[i]
    for i in L:
        t = L[i]
        x,y = t[0],t[1]
        avgL += y
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgLC += y2
    for i in R:
        t = R[i]
        x,y = t[0],t[1]
        avgR += y
        t2 = data.calibrateDic[i]
        x2,y2 = t2[0],t2[1]
        avgRC += y2
    avgLVal = (avgL/4)
    avgRVal = (avgR/4)

    Y = (avgLVal + avgRVal)/2

    avgLCVal = (avgLC / 4)
    avgRCVal = (avgRC / 4)

    YC = (avgLCVal + avgRCVal) / 2

    if Y > YC:
#        print("frown :(")
        return True
    else:
#        print("not frown")
        return False
