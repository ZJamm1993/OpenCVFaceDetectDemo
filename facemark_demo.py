# coding:utf-8
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import numpy as np
from cv2 import cv2
import math

def RectContainsPoint(rect, point):
    x, y, w, h = rect
    px, py = point
    return px >= x and px < x + w and py >= y and py < y + h

def AlphaBlending(fgImg, bgImg, alphaMask):
    fgImg = fgImg.astype(np.float)
    bgImg = bgImg.astype(np.float)
    alphaMask = alphaMask.astype(np.float)
    fg = cv2.multiply(alphaMask, fgImg)
    bg = cv2.multiply(1.0 - alphaMask, bgImg)
    mix = cv2.add(fg, bg)
    return mix

def DrawLandmarks(src, landmarks):
    boundheight, boundwidth = src.shape[:2]
    bounds = (0, 0, boundwidth, boundheight)
    subdv = cv2.Subdiv2D(bounds)
    for (i, (x, y)) in enumerate(landmarks):
        cv2.circle(src, (x, y), 1, (255, 255, 0), 2)
        cv2.putText(src, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
    #     if RectContainsPoint(bounds, (x, y)):
    #         subdv.insert((x, y))
    # edges = subdv.getEdgeList()
    # for (i, (epx1, epy1, epx2, epy2)) in enumerate(edges):
    #     if RectContainsPoint(bounds, (epx1, epy1)) and RectContainsPoint(bounds, (epx2, epy2)):
    #         cv2.line(src, (epx1, epy1), (epx2, epy2), (0, 0, 255), 2)

def DrawCartoonEyes(src, landmarks):
    # 36, 39, 42, 45 if 68 marks
    # 2, 3, 1, 0 if 5 marks
    eyeimg = cv2.imread('face/eye.png', cv2.IMREAD_UNCHANGED)
    eyesPoints = [[landmarks[36], landmarks[39]], 
                    [landmarks[42], landmarks[45]]]
    for (p1, p2) in eyesPoints:
        x1, y1 = p1
        x2, y2 = p2
        # 计算ROI区域
        centerx = (x1 + x2) / 2
        centery = (y1 + y2) / 2
        diameter = int(math.hypot(x1 - x2, y1 - y2) * 2)
        roix = centerx - (diameter / 2)
        roiy = centery - (diameter / 2)
        # 前景图缩放至ROI大小
        eyecopy = cv2.resize(eyeimg, (diameter, diameter)).astype(np.float)
        srcroi = src[roiy:roiy + diameter, roix: roix + diameter]
        eyecolor = eyecopy[:, :, :3]
        eyechannel3 = eyecopy[:, :, 3] / 255.0
        eyealpha3 = np.zeros((diameter, diameter, 3))
        eyealpha3[:, :, 0] = eyechannel3
        eyealpha3[:, :, 1] = eyechannel3
        eyealpha3[:, :, 2] = eyechannel3
        mix = AlphaBlending(eyecolor, srcroi, eyealpha3)
        srcroi[:,:] = mix[:,:]

detector = dlib.get_frontal_face_detector()
modelpath = 'model/shape_predictor_5_face_landmarks.dat' 
modelpath = '../shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(modelpath)

vs = VideoStream(src=0).start()
while True:
    originframe = vs.read()
    if originframe is None:
        break
    
    originframe = imutils.resize(originframe, width = 600)
    originframe = np.fliplr(originframe)

    # progress
    frame = originframe.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        for rect in rects:
            # bx, by, bw, bh = face_utils.rect_to_bb(rect)
            # cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            # DrawLandmarks(frame, shape)
            DrawCartoonEyes(frame, shape)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
