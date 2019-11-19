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
    eyesPoints = [[landmarks[2], landmarks[3]], 
                    [landmarks[1], landmarks[0]]]
    for (p1, p2) in eyesPoints:
        x1, y1 = p1
        x2, y2 = p2
        centerx = (x1 + x2) / 2
        centery = (y1 + y2) / 2
        diameter = int(math.hypot(x1 - x2, y1 - y2) * 1.2)
        eyecopy = cv2.resize(eyeimg, (diameter, diameter))
        roix = centerx - (diameter / 2)
        roiy = centery - (diameter / 2)
        srcroi = src[roiy:roiy + diameter, roix: roix + diameter]
        eyecolor = eyecopy[:, :, :3]
        eyealpha = eyecopy[:, :, 3]
        where1, where2 = np.where(eyealpha > 0)
        srcroi[where1, where2] = eyecolor[where1, where2]

detector = dlib.get_frontal_face_detector()
modelpath = 'model/shape_predictor_5_face_landmarks.dat' 
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
