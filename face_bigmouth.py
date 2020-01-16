# coding:utf-8
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import numpy as np
from cv2 import cv2
import math

def AlphaBlending(fgImg, bgImg, alphaMask):
    fgImg = fgImg.astype(np.float)
    bgImg = bgImg.astype(np.float)
    alphaMask = alphaMask.astype(np.float) / 255.0
    fg = cv2.multiply(alphaMask, fgImg)
    bg = cv2.multiply(1.0 - alphaMask, bgImg)
    mix = cv2.add(fg, bg)
    mix = mix.astype(np.uint8)
    return mix

def DrawLandmarks(src, landmarks):
    boundheight, boundwidth = src.shape[:2]
    bounds = (0, 0, boundwidth, boundheight)
    for (i, (x, y)) in enumerate(landmarks):
        cv2.circle(src, (x, y), 1, (255, 255, 0), 2)
        cv2.putText(src, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)

def handleFrameShape(frame, shape):
    # DrawLandmarks(frame, shape)
    outerlipspoint = shape[48:60]
    height, width = frame.shape[0:2]

    rect = cv2.boundingRect(np.array(outerlipspoint))
    rx, ry, rw, rh = rect
    center = (rx + (rw / 2), ry + (rh / 2))
    cx, cy = center
    # cv2.circle(frame, center, 5, (0, 255, 0), thickness=2)
    # return frame

    colormask = np.zeros_like(frame)
    mask = colormask.copy()
    mask = cv2.fillConvexPoly(mask, np.int32(outerlipspoint), (255,255,255))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    scale = 2.0
    sw = int(scale * width)
    sh = int(scale * height)
    scalemask = cv2.resize(mask, (sw, sh))
    scalefg = cv2.resize(frame, (sw, sh))
    scaledroix = int(cx * (scale - 1.0))
    scaledroiy = int(cy * (scale - 1.0))
    roiofmask = scalemask[scaledroiy: scaledroiy + height, scaledroix: scaledroix + width]
    roifg = scalefg[scaledroiy: scaledroiy + height, scaledroix: scaledroix + width]

    lip = AlphaBlending(roifg, frame, roiofmask)
    cv2.imshow("roimask", roiofmask)
    cv2.imshow("lip", lip)


    return frame

# 48~59 outer lip, 60~67 inner lip

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
            frame = handleFrameShape(frame, shape)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()