# coding:utf-8
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import numpy as np
from cv2 import cv2
import math
import zzimgtool

def handleFrameShape(frame, shape):
    zzimgtool.DrawLandmarks(frame, shape)
    # return frame
    outerlipspoint = shape[48:60]
    height, width = frame.shape[0:2]

    innerliptop = shape[62]
    innerlipbottom = shape[66]
    innerlipleft = shape[60]
    innerlipright = shape[64]
    innerlipheight = zzimgtool.DistancePoints(innerliptop, innerlipbottom)
    innerlipwidth = zzimgtool.DistancePoints(innerlipleft, innerlipright)
    # print('lip wid: ', innerlipwidth)
    # print('lip hei: ', innerlipheight)
    
    scale = 1.0
    radio = innerlipheight / innerlipwidth
    maxscale = 5.0
    # print('lip radio: ', radio)
    if radio > 0.05:
        scale += radio * 4
    if scale > maxscale:
        scale = maxscale
    # innerlips = [innerliptop, innerlipbottom, innerlipleft, innerlipright]
    # for (i, (x, y)) in enumerate(innerlips):
    #     cv2.circle(frame, (x, y), 1, (255, 255, 0), 2)
    # return frame

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

    # scale = 2.0
    sw = int(scale * width)
    sh = int(scale * height)
    scalemask = cv2.resize(mask, (sw, sh))
    scalefg = cv2.resize(frame, (sw, sh))
    scaledroix = int(cx * (scale - 1.0))
    scaledroiy = int(cy * (scale - 1.0))
    roiofmask = scalemask[scaledroiy: scaledroiy + height, scaledroix: scaledroix + width]
    roifg = scalefg[scaledroiy: scaledroiy + height, scaledroix: scaledroix + width]

    lip = zzimgtool.AlphaBlending(roifg, frame, roiofmask)
    # cv2.imshow("roimask", roiofmask)
    # cv2.imshow("lip", lip)

    return lip

# 48~59 outer lip, 60~67 inner lip

vs = VideoStream(src=0).start()
while True:
    originframe = vs.read()
    if originframe is None:
        break
    
    # originframe = imutils.resize(originframe, width = 600)
    # originframe = np.fliplr(originframe)
    originframe = cv2.imread('face/testface2.jpg')

    # progress
    frame = originframe.copy()
    faces = zzimgtool.facesLandmarks(frame)
    if len(faces) > 0:
        for shape in faces:
            # DrawLandmarks(frame, shape)
            frame = handleFrameShape(frame, shape)

        text = "{} face(s) found".format(len(faces))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()