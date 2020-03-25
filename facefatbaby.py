# coding:utf-8
from cv2 import cv2
import numpy as np
import zzimgtool
import imutils

def Babyface(img):
    faces = zzimgtool.facesLandmarks(img)

    points = faces[0]
    # # draw land marks
    # for x, y in points:
    #     cv2.circle(img, (x, y), 4, (0, 255, 0))
    hei, wei, _ = img.shape
    points = zzimgtool.Addpointsatcorners(points, wei, hei)

    # 移动特征点
    originalps = points.copy()
    mouthps = points[48:68]
    zzimgtool.Movepointstotarget(mouthps, points[27], 0.11)
    xx = points[49:54]
    zzimgtool.Movepointstotarget(xx, points[62], 0.1)
    yy = points[56:59]
    zzimgtool.Movepointstotarget(yy, None, -0.1)

    noseps = points[28:36]
    zzimgtool.Movepointstotarget(noseps, points[27], 0.1)
    points[33][1] = points[33][1] - 2

    chinp = points[0:17]
    zzimgtool.Movepointstotarget(chinp, points[27], 0.09, toX=False)
    zzimgtool.Movepointstoaverage(chinp)

    eyeleft = points[36:42]
    zzimgtool.Movepointstotarget(eyeleft, points[39], -0.1)
    zzimgtool.Movepointstotarget(eyeleft, None, -0.1, toX=False)
    eyeright = points[42:48]
    zzimgtool.Movepointstotarget(eyeright, points[42], -0.1)
    zzimgtool.Movepointstotarget(eyeright, None, -0.1, toX=False)

    res = zzimgtool.warpImage(img, originalps, points)

    # 调整亮度色相
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].astype(np.float)
    # v -= 10
    v += 32
    v *= 0.9
    v[v < 0] = 0
    v[v > 255] = 255
    hsv[:,:,2] = v
    h = hsv[:,:,0].astype(np.float)
    h -= 1
    h[h < 0] += 180
    hsv[:,:,0] = h
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 化妆减高光，滤色融合
    makeupmask = np.zeros_like(res)
    makeupcolor = (128, 128, 176)
    leftcheek = points[[5, 1, 36, 48]]
    cv2.fillConvexPoly(makeupmask, leftcheek, makeupcolor)
    rightcheek = points[[12, 16, 45, 54]]
    cv2.fillConvexPoly(makeupmask, rightcheek, makeupcolor)
    chins = points[[6, 10, 57]]
    cv2.fillConvexPoly(makeupmask, chins, makeupcolor)
    rigdtnose = points[[39, 27, 42, 33]]
    cv2.fillConvexPoly(makeupmask, rigdtnose, makeupcolor)
    philtrum = points[[33, 50, 52]]
    cv2.fillConvexPoly(makeupmask, philtrum, makeupcolor)
    # forehead
    forehead = points[[19, 25, 27]].copy()
    zzimgtool.Movepointstotarget(forehead, points[33], -0.5, toX=False)
    cv2.polylines(makeupmask, [forehead], True, makeupcolor, thickness=30)
    blursize = int(float(makeupmask.shape[0]) * 0.3)
    if blursize % 2 == 0:
        blursize += 1
    makeupmask = cv2.GaussianBlur(makeupmask, (blursize, blursize), 0)
    # cv2.imshow("makeup", makeupmask)

    # resultBase = res.copy()
    # screen mask
    screen = zzimgtool.ScreenBlend(res, makeupmask)
    screen = cv2.bilateralFilter(screen, 9, 90, 90)

    # 本来觉得能扣五官，其他部分做模糊，但如果特征点不准确则看起来很可怕
    # mask = np.zeros_like(res)
    # cv2.fillConvexPoly(mask, mouthps, (255,255,255))
    # cv2.fillConvexPoly(mask, eyeleft, (255,255,255))
    # cv2.fillConvexPoly(mask, eyeright, (255,255,255))
    # mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # res = zzimgtool.AlphaBlending(res, screen, mask)
    res = screen

    # res = USM(res)
    # blursize = 99
    # blr = 11
    # res = cv2.GaussianBlur(res, (blr, blr), 0)
    res = zzimgtool.USM(res, 11, val = 2)
    # facemask = np.zeros_like(res)
    # hulls = cv2.convexHull(points[0:27], returnPoints=True)
    # cv2.fillConvexPoly(facemask, hulls, (255,255,255))
    # facemask = cv2.GaussianBlur(facemask, (5, 5), 0)
    # facenew = zzimgtool.AlphaBlending(res, resultBase, facemask)
    return res

img = cv2.imread('face/testface6.png')
img = imutils.resize(img, height = 600)
res = Babyface(img)
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()