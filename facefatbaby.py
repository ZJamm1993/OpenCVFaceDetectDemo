from cv2 import cv2
import numpy as np
import zzimgtool
import math

def distance2points(p1, p2):
    d0 = p1[0] - p2[0]
    d1 = p1[1] - p2[1]
    ad = d0 * d0 + d1 * d1
    res = math.sqrt(ad)
    return res

def movepointstotarget(pois, target, rate, toX = True, toY = True, keepRatio = False):
    tos = [toX, toY]
    if keepRatio:
        count = pois.shape[0]
        avgdis = [0, 0]
        for p in pois:
            for i in range(2):
                d = target[i] - p[i]
                avgdis[i] += d
        for i in range(2):
            avgdis[i] = int(float(avgdis[i]) / count)

    for p in pois:
        for i in range(2):
            toi = tos[i]
            if toi == True:
                dist = float(avgdis[i]) if keepRatio else float(target[i] - p[i])
                di = dist * rate
                p[i] = int(float(p[i]) + di)

def movepointstocircle(pois, target):
    count = pois.shape[0]
    first = 0
    last = count - 1
    p0 = pois[first]
    p1 = pois[last]

    dis0 = distance2points(p0, target)
    dis1 = distance2points(p1, target)
    print(dis0)

    firstfloat = float(first)
    lastfloat = float(last)
    for i in range(count):
        if i == first or i == last:
            continue
        thisp = pois[i]
        thisdis = distance2points(thisp, target)
        ifloat = float(i)
        supposeddis = dis0 * ((lastfloat - ifloat) / lastfloat) + dis1 * (ifloat / lastfloat)
        print(supposeddis)
        for xy in range(2):
            dxy = float(target[xy] - thisp[xy])
            supposeddxy = dxy / thisdis * supposeddis
            thisp[xy] = target[xy] - supposeddxy
    print(dis1)

def addpointsatcorners(pois, width, height):
    pois = pois.tolist()
    cut = 1
    height -= cut
    width -= cut
    cut = 0
    # add corners
    pois.append((cut, cut))
    pois.append((width, height))
    pois.append((cut, height))
    pois.append((width, cut))
    # add edge center
    pois.append((cut, height / 2))
    pois.append((width, height / 2))
    pois.append((width / 2, cut))
    pois.append((width / 2, height))
    return np.array(pois)

img = cv2.imread('face/F/img_girl_1.png')
faces = zzimgtool.facesLandmarks(img)

points = faces[0]
hei, wei, _ = img.shape
points = addpointsatcorners(points, wei, hei)
originalps = points.copy()
# eyetargetp = points[28]
# eyesps = points[36:48]

mouthps = points[48:68]
movepointstotarget(points[48:68], points[27], 0.15)

noseps = points[28:36]
movepointstotarget(noseps, points[27], 0.2)
movepointstotarget(noseps, points[27], 0.2, toY=False)

# chinpleft = points[0:9]
# movepointstocircle(chinpleft, points[28])
chinpright = points[2:14]
movepointstocircle(chinpright, points[30])

res = zzimgtool.warpImage(img, originalps, points)

res = cv2.GaussianBlur(res, (3, 3), 0)

cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()