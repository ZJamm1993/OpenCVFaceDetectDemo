# coding:utf-8
import numpy as np
import cv2
from imutils import face_utils
import imutils
import dlib

detector = dlib.get_frontal_face_detector()
modelpath = '../shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(modelpath)
print('inited face detector')


def DrawLandmarks(src, landmarks):
    boundheight, boundwidth = src.shape[:2]
    bounds = (0, 0, boundwidth, boundheight)
    for (i, (x, y)) in enumerate(landmarks):
        cv2.circle(src, (x, y), 1, (255, 255, 0), 2)
        cv2.putText(src, str(i), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
    return src

def DistancePoints(obj0, obj1):
    size = obj0.shape[0]
    tota = 0
    for i in range(size):
        tota += np.square(obj0[i] - obj1[i])
    res = np.sqrt(tota)
    return res

def AlphaBlending(fgImg, bgImg, alphaMask):
    fgImg = fgImg.astype(np.float)
    bgImg = bgImg.astype(np.float)
    alphaMask = alphaMask.astype(np.float) / 255.0
    fg = cv2.multiply(alphaMask, fgImg)
    bg = cv2.multiply(1.0 - alphaMask, bgImg)
    mix = cv2.add(fg, bg)
    mix = mix.astype(np.uint8)
    return mix

def facesLandmarks(src):
    faces = []
    # 人脸特征点由dlib给出
    gray = src
    if len(gray.shape) == 3:
        if gray.shape[2] == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        faces.append(shape)
    return faces

def delaunayTriangleIndexes(bounds, points):
    # 求三角分割，只能用其中一组点！！
    subdv = cv2.Subdiv2D(bounds)
    enumPoints = points
    for p in enumPoints:
        x, y = p
        subdv.insert((x, y))
    trianglepoints = subdv.getTriangleList()
    # 将三角形点坐标转化为三角形点索引
    triangleindexes = []
    for tri in trianglepoints:
        x1, y1, x2, y2, x3, y3 = tri
        loop3 = 0
        for i, (px, py) in enumerate(enumPoints):
            is1 = (x1 == px) and (y1 == py)
            is2 = (x2 == px) and (y2 == py)
            is3 = (x3 == px) and (y3 == py)
            if is1 or is2 or is3:
                triangleindexes.append(i)
                loop3 += 1
            if loop3 >= 3: # 当长宽太小时，会错误的加入更多的点，使得不能整除3
                break
    triangleindexes = np.array(triangleindexes).reshape(-1, 3)
    return triangleindexes

def warpTriangleROI(src, triangleSrc, dst, triangleDst):
    src, triangleSrc = (src, triangleSrc)
    dst, triangleDst = (dst, triangleDst)

    # 找出围绕三角形的最小矩形
    r1 = cv2.boundingRect(np.array([triangleSrc]))
    rM = cv2.boundingRect(np.array([triangleDst]))

    r1x, r1y, r1w, r1h = r1
    rMx, rMy, rMw, rMh = rM

    # 三角形三点于矩形左上角的偏移，用作ROI局部的三角形坐标
    offsets1 = []
    offsetsM = []

    for i in range(0, 3):
        offsetsM.append([(triangleDst[i][0] - rMx),(triangleDst[i][1] - rMy)])
        offsets1.append([(triangleSrc[i][0] - r1x),(triangleSrc[i][1] - r1y)])

    # 填充M的三角形遮罩
    mask = np.zeros((rMh, rMw, 3), dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(offsetsM), (1, 1, 1))

    # 取出ROI
    srcROI = src[r1y:r1y + r1h, r1x:r1x + r1w]

    size = (rMw, rMh)
    # 将ROI变换
    # warpROI = applyTransform(srcROI, offsets1, offsetsM, size)
    # ROI仿射变换
    warpMat = cv2.getAffineTransform(np.float32(offsets1), np.float32(offsetsM))
    warpROI = cv2.warpAffine(srcROI, warpMat, (rMw, rMh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 赋值到结果的ROI的#三角形mask#上！！
    dstROI = dst[rMy:rMy+rMh, rMx:rMx+rMw] 
    dstROI[:,:] = dstROI * (1 - mask) + warpROI * mask
    return dst

def warpImage(src, srcpoints, dstpoints):
    # 空白结果
    imgWarp = np.zeros(src.shape, dtype = src.dtype)

    boundheight, boundwidth = src.shape[:2]
    bounds = (0, 0, boundwidth, boundheight)

    triangleindexes = delaunayTriangleIndexes(bounds, srcpoints)
    # 根据三角形索引在取出3组点中的三角形点坐标
    for x, y, z in triangleindexes:
        trianglesrc = [srcpoints[x], srcpoints[y], srcpoints[z]]
        triangledst = [dstpoints[x], dstpoints[y], dstpoints[z]]
        # Morph one triangle at a time.
        warpTriangleROI(src, trianglesrc, imgWarp, triangledst)
    return imgWarp