# coding:utf-8
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
import sys
import json

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyTransform(src, srcTri, dstTri, size):
    '''
    # 为什么用透视变换会错？
    ps1 = srcTri[:]
    ps2 = dstTri[:]
    makeupPoint = (-1234, -8765)
    ps1.append(makeupPoint)
    ps2.append(makeupPoint)
    warpMat = cv2.getPerspectiveTransform(np.float32(ps1), np.float32(ps2))
    dst = cv2.warpPerspective(src, warpMat, (size[0], size[1]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return dst
    '''
    # 仿射变换
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morphTriangleROI(obj1, objM):
    img1, triangle0 = obj1
    imgRes, triangleM = objM

    # 找出围绕三角形的最小矩形
    r1 = cv2.boundingRect(np.array([triangle0]))
    rM = cv2.boundingRect(np.array([triangleM]))

    r1x, r1y, r1w, r1h = r1
    rMx, rMy, rMw, rMh = rM

    # 三角形三点于矩形左上角的偏移，用作ROI局部的三角形坐标
    offsets1 = []
    offsetsM = []

    for i in range(0, 3):
        offsetsM.append([(triangleM[i][0] - rMx),(triangleM[i][1] - rMy)])
        offsets1.append([(triangle0[i][0] - r1x),(triangle0[i][1] - r1y)])

    # 填充M的三角形遮罩
    mask = np.zeros((rMh, rMw, 3), dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(offsetsM), (1, 1, 1))

    # 取出ROI
    img1ROI = img1[r1y:r1y + r1h, r1x:r1x + r1w]

    size = (rMw, rMh)
    # 将ROI变换
    # warpROI = applyTransform(img1ROI, offsets1, offsetsM, size)
    # ROI仿射变换
    warpMat = cv2.getAffineTransform(np.float32(offsets1), np.float32(offsetsM))
    warpROI = cv2.warpAffine(img1ROI, warpMat, (rMw, rMh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 赋值到结果的ROI的#三角形mask#上！！
    imgResROI = imgRes[rMy:rMy+rMh, rMx:rMx+rMw] 
    imgResROI[:,:] = imgResROI * (1 - mask) + warpROI * mask

def morphImgs(obj1, obj2, alpha):
    img1, points1 = obj1
    img2, points2 = obj2
    # 变化中的特征点
    points = []
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((int(x),int(y)))

    boundheight, boundwidth = img1.shape[:2]
    bounds = (0, 0, boundwidth, boundheight)

    # 求三角分割，只能用其中一组点！！
    subdv = cv2.Subdiv2D(bounds)
    enumPoints = points
    for p in enumPoints:
        subdv.insert(p)
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

    # 空白结果
    imgMorph1 = np.zeros(img1.shape, dtype = img1.dtype)
    imgMorph2 = np.zeros(img2.shape, dtype = img2.dtype)

    # 根据三角形索引在取出3组点中的三角形点坐标
    for x, y, z in triangleindexes:
        triangle0 = [points1[x], points1[y], points1[z]]
        triangle2 = [points2[x], points2[y], points2[z]]
        triangleM = [points[x], points[y], points[z]]
        # Morph one triangle at a time.
        obj1 = (img1, triangle0)
        obj2 = (img2, triangle2)
        objM1 = (imgMorph1, triangleM)
        objM2 = (imgMorph2, triangleM)
        morphTriangleROI(obj1, objM1)
        morphTriangleROI(obj2, objM2)
    # 两张变化图加权
    imgMorph = cv2.addWeighted(imgMorph1, 1 - alpha, imgMorph2, alpha, 1)
    return imgMorph

def pointsFromImageJSON(img, jsonfile):
    f = open(jsonfile)
    string = f.read()
    f.close()
    dic = json.loads(string)
    d_points = dic['points']
    res_points = []

    height, width = img.shape[:2]
    height = float(height)
    width = float(width)
    # 为了不受图片缩放的影响，json记录了x，y的百分比，读取时乘以width，height
    for _, x, y in d_points:
        x = int(x * width)
        y = int(y * height)
        res_points.append((x, y))
    return res_points

def testAnimalFaceMorph():
    '''
    # 两个脸融合
    - 获取两个脸的特征点额外添加图片边角的点（确保点的数量和顺序是一致的）
    - 求得变化中的点（p = p1 * alpha + p2 * (1 - alpha))
    - 选取一组点做三角分割
    - 取得分割结果对于此组点的索引
    - 根据三角形三点索引取得三组点对应的三角形三点
    - 取出三角形的ROI变换
    - 变换结果赋值回三角形mask
    '''
    
    # 用人脸于动物脸融合
    animalname = 'lion'
    img1 = cv2.imread('animal/{}.jpg'.format(animalname))
    img2 = cv2.imread('face/testface.jpg')

    sizeWidth = 300
    sizeHeight = 400
    img1 = cv2.resize(img1, (sizeWidth, sizeHeight))
    img2 = cv2.resize(img2, (sizeWidth, sizeHeight))

    # 动物脸特征点由文件给出
    points1 = pointsFromImageJSON(img1, 'animal/{}.json'.format(animalname))
    points2 = []

    # 人脸特征点由dlib给出
    detector = dlib.get_frontal_face_detector()
    modelpath = '../shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(modelpath)
    testarr = [(img2, points2)]
    for img, pois in testarr:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        if len(rects) > 0:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            for p in shape:
                x, y = p
                pois.append((x, y))
    if len(points2) == 0:
        print('no face found in image 2')
        exit()
    testarr.append((img1, points1))

    # 给两组特征点添加图片边角
    for img, pois in testarr:
        height, width = img.shape[:2]
        cut = 1
        height -= cut
        width -= cut
        cut = 0
        # add corners
        pois.insert(0, (cut, cut))
        pois.insert(0, (width, height))
        pois.insert(0, (cut, height))
        pois.insert(0, (width, cut))
        # add edge center
        pois.insert(0, (cut, height / 2))
        pois.insert(0, (width, height / 2))
        pois.insert(0, (width / 2, cut))
        pois.insert(0, (width / 2, height))

    # alpha = 0.5
    # obj1 = (img1, points1)
    # obj2 = (img2, points2)
    # imgMorph = morphImgs(obj2, obj1, alpha)
    # cv2.imshow("Morphed Face", imgMorph)
    # key = cv2.waitKey(0) & 0xFF
    # exit()

    # 脸1渐变脸2
    results = []
    rang = range(0, 101, 2)
    for alpha in rang:
        alpha = float(alpha) / 100.0
        obj1 = (img1, points1)
        obj2 = (img2, points2)
        imgMorph = morphImgs(obj2, obj1, alpha)
        results.append(imgMorph)
        cv2.imshow("Morphed Face", imgMorph)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()
    waitTime = len(results) * 1000 / 30
    while(True):
        for img in results:
            cv2.imshow("Morphed Face", img)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                exit()

if __name__ == '__main__' :
    testAnimalFaceMorph()