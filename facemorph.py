# coding:utf-8
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
import sys
import json
import zzimgtool
import faceswap

def morphImgs(obj1, obj2, alpha):
    img1, points1 = obj1
    img2, points2 = obj2
    # 变化中的特征点
    points = []
    for i in range(0, len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        p = (int(x),int(y))
        points.append(p)

    # 绘制结果
    imgMorph1 = zzimgtool.warpImage(img1, points1, points)
    imgMorph2 = zzimgtool.warpImage(img2, points2, points)

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
    # animalname = 'lion'
    # img1 = cv2.imread('animal/{}.jpg'.format(animalname))
    img1 = cv2.imread('face/baby/baby6.png')
    img2 = cv2.imread('face/testface3.jpg')
    # img1 = faceswap.swapfaceimg(img1, img2)

    sizeWidth = 400
    sizeHeight = 400
    img1 = cv2.resize(img1, (sizeWidth, sizeHeight))
    img2 = cv2.resize(img2, (sizeWidth, sizeHeight))

    # 动物脸特征点由文件给出
    points1 = zzimgtool.facesLandmarks(img1)[0].tolist() # pointsFromImageJSON(img1, 'animal/{}.json'.format(animalname))
    # 人脸特征点由dlib给出
    points2 = zzimgtool.facesLandmarks(img2)[0].tolist()
    
    if len(points2) == 0:
        print('no face found in image 2')
        exit()
    testarr = ((img1, points1), (img2, points2))

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
    for _ in range(10):
        results.append(img2)
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
    
    for _ in range(10):
        results.append(img1)
    while(True):
        for img in results:
            cv2.imshow("Morphed Face", img)
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q"):
                exit()

if __name__ == '__main__' :
    testAnimalFaceMorph()