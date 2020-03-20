print "hello world"

from cv2 import cv2
import numpy as np 
import os
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

usingCaffe = False
if usingCaffe:
    net = cv2.dnn.readNetFromCaffe('model/gender/deploy_gender.prototxt', 'model/gender/gender_net.caffemodel')
else :
    gender_classifier = load_model('model/gender/gender_mini_XCEPTION.2878-0.93.hdf5', compile=False)
    gender_target_size = gender_classifier.input_shape[1:3]

filepaths = []

def appendimagepathfromdir(res, dir):
    filenames = os.listdir(dir)
    for nam in filenames:
        if nam.endswith('png'):
            path = os.path.join(dir, nam)
            if os.path.isfile(path):
                res.append(path)

appendimagepathfromdir(filepaths, dir = "face/M")
appendimagepathfromdir(filepaths, dir = "face/F")

imagecount = len(filepaths)
numcols = 8
numrows = imagecount / numcols
if imagecount % numcols > 0:
    numrows += 1
for i in range(len(filepaths)):
    # test gender with face 
    img = cv2.imread(filepaths[i])
    # faceroi

    if usingCaffe:
    # using caffe model
        faceroi = cv2.resize(img, (227, 227))
        blobface = cv2.dnn.blobFromImage(faceroi) 
        net.setInput(blobface)
        detections = net.forward()
        detections = detections.reshape(-1)
        isGen = detections[0] > detections[1]
    else :
    # using keras
        faceroi = cv2.resize(img, gender_target_size)
        # faceroi = cv2.cvtColor(faceroi, cv2.COLOR_BGR2GRAY)
        testface = (faceroi - 127.5)*0.0078125
        gender_facecrops = np.array([testface]).astype(np.float32)
        gender_pred = gender_classifier.predict(gender_facecrops)
        detections = gender_pred[0]
        isGen = detections[0] < detections[1]

    genstr = "M" if isGen else "F"
    winname = genstr
    plt.subplot(numrows, numcols, i + 1)
    faceroi = cv2.cvtColor(faceroi, cv2.COLOR_BGR2RGB)
    plt.imshow(faceroi)
    plt.title(winname)
    plt.xticks([])
    plt.yticks([])
plt.show()
    
    
