print "hello world"

from cv2 import cv2
import numpy as np 

net = cv2.dnn.readNetFromCaffe('model/deploy.prototxt.txt', 'model/res10_300x300_ssd_iter_140000.caffemodel')

def detectimage(src, threshold = 0.9) :
    img = src.copy()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img)
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] 
        if confidence > threshold:
            box = detections[0, 0, i, 3:7].copy()
            box[0] *= w
            box[1] *= h
            box[2] *= w
            box[3] *= h # the box is scale value for w and h !??
            box = np.int_(box)
            (startX, startY, endX, endY) = box
            # draw the bounding box of the face along with the associated
            # probability
            text = str(confidence)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return (img, detections)

# img = cv2.imread('face_test.jpg')
# res, detections = detectimage(img, 0.7)
# print detections
# cv2.imshow("Output", res)

# using video

cap = cv2.VideoCapture(0)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.resize(frame,(320,180))
    res, detections = detectimage(frame, 0.7)

    # Display the resulting frame
    cv2.imshow('frame',res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
