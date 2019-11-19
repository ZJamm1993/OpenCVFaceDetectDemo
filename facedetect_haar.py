from cv2 import cv2
import numpy as numpy

face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('model/haarcascade_eye.xml')

def detectFacesEyes(srcimg):
    resimg = srcimg.copy()
    gray = cv2.cvtColor(resimg,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,2,5)
    eyesres = []
    for (x,y,w,h) in faces:
        resimg = cv2.rectangle(resimg,(x,y),(x+w,y+h),(0,255,0))
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = resimg[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255))
    return (resimg, faces, eyesres)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320,180))
    frame = cv2.flip(frame,1)
    resimg, faces, eyes = detectFacesEyes(frame)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame',resimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# img = cv2.imread('resource/lena.jpg')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray,1.3,5)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0))
#     roi_gray = img[y:y+h,x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255))

# cv2.imshow('img',img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()