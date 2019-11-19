from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import numpy as np
from cv2 import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_5_face_landmarks.dat')
# using 5 marks model

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
            bx, by, bw, bh = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (i, (x, y)) in enumerate(shape):
                cv2.circle(frame, (x, y), 1, (255, 255, 0), 2)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
