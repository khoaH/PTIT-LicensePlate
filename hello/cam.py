import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("D:\\PTIT\\HK6\\xulyanh\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("D:\\PTIT\\HK6\\xulyanh\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml")
while True:
    ret, frame = cap.read()

    print(ret)
    print(frame)
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    # cv2.imshow("frame", frame)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow("img", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()