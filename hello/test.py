import cv2
import urllib.request
import numpy as np

URL = 'http://192.169.1.102:8080/shot.jpg'

while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()),dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    cv2.imshow("img", img)
    if cv2.waitKey(1) == 27:
        break