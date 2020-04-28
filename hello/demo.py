import cv2
import numpy as np
import os
import imutils
import shutil
import random as rng

#Đọc Ảnh, file pre-train
img = cv2.imread('D:\\PTIT\\HK6\\xulyanh\\Bike_back\\2.jpg',1);
plate_cascade = cv2.CascadeClassifier("D:\\PTIT\\HK6\\xulyanh\\output\\cascade.xml")
#nhận diện biển trong img
plates = plate_cascade.detectMultiScale(img, 1.1, 3)
#Tạo ảnh trước khi cắt
cropped = img
#in vùng chứa biển số và cắt
for (x,y,w,h) in plates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
    cropped = img[y:y+h, x:x+w]
cv2.imshow('img', img)
cv2.imshow("crop",cropped)

#khởi tạo kích thước của kí tự trên biển số
digit_w =30
digit_h =60

#Load file train đọc số
model_svm =cv2.ml.SVM_load('D:\\PTIT\\HK6\\xulyanh\\svm.xml')

# xóa thư mục (reset) "number" để lưu số đã cắt
# shutil.rmtree('./number', ignore_errors=True)
# tạo thư mục number
# os.mkdir('number')



#img = cv2.imread("./img/xh5.jpg")
#cv2.imshow("Original image", img)
(himg,wimg,chanel)=cropped.shape
print(himg)
#tiền xử lí ảnh
# kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, 0]], np.float32)

# kernel=1/3*kernel
# cropped = cv2.filter2D(cropped, -1, kernel)
#cropped = cv2.fastNlMeansDenoisingColored(cropped)
grayImg = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(grayImg,(5, 5), 0)
# canny = cv2.Canny(blur, 100, 200)
# cv2.imshow("blur", blur)
# cv2.imshow("canny", canny)
# _, contourToPoLy, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_poly = [None]*len(contourToPoLy)
# boundRect = [None]*len(contourToPoLy)
# centers = [None]*len(contourToPoLy)
# radius = [None]*len(contourToPoLy)
# for i, c in enumerate(contourToPoLy):
#     contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#     boundRect[i] = cv2.boundingRect(contours_poly[i])
# drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
#binImg = cv2.fillPoly(canny, pts=contours_poly, color=(255, 255, 255), )
_, binImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# for i in range(len(contourToPoLy)):
#     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#     cv2.drawContours(drawing, contours_poly, i, color)
#     cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#         (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
# cv2.imshow("contour",drawing)
#ret, binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY_INV)
#binImg = canny
#binImg = cv2.adaptiveThreshold(grayImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
#binImg = cv2.fastNlMeansDenoising(binImg)
#tìm contour
cnts = cv2.findContours(binImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
#tạo ảnh tạm để giữ ảnh gốc k bị edit
imgtemp=cropped.copy()
cv2.drawContours(imgtemp,cnts,-1,(0,120,0),1)

#khởi tạo
plate_number=''
count=0
coorarr=[]


#duyệt từng cái contour
#for c in cnts:
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if h/w >1.5 and h/w <4 and cv2.contourArea(c)>100:
        #Đóng khung cho kí tự
        cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 0, 255),2)
        #crop thành những số riêng lẻ
        crop=img[y:y+h, x:x+w]
        #dùng để ghi vào thư mục number
        count+=1
        cv2.imwrite('./number/number%d.jpg'% count,crop)
        #lưu vào mảng tọa độ để tí xài
        coorarr.append((x,y))

        #tách số và predict
        #sao chép cái ảnh bin để khỏi hư cái kia :)) hơi dài nhưng an toàn
        binImgtemp=binImg
        #cắt ra từng số như cái crop ở trên nhưng t dùng biến khác để bây đỡ rối
        curr_num=binImgtemp[y:y+h, x:x+w]
        #xử lí để tí nữa đưa cái này vào hàm nó ràng buộc input phải kiểu dữ liệu như v
        #đầu tiên là resize lại cho nó cùng kích thước nhau cũng như= kích thước khi train
        curr_num=cv2.resize(curr_num,dsize=(digit_w,digit_h))
        _, curr_num=cv2.threshold(curr_num,30,255,cv2.THRESH_BINARY)
        #chuyển thành np để tí xài tạo thàn h mảng numpy
        curr_num= np.array(curr_num,dtype=np.float32)
        #reshape lại nha ae tự sắp xếp thành hàng ngang thì phải
        #ví dụ ảnh cao 2 ngang 10 thì giờ thành 1 hàng n20 px nha ae 
        curr_num=curr_num.reshape(-1,digit_w*digit_h)

        #train
        #chỗ này số 1 đằng sau t chưa hiểu ai hiểu chỉ t nha
        result=model_svm.predict(curr_num)[1]
        result= int(result[0,0])

        if result<=9: 
            result= str(result)
        else:
            result=chr(result)
        plate_number +=result+' '
        #này dùng viết lên màn hình thui ae
        cv2.putText(imgtemp,result,(x-50,y+50),cv2.FONT_HERSHEY_COMPLEX,3,(0, 255, 0), 2, cv2.LINE_AA)

#do t thêm dấu cách nên t cắt dấu cắt dư
stringarr=plate_number.strip()
#tạo thành 1 cái list trong python 
stringarr=stringarr.split(" ")

# print('stringarr chua sap xep',stringarr)
# print('coor chua sap xep', coorarr)


#sắp xếp lại các con số theo y
for i in range(len(coorarr)):
    #so sánh tọa độ y
    for j in range(i+1,len(coorarr)):
        # nếu y của i > y của j 
        if coorarr[i][1]- coorarr[j][1] >15:
            temp=stringarr[i]
            stringarr[i]=stringarr[j]
            stringarr[j]=temp
            tempp=coorarr[i]
            coorarr[i]=coorarr[j]
            coorarr[j]=tempp
        elif coorarr[i][0]- coorarr[j][0] >0:
            temp=stringarr[i]
            stringarr[i]=stringarr[j]
            stringarr[j]=temp
            tempp=coorarr[i]
            coorarr[i]=coorarr[j]
            coorarr[j]=tempp
            

# print('stringarr da sap xep',stringarr)
# print('coor da sap xep', coorarr)
#sau khi sắp xếp tao cho nó thành string lại nè
plate_number=''.join(stringarr)
print('bien so xe: ',plate_number)



cv2.imshow('binary',binImg)
cv2.imshow("gray", grayImg)
cv2.imshow('result',imgtemp)

#mở thư mục number để xe,
# os.startfile('number')

cv2.waitKey()
cv2.destroyAllWindows()

