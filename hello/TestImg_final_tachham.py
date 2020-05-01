import cv2
import numpy as np
import imutils
import os
import shutil



#khởi tạo kích thước của kí tự trên biển số
digit_w =30
digit_h =60

def Pretreatment(imgLP):

    #tiền xử lí ảnh
    grayImg = cv2.cvtColor(imgLP, cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(grayImg,9,75,75)
    # equal_histogram = cv2.equalizeHist(noise_removal)
    ret, binImg = cv2.threshold(grayImg, 100, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    binImg = cv2.morphologyEx(binImg,cv2.MORPH_DILATE,kerel3)
    return binImg

def contours_detect(binImg):
    #tìm contour
    #cnts, _ = cv2.findContours(binImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #tạo ảnh tạm để giữ ảnh gốc k bị edit
    return cnts
def draw_rects_on_img(img, cnts):
    imgtemp=img.copy()
    cv2.drawContours(imgtemp,cnts,-1,(0,120,0),1)
    return imgtemp

    #cv2.imshow('Number on License plate ',imgtemp)
    # print (cnts);

#khởi tạo 
plate_number=''
coorarr=[]

def find_number(cnts,binImg,imgtemp):
    count=0
    global plate_number
    global coorarr
    model_svm =cv2.ml.SVM_load('svm.xml')
    #duyệt từng cái contour
    # folder = './number/'
    # for filename in os.listdir(folder):
    #     file_path = os.path.join(folder, filename)
    #     try:
    #         if os.path.isfile(file_path) or os.path.islink(file_path):
    #             os.unlink(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print('Failed to delete %s. Reason: %s' % (file_path, e))
    plate_number = ''
    for c in (cnts):
        x,y,w,h=cv2.boundingRect(c)
        cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 255, 0), 1)
        if h/w >1.5 and h/w <4 and cv2.contourArea(c)>4500:#and h > himg*0.5:
            #Đóng khung cho kí tự
            cv2.rectangle(imgtemp, (x, y), (x + w, y + h), (0, 0, 255),2)
            #crop thành những số riêng lẻ
            crop=imgtemp[y:y+h, x:x+w]
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
            result=model_svm.predict(curr_num)[-1]
            result= int(result[0,0])

            if result<=9: 
                result= str(result)
            else:
                result=chr(result)
            plate_number +=result+' '
            #này dùng viết lên màn hình thui ae
            cv2.putText(imgtemp,result,(x-50,y+50),cv2.FONT_HERSHEY_COMPLEX,3,(0, 255, 0), 2, cv2.LINE_AA)
    return imgtemp

def sortNumber():
    global plate_number
    global coorarr
    #do t thêm dấu cách nên t cắt dấu cắt dư
    stringarr=plate_number.strip()
    #tạo thành 1 cái list trong python 
    stringarr=stringarr.split(" ")
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
            
    #sau khi sắp xếp tao cho nó thành string lại nè
    plate_number=''.join(stringarr)
    return plate_number

def detect(img):
    (himg,wimg,chanel)=img.shape
    if(wimg/himg >2):
        img=cv2.resize(img,dsize=(1000,200))
    else:
        img=cv2.resize(img,dsize=(800,500))

    binImg=Pretreatment(img)
    cnts=contours_detect(binImg)
    imgtemp=draw_rects_on_img(img,cnts)
    imgtemp2=find_number(cnts,binImg,imgtemp)
    sort_number = sortNumber();
    print('bien so xe: ',sort_number)
    plate_number=''
    coorarr.clear()
    return sort_number

def findLP_img(OriImg): #find License plate
    # xóa thư mục (reset) "number" để lưu số đã cắt
    # shutil.rmtree('./number', ignore_errors=True)
    # tạo thư mục number
    # os.mkdir('number')

    #Đọc file pre-train
    plate_cascade = cv2.CascadeClassifier("./cascade2.xml")
    #nhận diện biển trong img
    plates = plate_cascade.detectMultiScale(OriImg, 1.1, 3)
    #Tạo ảnh trước khi cắt
    img = OriImg
    #in vùng chứa biển số và cắt
    for (x,y,w,h) in plates:
        cv2.rectangle(OriImg,(x,y),(x+w,y+h),(255,0,0),1)
        img = OriImg[y:y+h, x:x+w]
        plate_num = detect(img)
        cv2.putText(OriImg, plate_num, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
    cv2.imshow("Original image", OriImg)
    # cv2.imshow("crop",img)
    # return img

def video_cap():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        print(ret)
        # print(frame)
        # frame = cv2.flip(frame, 1)
        img = findLP_img(frame);
        # cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":

    OriImg = cv2.imread('./Bike_back/5.jpg',1);
    video_cap()
    #Tìm biển số
    # img=findLP_img(OriImg);
    # #resize lại hình
    # (himg,wimg,chanel)=img.shape
    # if(wimg/himg >2):
    #     img=cv2.resize(img,dsize=(1000,200))
    # else:
    #     img=cv2.resize(img,dsize=(800,500))
    # cv2.imshow('Image',img)

    # binImg=Pretreatment(img)
    # cnts=contours_detect(binImg)
    # imgtemp=draw_rects_on_img(img,cnts)
    # imgtemp=find_number(cnts,binImg,imgtemp)
    # sort_number = sortNumber();

    # cv2.imshow('binary',binImg)
    # cv2.imshow('result',imgtemp)
    # print('bien so xe: ',sort_number)
    #mở thư mục number để xe,
    # os.startfile('number')
    # cv2.waitKey()
    # cv2.destroyAllWindows()