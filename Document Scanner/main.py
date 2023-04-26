import cv2
import numpy as np
# frameWidth = 640
# frameHeigth = 480
widthImg, heightImg = 480,640
kernelSize = np.ones((5,5))
cap = cv2.VideoCapture(0)
cap.set(3,widthImg)  # for width dim (3 is width ID)
cap.set(4,heightImg) # for height dim (4 is height ID)
cap.set(10,150)  # for brightness control (10 is the brightness control ID)

def preProcessImag(img, kernel):
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    imgDilation = cv2.dilate(imgCanny,kernel, iterations=2)
    imgErode = cv2.erode(imgDilation,kernel, iterations=1)
    return imgErode

def getCountours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area>1500:
            # cv2.drawContours(imgContour, cnt, -1, (255,0,0),3)
            peri = cv2.arcLength(cnt,True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            # print(len(approx))
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def reorder (myPoints):
    myPoints = myPoints.reshape((4,2))  #as the biggest has shape (4,1,2) where 4 is the number of points 2 is the pair of x,y and 1 is redundant so we eliminate it by reshapping
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints,axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[3] = myPoints[np.argmax(diff)]
    return myPointsNew

def getWarp(img,biggest):
    biggest = reorder(biggest)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgOutput = cv2.warpPerspective(img,matrix,(widthImg,heightImg))
    return imgOutput



while True:
    success, img = cap.read()
    cv2.resize(img, (widthImg,heightImg))
    # cv2.imshow("video", img)
    imgContour = img.copy()
    imgThres = preProcessImag(img,kernelSize)
    # cv2.imshow("video", imgThres)
    biggest = getCountours(imgThres)
#     cv2.imshow("video", imgContour)
    if biggest.size != 0:
        imgWarped = getWarp(img,biggest)
        cv2.imshow("video", imgWarped)


    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
