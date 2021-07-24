# Creating a document scanner project using openCV
import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import argmax, argmin


# # accesing webcam
imgWidth = 480
imgHeight = 620
# brightness = 100

# webcam = cv.VideoCapture(0)
# webcam.set(3,imgWidth)
# webcam.set(4,imgHeight)
# webcam.set(10,brightness)

# Path for manual input(file)
path = "/media/deepaksagar/Study Materials/Programming/python/OpenCV/resources/book2.jpg"
img=cv.imread(path);
img=cv.resize(img,(imgWidth,imgHeight))
imgContour = img.copy()

def preProcessing(img):
    grayedImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blurredImg = cv.GaussianBlur(grayedImg,(5,5),1)
    canniedImg = cv.Canny(blurredImg,200,80)
    kernel = np.ones((5,5))
    dilatedImg = cv.dilate(canniedImg,kernel,iterations=2)
    thresImg = cv.erode(dilatedImg,kernel,iterations=1)

    return canniedImg;
 
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area>5000:
            # Drawing the shapes that were detected
            # syntax, drawContours(imgSrc,presentIterationNo,-1=for all detected contours,color,thickness)
            # cv.drawContours(imgContour,cnt,-1,(255,0,0),2)

            # Calculation of perimeter
            # syntax, srcLength(presentContour,isItClosedContour)
            peri = cv.arcLength(cnt,True)

            # Aproximating the no of cornor point
            # syntax, approxPolyDP(presntContour,resolution,isClosed)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            if area>maxArea and len(approx)==4:
                biggest = approx
                maxArea = area
    cv.drawContours(imgContour,biggest,-1,(255,0,0),20)
    return biggest;

def reOrderPts(myPoints):
    myPoints = myPoints.reshape((4,2))
    myNewPoints = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    
    myNewPoints[0] = myPoints[argmin(add)]
    myNewPoints[3] = myPoints[argmax(add)]

    diff = np.diff(myPoints,axis=1);
    myNewPoints[1] = myPoints[argmin(diff)]
    myNewPoints[2] = myPoints[argmax(diff)]

    return myNewPoints

def getWarped(img,biggest):
    biggest = reOrderPts(biggest)
    # define the points for which we wanting to change the perspective
    pts1 = np.float32(biggest)
    # define the points about which we are going to change the view 
    pts2 = np.float32([[0,0],[imgWidth,0],[0,imgHeight],[imgWidth,imgHeight]])

    # generate the transformation matrix for warping it
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    print(matrix)

    # now warping the image using the transformation matrix and deine resolution
    imgOutput = cv.warpPerspective(img,matrix,(imgWidth,imgHeight))

    return imgOutput


while(1):
    # success, img = webcam.read()
    # img=cv.resize(img,(imgWidth,imgHeight))
    # imgContour = img.copy()

    thresImg = preProcessing(img)
    biggest = getContours(thresImg)
    print(biggest);
    warpedImg = getWarped(img,biggest)




    cv.imshow("Result",warpedImg)
    cv.imshow("original",img)


    if cv.waitKey(1) & 0xFF == ord('q'):
        break


