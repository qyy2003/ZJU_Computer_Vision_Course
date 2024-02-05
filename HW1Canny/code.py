import copy
from queue import Queue
import cv2
import numpy as np
import math


def myCanny(img,threshold1,threshold2,sigma=5):
    assert threshold1>=0 and threshold2<=255
    #Gaussian Filter
    img = cv2.GaussianBlur(img, (sigma, sigma), cv2.BORDER_DEFAULT)
    img=np.array(img,dtype=int)
    #Calculate Gradient's Amplitude and Direction
    m,n=img.shape
    Gx=np.zeros([m-1,n-1])
    Gy=np.zeros([m-1,n-1])
    M=np.zeros([m-1,n-1])
    thetas=np.zeros([m-1,n-1])
    for i in range(m-1):
        for j in range(n-1):
            Gx[i][j]=(img[i,j+1]-img[i][j]+img[i+1][j+1]-img[i+1][j])/2
            Gy[i][j]=(img[i][j]-img[i+1][j]+img[i][j+1]-img[i+1][j+1])/2
            M[i][j]=(Gx[i][j]**2+Gy[i][j]**2)**0.5
            thetas[i][j]=math.atan2(Gy[i][j],Gx[i][j])
    M=M/np.max(M)*100
    def check(i,j):
        if(i<0 or i>=m-1):
            return -1
        if(j<0 or j>=n-1):
            return -1
        return M[i][j]
    thetas=thetas/math.pi*180
    N = copy.deepcopy(M)
    for i in range(m-1):
        for j in range(n-1):
            theta=thetas[i][j]
            if -157.5<theta<=-112.5 or 22.5<theta<=67.5:
                N[i][j] = (check(i - 1, j + 1) < M[i][j]) * N[i][j]
                N[i][j] = (check(i + 1, j - 1) < M[i][j]) * N[i][j]
            if -22.5<theta<=22.5 or theta<=-157.5 or theta >157.5:
                N[i][j] = (check(i , j -1) < M[i][j]) * N[i][j]
                N[i][j] = (check(i , j +1) < M[i][j]) * N[i][j]
            if -67.5<theta<=-22.5 or 112.5<theta<=157.5:
                N[i][j] = (check(i + 1, j +1) < M[i][j]) * N[i][j]
                N[i][j] = (check(i - 1, j -1) < M[i][j]) * N[i][j]
            if -112.5<theta<=-67.5 or 67.5<theta<=112.5:
                N[i][j] = (check(i+1 , j ) < M[i][j]) * N[i][j]
                N[i][j] = (check(i-1 , j ) < M[i][j]) * N[i][j]

    def checkN(i, j):
        if (i < 0 or i >= m - 1):
            return -1
        if (j < 0 or j >= n - 1):
            return -1
        return N[i][j]
    AN=np.zeros([m,n],np.uint8)
    q=Queue()
    for i in range(m-1):
        for j in range(n-1):
            if(N[i][j]>threshold2):
                q.put([i,j])
                AN[i][j]=1
    # return AN*255
    while(q.empty() is False):
        x,y=q.get()
        for i in [x-1,x,x+1]:
            for j in [y-1,y,y+1]:
                if(checkN(i,j)>threshold1):
                    if(AN[i][j]==0):
                        q.put([i,j])
                        AN[i][j]=1
    return AN


if __name__=="__main__":
    img = cv2.imread('me.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opencv_edge = cv2.Canny(gray, 100, 200)
    myedge = myCanny(gray, 10, 25)
    # myedge = myedge*img
    masked=copy.deepcopy(img)
    for i in range(3):
        masked[:,:,i]*=myedge
    # masked = cv2.bitwise_and(img, img,mask=myedge)
    cv2.imshow('Original Image', img)
    cv2.imshow('Opencv Edge', opencv_edge)
    cv2.imshow('My Edge', myedge*255)
    cv2.imshow('Masked Image', masked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
