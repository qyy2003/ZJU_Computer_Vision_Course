import cv2
import os
import numpy as np
def mix(img1,img2,M):
    M=M[:2]
    height1, width1, channels = img1.shape
    height2, width2, channels = img2.shape
    min_width=0
    max_width=width2
    min_height=0
    max_height=height2
    for X in [[0,0,1],[width1,0,1],[0,height1,1],[width1,height1,1]]:
        Y=np.matmul(np.array(M),np.array([X]).T)
        # print(Y)
        min_width=min(Y[0],min_width)
        max_width=max(Y[0],max_width)
        min_height=min(Y[1],min_height)
        max_height=max(Y[1],max_height)
    dst=np.zeros((int(max_height-min_height),int(max_width-min_width),3),dtype=np.uint8)
    dst[int(-min_height):int(-min_height)+height2,int(-min_width):int(-min_width)+width2]=img2

    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    M[0][2]+=int(-min_width)
    M[1][2]+=int(-min_height)
    # dst2 = cv2.warpPerspective(img1, M, (int(max_width-min_width),int(max_height-min_height)))
    dst2 = cv2.warpAffine(img1, M, (int(max_width-min_width),int(max_height-min_height)))
    cv2.imshow("dst1", dst2)
    cv2.waitKey(0)
    dst_final=(dst==np.array([0,0,0])).astype(np.uint8)*dst2+dst
    cv2.imshow("new", dst_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst_final
    # print(M)
    # print(mask)
    # transform the image
    height1, width1, channels1 = img1.shape
    height2, width2, channels2 = img2.shape

    dst = cv2.warpAffine(img1, M[:2], (width1 + width2, height1 + height2))
    cv2.imshow("warped", dst)
    # Use homography
    # height, width, channels = img2.shape
    # img1Reg = cv2.warpPerspective(img1, M, (width, height))
    # cv2.imshow("warped",img1Reg)

    # switch the image using addWeighted
    # img2 = cv2.addWeighted(dst,0.5,img2,0.5,0)
    cv2.imshow("new", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.waitKey(0)
    return img2
def image_switch(img1,img2):
    # switch the imagen using opencv

    #detect 128 keypoints and compute descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    #create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # fitting the transformation matrix
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    kp11=[]
    kp22=[]
    for i in matches[:1280]:
        # print(i.distance)
        kp11.append(kp1[i.queryIdx].pt)
        kp22.append(kp2[i.trainIdx].pt)
    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1, kp11, img2, kp22, None)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None)
    #draw paied points on img3
    for i in range(10):
        cv2.circle(img3,(int(kp11[i][0]),int(kp11[i][1])),10,(0,0,255),-1)
        # cv2.circle(img3,(int(kp22[i][0]),int(kp22[i][1])),10,(0,0,255),-1)

    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("matches",img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #find the transformation matrix using RANSAC in 2d
    M, mask = cv2.findHomography(np.float32(kp11), np.float32(kp22), cv2.RANSAC,confidence=0.99)

    # To test what M means
    X=list(kp11[0])
    print(np.array(M[:2]),np.array([[X[0],X[1],1]]).T)
    Y=np.matmul(np.array(M),np.array([[X[0],X[1],1]]).T)
    print(Y,":",kp22[0])
    return mix(img1,img2,M)

if __name__ == '__main__':
    # Read image
    filename="yosemite-test"
    imgs=[]
    # iterate through all the files in folder in name order
    paths=os.listdir(filename)
    paths.sort()
    for file in paths:
        # Check whether file is in .jpg format or not
        if file.endswith(".jpg"):
            # read the image
            img_single = cv2.imread(os.path.join(filename, file))
            imgs.append(img_single)

    img=imgs.pop(0)
    while(len(imgs)>0):
        img0=imgs.pop(0)
        img=image_switch(img,img0)
        # cv2.imshow("now_image",img)
        # break