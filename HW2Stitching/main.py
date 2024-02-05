import cv2
import os
import numpy as np
filename=None
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
    for i in matches[:128]:
        # print(i.distance)
        kp11.append(kp1[i.queryIdx].pt)
        kp22.append(kp2[i.trainIdx].pt)

    # Draw first 128 matches.
    # img3 = cv2.drawMatches(img1, kp11, img2, kp22, None)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:128],None)
    #draw paied points on img3
    for i in range(10):
        cv2.circle(img3,(int(kp11[i][0]),int(kp11[i][1])),10,(0,0,255),-1)
        # cv2.circle(img3,(int(kp22[i][0]),int(kp22[i][1])),10,(0,0,255),-1)

    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("output/{}_matched_{}.jpg".format(filename, index),img3)

    # cv2.imshow("matches",img3)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #find the transformation matrix using RANSAC in 2d
    M, mask = cv2.findHomography(np.float32(kp11), np.float32(kp22), cv2.RANSAC,confidence=0.99)


    # width1,height1=img1.shape[:2]
    # to cancel black side
    img_mask=img1.sum(-1)!=0
    edges_x, edges_y = np.where(img_mask)
    width1 = edges_x.max() - edges_x.min() + 1
    height1 = edges_y.max() - edges_y.min() + 1

    min_width = 0
    max_width = img2.shape[0]
    min_height = 0
    max_height = img2.shape[1]

    for X in [[0,0,1],[height1,0,1],[0,width1,1],[height1,width1,1]]:
        Y=np.matmul(np.array(M),np.array([X]).T)
        min_width=min(Y[1],min_width)
        max_width=max(Y[1],max_width)
        min_height=min(Y[0],min_height)
        max_height=max(Y[0],max_height)
    min_width,min_height,max_width,max_height=int(min_width),int(min_height),int(max_width),int(max_height)
    T_x=np.array([[1,0,-min_height],[0,1,-min_width],[0,0,1]])

    #importent error
    # M[2]=[0,0,1]
    # tranRes=cv2.warpAffine(img1, np.matmul(T_x,M)[:2], (max_height-min_height, max_width-min_width))

    tranRes = cv2.warpPerspective(img1, np.matmul(T_x,M), (max_height-min_height, max_width-min_width))

    # to change which image is on the top
    # result=(tranRes==np.zeros([3])).astype(np.uint8)*img2+tranRes
    tranRes[-min_width:-min_width+img2.shape[0],-min_height:-min_height+img2.shape[1]]=img2

    cv2.imwrite("output/{}_result_{}.jpg".format(filename,index), tranRes)

    # cv2.imshow("result", tranRes)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return tranRes

if __name__ == '__main__':
    # Read image
    filename="yosemite-test"
    # filename="lab"
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
    index=0
    while(len(imgs)>0):
        index+=1
        img0=imgs.pop(0)
        img=image_switch(img,img0)
        # cv2.imshow("now_image",img)
        # break