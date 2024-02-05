#include"correct_with_eye.h"

int SIZE_l=96,SIZE_w=72,dx1=18,dx2=54,dy1=40,dy2=40;

Mat correct_with_eye1(Mat img,int x1,int y1,int x2,int y2)
{
    double dx,dy,dz;
    dx=x2-x1;dy=y2-y1;dz=sqrt(dx*dx+dy*dy);
    Mat middle_point=Mat::zeros(3,1,CV_64F);
    middle_point.at<double>(0,0)=(x1+x2)/2.0;
    middle_point.at<double>(1,0)=(y1+y2)/2.0;
    middle_point.at<double>(2,0)=1;

    Mat srcp=Mat::zeros(3,2,CV_64F);
    srcp.at<double>(0,0)=x1;srcp.at<double>(0,1)=x2;srcp.at<double>(2,0)=1;
    srcp.at<double>(1,0)=y1;srcp.at<double>(1,1)=y2;srcp.at<double>(2,1)=1;
    // std::cout << srcp << std::endl;

    Mat M=Mat::zeros(2,3,CV_64F);
    M.at<double>(0,0)=dx/dz;M.at<double>(0,1)=dy/dz;
    M.at<double>(1,0)=-dy/dz;M.at<double>(1,1)=dx/dz;  
    // std::cout << M << std::endl;

    Mat img_dst0=Mat::zeros(SIZE_l,SIZE_w,img.type());
    warpAffine(img,img_dst0,M,Size(SIZE_w,SIZE_l));
    Mat img_dst;
    cvtColor( img_dst0, img_dst0, COLOR_BGR2GRAY );
    // img_dst0.convertTo(img_dst1, CV_8U);
    // std::cout<<"!"<<img_dst1.type()<<std::endl;
    equalizeHist( img_dst0, img_dst );
    // namedWindow( "Display frame",WINDOW_NORMAL);
    // resizeWindow ("Display frame", 900, 1200);
    // imshow("Display frame", img);
    // waitKey(0); 
    // imshow("Display frame", img_dst);
    // waitKey(0); 
    // img_dst=img_dst.reshape(1,1);
    // std::cout << img_dst.size() << std::endl;
    return img_dst;
}
Mat correct_with_eye2(Mat img,int x1,int y1,int x2,int y2)
{
    double dx,dy,dz;
    dx=x2-x1;dy=y2-y1;dz=sqrt(dx*dx+dy*dy);
    Mat middle_point=Mat::zeros(3,1,CV_64F);
    middle_point.at<double>(0,0)=(x1+x2)/2.0;
    middle_point.at<double>(1,0)=(y1+y2)/2.0;
    middle_point.at<double>(2,0)=1;

    Mat srcp=Mat::zeros(3,2,CV_64F);
    srcp.at<double>(0,0)=x1;srcp.at<double>(0,1)=x2;srcp.at<double>(2,0)=1;
    srcp.at<double>(1,0)=y1;srcp.at<double>(1,1)=y2;srcp.at<double>(2,1)=1;
    // std::cout << srcp << std::endl;

    Mat M=Mat::zeros(2,3,CV_64F);
    M.at<double>(0,0)=dx/dz;M.at<double>(0,1)=dy/dz;
    M.at<double>(1,0)=-dy/dz;M.at<double>(1,1)=dx/dz;  
    // std::cout << M << std::endl;

    Mat new_middle=M*middle_point;
    // M.at<double>(0,2)=(dx1+dx2)/2.0-new_middle.at<double>(0,0);
    // M.at<double>(1,2)=(dy1+dy2)/2.0-new_middle.at<double>(1,0);
    M.at<double>(0,2)=(dx1+dx2)/2-new_middle.at<double>(0,0);
    M.at<double>(1,2)=(dy1+dy2)/2-new_middle.at<double>(1,0);

    Mat img_dst0=Mat::zeros(SIZE_l,SIZE_w,img.type());
    warpAffine(img,img_dst0,M,Size(SIZE_w,SIZE_l));
    Mat img_dst;
    cvtColor( img_dst0, img_dst0, COLOR_BGR2GRAY );
    // img_dst0.convertTo(img_dst1, CV_8U);
    // std::cout<<"!"<<img_dst1.type()<<std::endl;
    equalizeHist( img_dst0, img_dst );
    // namedWindow( "Display frame",WINDOW_NORMAL);
    // resizeWindow ("Display frame", 900, 1200);
    // imshow("Display frame", img);
    // waitKey(0); 
    // imshow("Display frame", img_dst);
    // waitKey(0); 
    // img_dst=img_dst.reshape(1,1);
    // std::cout << img_dst.size() << std::endl;
    return img_dst;
}
Mat correct_with_eye3(Mat img,int x1,int y1,int x2,int y2)
{
    double dx,dy,dz;
    dx=x2-x1;dy=y2-y1;dz=sqrt(dx*dx+dy*dy);
    Mat middle_point=Mat::zeros(3,1,CV_64F);
    middle_point.at<double>(0,0)=(x1+x2)/2.0;
    middle_point.at<double>(1,0)=(y1+y2)/2.0;
    middle_point.at<double>(2,0)=1;

    Mat srcp=Mat::zeros(3,2,CV_64F);
    srcp.at<double>(0,0)=x1;srcp.at<double>(0,1)=x2;srcp.at<double>(2,0)=1;
    srcp.at<double>(1,0)=y1;srcp.at<double>(1,1)=y2;srcp.at<double>(2,1)=1;
    // std::cout << srcp << std::endl;

    Mat M=Mat::zeros(2,3,CV_64F);
    M.at<double>(0,0)=dx/dz;M.at<double>(0,1)=dy/dz;
    M.at<double>(1,0)=-dy/dz;M.at<double>(1,1)=dx/dz;  
    // std::cout << M << std::endl;

    Mat new_middle=M*middle_point;
    // M.at<double>(0,2)=(dx1+dx2)/2.0-new_middle.at<double>(0,0);
    // M.at<double>(1,2)=(dy1+dy2)/2.0-new_middle.at<double>(1,0);
    M.at<double>(0,2)=-new_middle.at<double>(0,0);
    M.at<double>(1,2)=-new_middle.at<double>(1,0);
    M*=sqrt(1.0f*(dx1-dx2)*(dx1-dx2)+1.0f*(dy1-dy2)*(dy1-dy2))/dz;
    M.at<double>(0,2)+=(dx1+dx2)/2;
    M.at<double>(1,2)+=(dy1+dy2)/2;
    // std::cout << M << std::endl;
    // Mat final_middle1=M*middle_point;
    // std::cout << final_middle1 << std::endl;
    // Mat final_middle=M*srcp;
    // std::cout << final_middle << std::endl;
    // Mat M=Mat::zeros(2,3,CV_64F);
    // M.at<double>(0,0)=M0.at<double>(0,0);M.at<double>(0,1)=M0.at<double>(0,1);
    // M.at<double>(1,0)=M0.at<double>(1,0);M.at<double>(1,1)=M0.at<double>(1,1);  
    Mat img_dst0=Mat::zeros(SIZE_l,SIZE_w,img.type());
    warpAffine(img,img_dst0,M,Size(SIZE_w,SIZE_l));
    Mat img_dst;
    cvtColor( img_dst0, img_dst0, COLOR_BGR2GRAY );
    // img_dst0.convertTo(img_dst1, CV_8U);
    // std::cout<<"!"<<img_dst1.type()<<std::endl;
    equalizeHist( img_dst0, img_dst );
    // namedWindow( "Display frame",WINDOW_NORMAL);
    // resizeWindow ("Display frame", 900, 1200);
    // imshow("Display frame", img);
    // waitKey(0); 
    // imshow("Display frame", img_dst);
    // waitKey(0); 
    // img_dst=img_dst.reshape(1,1);
    // std::cout << img_dst.size() << std::endl;
    return img_dst;
}
Mat correct_with_eye(Mat img,int x1,int y1,int x2,int y2)
{
    Mat img1=correct_with_eye1(img,x1,y1,x2,y2);
    Mat img2=correct_with_eye2(img,x1,y1,x2,y2);
    Mat img3=correct_with_eye3(img,x1,y1,x2,y2);
    resize(img,img,Size(SIZE_w,SIZE_l));
    cvtColor( img, img, COLOR_BGR2GRAY );
    Mat imgs[]={img,img1,img2,img3};
    Mat img_show;
    hconcat(imgs,4,img_show);
    namedWindow( "Display frame",WINDOW_NORMAL);
    resizeWindow ("Display frame", 72*4*5,96*5);
    imshow("Display frame", img_show);
    imwrite("correct_with_eye_show.jpg",img_show);
    waitKey(0); 
    return img3;
}

Mat correct_with_eye_test(Mat img,int x1,int y1,int x2,int y2){
    Point2f srcTri[3];
    srcTri[0] = Point2f( x1, y1 );
    srcTri[1] = Point2f( (x1+x2)/2.0 ,(y1+y2)/2.0 );
    srcTri[2] = Point2f( x2, y2 );
    Point2f dstTri[3];
    dstTri[0] = Point2f( dx1,dy1 );
    dstTri[1] = Point2f( (dx1+dx2)/2.0 ,(dy1+dy2)/2.0 );
    dstTri[2] = Point2f( dx2,dy2 );
    Mat M = getAffineTransform( srcTri, dstTri );
    Mat img_dst=Mat::zeros(SIZE_l,SIZE_w,img.type());
    std::cout << M<< std::endl;
    warpAffine(img,img_dst,M,Size(SIZE_w,SIZE_l));
    namedWindow( "Display frame",WINDOW_NORMAL);
    resizeWindow ("Display frame", 900, 1200);
    imshow("Display frame", img);
    waitKey(0); 
    imshow("Display frame", img_dst);
    waitKey(0); 
    return img_dst;
}