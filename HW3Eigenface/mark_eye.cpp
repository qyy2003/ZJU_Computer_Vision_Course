#include"mark_eye.h"

int now=0;
FILE* Files;


static void onMouse( int event, int x, int y, int, void* ){
    if( event != EVENT_LBUTTONDOWN )
        return;
    if(now==0){fprintf(Files,"{\n  \"centre_of_left_eye\": [\n    %d,\n    %d\n   ],\n",x,y);}
    else{
    fprintf(Files,"   \"centre_of_right_eye\": [\n    %d,\n    %d\n   ]\n}",x,y);}
    // circle(img,(447,63), 63, (0,0,255), -1)
    now+=1;
    // std::cout << x << " " << y << std::endl;
}
int mark_eye(std::string input_path/* = "att-face/s1"*/,std::string output_path /*= "ATT-eye-location/s0/"*/)
{
    // std::string image_path = "att-face/s1/1.pgm";
    // std::string output_path = "ATT-eye-location/s0/";
    fs::create_directories(output_path);
    for (const auto& dirEntry : recursive_directory_iterator(input_path)){
        printf("Processing File %s ...\n",dirEntry.path().c_str());
        std::string name=fs::path(dirEntry).stem();
        std::string output_name = output_path+name+".json";
        Files=fopen(output_name.c_str(),"w");

        now=0;
        Mat img = imread(dirEntry.path().string());
        // print(img);
        namedWindow( "Display frame",WINDOW_NORMAL);
        resizeWindow ("Display frame", 900, 1200);
        imshow("Display frame", img);
        setMouseCallback("Display frame",onMouse);
        waitKey(0); // Wait for a keystroke in the window
        // while(now<2){
        //     waitKey(0);
        // }
        destroyAllWindows();
        fclose(Files);
    }
    // Mat img = imread(image_path);
    // // print(img);
    // namedWindow( "Display frame",WINDOW_NORMAL);
    // resizeWindow ("Display frame", 900, 1200);
    // imshow("Display frame", img);
    // setMouseCallback("Display frame",onMouse);
    // int k = waitKey(0); // Wait for a keystroke in the window
    return 0;
}