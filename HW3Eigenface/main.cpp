#include "mark_eye.h"
#include "correct_with_eye.h"

int new_shape[] ={1,1};
void prepare_one(std::string image_path,std::string json_path,std::string output_path){
    // std::string image_path = "att-face/s1/1.pgm";
    std::cout<<image_path<<" "<<json_path<<" "<<output_path<<std::endl;
    Mat img=imread(image_path);
    // std::string json_path = "ATT-eye-location/s1/1.json";
    FILE* Files;
    Files=fopen(json_path.c_str(),"r");
    int x[4],idx=0;
    char ch;
    x[0]=x[1]=x[2]=x[3]=0;
    while(!feof(Files)){
        fread(&ch,1,1,Files);
        if(ch<'0'||ch>'9'){
            if(x[idx]!=0){
                idx+=1;
            }
            continue;
        }
        x[idx]=x[idx]*10+ch-'0';
    }
    // std::cout<<x[0]<<" "<<x[1]<<" "<<x[2]<<" "<<x[3]<<std::endl;
    Mat img_out;
    img_out=correct_with_eye(img,x[0],x[1],x[2],x[3]);
    imwrite(output_path,img_out);
}
void prepare_all(std::string image_path="att-face",std::string json_path="ATT-eye-location",std::string output_path="processed_image"){
    for (const auto& dirEntry : recursive_directory_iterator(image_path)){
        if(fs::is_directory(dirEntry)){
            std::string middle=fs::path(dirEntry).relative_path().string().substr(image_path.length()+1);
            fs::create_directories(output_path+"/"+middle);
            continue;
        }
        printf("Processing File %s ...\n",dirEntry.path().c_str());
        std::string name=fs::path(dirEntry).stem();
        std::string middle=fs::path(dirEntry).relative_path().remove_filename().string().substr(image_path.length()+1);
        // std::string json_name = output_path+name+".json";
        // std::cout<<fs::path(dirEntry).relative_path().remove_filename().string().substr(image_path.length()+1)<<" "<<output_name<<std::endl;
        prepare_one(dirEntry.path().string(),json_path+"/"+middle+name+".json",output_path+"/"+middle+name+".pgm");
        break;
    }
    // correct_with_eye(image_path,json_path);
}
void load_dataset(float split,std::string path,std::vector<std::pair<Mat,int>> &train,std::vector<std::pair<Mat,int>> &test){
    int num=int(split*10);
    int a[]={1,2,3,4,5,6,7,8,9,10};
    // std::vector<pair<Mat,int>> train,test;
    for(int i=0;i<=41;i++){
        std::random_shuffle(a,a+10);
        for(int j=0;j<num;j++){
            int x=a[j];
            std::string name=path+"/s"+std::to_string(i)+"/"+std::to_string(x)+".pgm";
            // std::cout<<name<<std::endl;
            Mat img=imread(name,cv::IMREAD_GRAYSCALE);
            // new_shape=img.size();
            new_shape[0]=img.rows;
            new_shape[1]=img.cols;
            // std::cout<<SIZE_X<<" "<<SIZE_Y<<std::endl;
            img=img.reshape(1,1);
            img.convertTo(img, CV_64F);
            train.push_back(std::make_pair(img,i));
        }
        for(int j=num;j<10;j++){
            int x=a[j];
            std::string name=path+"/s"+std::to_string(i)+"/"+std::to_string(x)+".pgm";
            // std::cout<<name<<std::endl;
            Mat img=imread(name,cv::IMREAD_GRAYSCALE).reshape(1,1);
            img.convertTo(img, CV_64F);
            test.push_back(std::make_pair(img,i));
        }
    }
    // return train,test;
}
Mat vec2img(Mat v){
    // std::cout<<v.size()<<std::endl;
    // std::cout<<SIZE_X<<" "<<SIZE_Y<<std::endl;
    // vector<int> new_shape={SIZE_X,SIZE_Y};
    v=v.reshape(1,2,new_shape);
    v.convertTo(v, CV_8U);
    // imshow("Display frame", v);
    // waitKey(0);
    return v;
}

void eigenface_test(std::string img_path,std::string model_path,std::string output_path=NULL){
    Mat img=imread(img_path,cv::IMREAD_GRAYSCALE);
    img=img.reshape(1,1);
    img.convertTo(img, CV_64F);
    cv::FileStorage file(model_path, cv::FileStorage::READ);
    Mat mean,u,data,X;
    std::vector<int> labels;
    file["mean"] >> mean;
    file["u"] >> u;
    file["labels"] >> labels;
    file["data"] >> data;
    file["X"] >> X;

    Mat img_show=vec2img(img);
    img=img-mean;
    Mat tmp=u*img.t();
    float last_sum=-1,sum=0;
    int last_idx=0;
    // Mat new_img=Mat::zeros(1,img.cols,CV_64F);
    for(int i=0;i<data.cols;i++){
        sum=0;
        for(int j=0;j<data.rows;j++){
            sum+=(data.at<double>(j,i)-tmp.at<double>(j,0))*(data.at<double>(j,i)-tmp.at<double>(j,0));
        }
        if(last_sum==-1||sum<last_sum){
            last_sum=sum;
            last_idx=i;
        }
    }
    Mat nearset=X(Range(last_idx,last_idx+1),Range(0,X.cols))+mean;
    std::cout<<img.size()<<" "<<nearset.size()<<std::endl;
    // return;
    hconcat(img_show,vec2img(nearset),img_show);
    cvtColor(img_show,img_show,COLOR_GRAY2BGR);
    putText(img_show, //target image
            img_path.substr(16, 5), //text
            cv::Point(5, img_show.rows-5), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(118, 185, 0), //font color
            2);
    putText(img_show, //target image
            "s"+std::to_string(labels[last_idx]), //text
            cv::Point(new_shape[1]+5, img_show.rows-5), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(255, 0, 0), //font color
            2);
    imwrite(output_path,img_show);
}
void eigenface_reconstruct(std::string img_path,std::string model_path,std::string output_path="test.jpg"){
    Mat img=imread(img_path,cv::IMREAD_GRAYSCALE);
    img=img.reshape(1,1);
    img.convertTo(img, CV_64F);
    cv::FileStorage file(model_path, cv::FileStorage::READ);
    Mat mean,u;
    std::vector<int> labels;
    file["mean"] >> mean;
    file["u"] >> u;
    file["labels"] >> labels;
    file.release();

    Mat img_show=vec2img(img);
    img=img-mean;
    Mat tmp=u*img.t();
    // Mat new_img=Mat::zeros(1,img.cols,CV_64F);
    for(int i=0;i<tmp.rows;i++){
        if(i==10||i==25||i==50||i==100){
            // vec2img(mean);
            hconcat(img_show,vec2img(mean),img_show);
        }
        mean+=tmp.at<double>(i,0)*u.row(i);
    }
    imwrite(output_path,img_show);
    // std::cout<<tmp.size()<<std::endl;
    // std::cout<<tmp<<std::endl;
}
void eigenface_train(float energy,std::string output_path,std::vector<std::pair<Mat,int>> &train){
    float n=train.size();
    int PCA_num=int((energy/100*n));
    int m=train[0].first.cols;
    std::cout<<n<<" "<<m<<std::endl;

    Mat mean=Mat::zeros(1,m,CV_64F);
    for(auto i : train){
        mean+=i.first;
    }
    mean=mean/n;
    Mat X;
    std::vector<int> labels;
    for(auto i : train){
        if(X.empty()){
            X=i.first-mean;
        }else{
            vconcat(X,i.first-mean,X);
        }
        labels.push_back(i.second);
    }
    Mat S=X*X.t();
    Mat  w, u, vt;
    SVD::compute(S, w, u, vt);
    // std::cout<<w<<std::endl;
    // std::cout<<u*u.t()<<std::endl;
    u=X.t()*u;

    u=u.t();
    int n0=u.rows,m0=u.cols;
    double sum=0,max=0;
    std::cout<<n0<<" "<<m0<<std::endl;
    u=u(Range(0,PCA_num),Range(0,m0));
    n0=u.rows,m0=u.cols;
    std::cout<<n0<<" "<<m0<<std::endl;
    Mat eigenface_show=vec2img(mean);
    for(int i=0;i<n0;i++){
        // vec2img(u.row(i)/378*255);
        sum=0;
        max=0;
        for(int j=0;j<m0;j++){
            sum+=u.at<double>(i,j)*u.at<double>(i,j);
        }
        // std::cout<<sum<<std::endl;
        sum=sqrt(sum);
        // if(sum<1e-8){
        //     continue;
        // }
        for(int j=0;j<m0;j++){
            u.at<double>(i,j)/=sum;
            if(abs(u.at<double>(i,j))>max){
                max=abs(u.at<double>(i,j));
            }
        }
        if(i<10)
            hconcat(eigenface_show,vec2img(u.row(i)/max*255),eigenface_show);
    }
    Mat data=u*X.t();
    imwrite(output_path+"_eigenface.jpg",eigenface_show);

    cv::FileStorage file(output_path, cv::FileStorage::WRITE);
    // cv::Mat someMatrixOfAnyType;

    // Write to file!
    file << "mean" << mean << "u" << u << "labels" << labels<<"data"<<data<<"X"<<X;

    // Close the file and release all the memory buffers
    file.release();
}

float eigenface_test(std::string model_path,std::vector<std::pair<Mat,int>> test,int PCA_num){
    cv::FileStorage file(model_path, cv::FileStorage::READ);
    Mat mean,u,data,X;
    std::vector<int> labels;
    file["mean"] >> mean;
    file["u"] >> u;
    file["labels"] >> labels;
    file["data"] >> data;
    file["X"] >> X;
    file.release();

    // int PCA_num=int((energy/100*test.size()));
    int n0=u.rows,m0=u.cols;
    u=u(Range(0,PCA_num),Range(0,m0));
    n0=u.rows,m0=u.cols;
    // std::cout<<n0<<" "<<m0<<std::endl;
    Mat img;
    float correct=0;
    for(auto xk: test){
        // important clone
        img=xk.first.clone();
        img=img-mean;
        Mat tmp=u*img.t();
        // std::cout<<tmp<<std::endl;
        // return 0;
        double last_sum=-1,sum=0;
        int last_idx=0;
        // Mat new_img=Mat::zeros(1,img.cols,CV_64F);
        for(int i=0;i<data.cols;i++){
            sum=0;
            // std::cout<<data.rows<<" "<<tmp.rows<<std::endl;
            for(int j=0;j<tmp.rows;j++){
                // std::cout<<data.at<double>(j,i)<<" "<<tmp.at<double>(j,0)<<" |";
                sum+=(data.at<double>(j,i)-tmp.at<double>(j,0))*(data.at<double>(j,i)-tmp.at<double>(j,0));
                // std::cout<<sum<<" ";
            }
            //  std::cout<<sum<<" ";
            if(last_sum<0||sum<last_sum){
                last_sum=sum;
                last_idx=i;
            }
            // if(i>3) break;
        }
        // std::cout<<last_sum<<" "<<xk.second<<" "<<last_idx<<" "<<labels[last_idx]<<std::endl;
        if(labels[last_idx]==xk.second){
            correct+=1;
        }
        // break;
    }
    // std::cout<<correct<<" "<<test.size()<<std::endl;
    return correct/test.size();
}
void test_all(){
    std::vector<std::pair<Mat,int>> train,test;
    load_dataset(0.5,"processed_image",train,test);
    eigenface_train(100,"model_test.yml",train);

    // float x=eigenface_test_all("model_test.yml",test,5);
    // std::cout<<x<<std::endl;
    cv::Mat image = cv::Mat::zeros(210, 200, CV_8UC3);
    image.setTo(cv::Scalar(255, 255, 255));
    std::vector<cv::Point> points;
    std::vector<double> points_py;
    std::cout<<"Start Ploting rank1-pc figure"<<std::endl;
    for(int PCA_num=1;PCA_num<=200;PCA_num+=1){
        float x=eigenface_test("model_test.yml",test,PCA_num);
        points.push_back(cv::Point(PCA_num, (1-x)*200));
        points_py.push_back(x);
        // std::cout<<"!"<<PCA_num<<" "<<x<<std::endl;
    }
    cv::FileStorage file("points_py.yml", cv::FileStorage::WRITE);
    // cv::Mat someMatrixOfAnyType;

    // Write to file!
    file << "points" << points_py<<"gap" << 1;

    // Close the file and release all the memory buffers
    file.release();
    // for (int i = 0; i < points.size(); i++){
	// 	cv::circle(image, points[i], 5, cv::Scalar(0, 0, 0), 2, 8, 0);
	// }
    cv::polylines(image, points, false, cv::Scalar(0, 255, 0), 1, 8, 0);
    imwrite("test_all.jpg",image);
    // x=eigenface_test("model_test.yml",test,5);
    // std::cout<<x<<std::endl;
    // x=eigenface_test_all("model_test.yml",test,5);
    // std::cout<<x<<std::endl;
}
int main(){
    // mark_eye("att-face/s41","ATT-eye-location/s41/");
    prepare_all();
    // std::vector<std::pair<Mat,int>> train,test;
    // load_dataset(0.5,"processed_image",train,test);
    // eigenface_train(50,"model.yml",train);
    // eigenface_reconstruct("processed_image/s41/1.pgm","model.yml","reconstruct.jpg");
    // eigenface_test("processed_image/s11/2.pgm","model.yml","test.jpg");
    // test_all();
    return 0;
}
// void eigenface_train_old(std::vector<std::pair<Mat,int>> &train,std::string output_path){
//     float n=train.size();
//     int m=train[0].first.cols;
//     std::cout<<n<<" "<<m<<std::endl;
//     Mat mean=Mat::zeros(1,m,CV_64F),A=Mat(m,m,CV_64F);
//     //iterate vector to calculate mean
//     std::cout<<mean.size()<<" "<<train[0].first.size()<<std::endl;
//     for(auto i : train){
//         // std::cout<<mean.size()<<" "<<i.first.size()<<std::endl;
//         mean+=i.first;
//     }
//     mean=mean/n;
//     std::cout<<mean<<std::endl;
//     for(auto i:train){
//         Mat tmp=i.first-mean;
//         // std::cout<<tmp<<std::endl;
//         // std::cout<<tmp<<std::endl;
//         A+=(tmp.t())*tmp;
//         // break;
//     }
//     A=A/n;
//     Mat w,u,vt;
//     SVD::compute(A, w);
//     std::cout<<w<<std::endl;
//     // print(n);
// }