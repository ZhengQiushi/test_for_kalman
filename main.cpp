/**
 * 本文件用于实现人脸的识别与预测
 *
 * 已有功能:
 * -实现了人脸的检测,定位与投影变换
 * -新添加了kalman滤波,并有匀速和匀加速两者模型
 *
 * 不足:
 * -只适用于单目标预测
 * -没有建立跟随体系
 *
 * */

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <cstdlib>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;


const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 177.0, 123.0);
const float confidenceThreshold = 0.5;

void face_detect_dnn();


int main(int argc, char** argv)
{
    /**录视频专用*/
//    VideoCapture cam(0);
//    VideoWriter vw;
//    vw.open("sample.avi",
//            VideoWriter::fourcc('X','2','6','4'),
//            30,
//            Size(cam.get(CAP_PROP_FRAME_WIDTH),cam.get(CAP_PROP_FRAME_HEIGHT))
//            );
//    Mat mm;
//
//    for(;;){
//        cam.read(mm);
//        imshow("i",mm);
//        vw.write(mm);
//        if(waitKey(5)=='q')
//            break;
//    }

    //kalman_sample();
    face_detect_dnn();
    //waitKey(0);
    return 0;
}
Rect last_frame =Rect(Point(0,0),Point(0,0));
/* detectAndDraw
 * params @ frame(扫描图像) net(网络结构) faces(输出的脸的二维坐标)
 * func @ 检测图像中的人脸（在输入图像上做好标记）并返回二维坐标
 * */
void detectAndDraw( Mat& frame, dnn::Net net,vector<Rect>& faces){
    faces.clear();
    int64 start = getTickCount();

    if (frame.channels() == 4)
        cvtColor(frame, frame, COLOR_BGRA2BGR);

    /*输入数据调整*/
    Mat inputBlob = blobFromImage(frame, inScaleFactor,
                                  Size(inWidth, inHeight), meanVal, false, false);
    net.setInput(inputBlob, "data");

    /*人脸检测*/
    Mat detection = net.forward("detection_out");

    vector<double> layersTimings;
    double freq = getTickFrequency() / 1000;
    double time = net.getPerfProfile(layersTimings) / freq;

    /*投影到人脸检测到的图像上*/
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    ostringstream ss;
    int count =0;
    for (int i = 0; i < detectionMat.rows; i++){
        /*置信度 0～1之间*/
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > confidenceThreshold){
            /*找到角点*/
            int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
            int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
            int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
            int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

            /*人脸矩形*/
            Rect object((int)xLeftBottom, (int)yLeftBottom,
                        (int)(xRightTop - xLeftBottom),
                        (int)(yRightTop - yLeftBottom));

            /*载入容器*/
            faces.push_back(object);

            /*框脸*/
            rectangle(frame, object, Scalar(0, 255, 0));
            cv::circle(frame, Point(object.tl().x+object.width/2, object.tl().y+object.height/2),
                       2, CV_RGB(0,255,0), -1);

            cout<<"face"<<++count<<endl;
            ss << confidence;
            String conf(ss.str());
            String label = "Face: " + conf;

            int baseLine = 0;
            /*打印出人脸标号及置信度*/
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
                                  Size(labelSize.width, labelSize.height + baseLine)),
                      Scalar(255, 255, 255), FILLED);

            putText(frame, label, Point(xLeftBottom, yLeftBottom),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
        }
    }

    float fps = getTickFrequency() / (getTickCount() - start);
    ss.str("");
    ss << "FPS: " << fps << " ; inference time: " << time << " ms";
    putText(frame, ss.str(), Point(20, 20), 0, 0.75, Scalar(0, 0, 255), 2, 8);

}
/*
 * */
string deleteAllMarks(string& str, const string& mark){
    size_t len = mark.length();
    while (1){
        size_t pos = str.find(mark);
        if (pos == string::npos){
            return str;
        }
        str.erase(pos, len);
    }
}

/*inputCalibration
 * params @ dir(输入的相机参数文本路径) cameraMatrix(相机矩阵) distCoeffs(畸变参数)
 * func @ 从指定路径读入相机外参
 * return @
 * */
int inputCalibration(const string & dir,Mat &cameraMatrix,Mat &distCoeffs ){
    std::cout << "准备读取相机标定参数，按任意键继续" << endl;
    waitKey(0);
    cameraMatrix = Mat(3, 3, CV_64FC1);
    distCoeffs = Mat(1, 5, CV_64FC1);
    vector<double> cameraMatrixVector;
    vector<double> distCoeffsVector;

    string line;
    ifstream in(dir);  //读入文件
    if(!in){
        std::cout << "未找到相应相机标定文件，按任意键继续" << endl;
        waitKey(0);
        return -1;
    }
    int i,j;
    for(i=0;getline(in, line);i++){
        deleteAllMarks(line,"[");
        char *str = (char *)line.c_str();//string --> char
        const char *split = ",";
        char *p = strtok (str,split);//逗号分隔依次取出
        double a;
        for(;p != NULL;){
            sscanf(p, "%lf", &a);
            cout<<a<<endl;
            p = strtok(NULL,split);
            if(i<3){
                cameraMatrixVector.push_back(a);
            }
            else{
                distCoeffsVector.push_back(a);
            }
        }
    }

    for(i=0;i<3;i++){
        for(j=0;j<3;j++)
            cameraMatrix.at<double>(i,j)=cameraMatrixVector[i*3+j];
    }
    for(i=0;i<5;i++){
        distCoeffs.at<double>(0,i)=distCoeffsVector[i];
    }//calibrateLists
    cout << "相机内参数矩阵：" << endl;
    cout<<cameraMatrix<<endl;
    cout << "畸变系数："<<endl;
    cout<<distCoeffs<<endl;

    std::cout << "相机标定参数已传入，按任意键继续" << endl;
    waitKey(0);
    return 1;
}

/*PNPsolver
 * params @ dir(输入的相机参数文本路径) cameraMatrix(相机矩阵) distCoeffs(畸变参数)
 * func @ 从指定路径读入相机外参
 * return @
 * */
bool PNPsolver(const std::vector<cv::Point2f>& img,
               const Mat& cameraMatrix,const Mat& distCoeffs,
               double &distance, std::vector<double>&angels, std::vector<double>&euroangels)
{
    /*object size*/
    const double halfwidth =  145 / 2.0;
    const double halfheight = 210 / 2.0;
    std::vector<Point3f> obj
            {
                    Point3f(-halfwidth,  halfheight, 0),   //tl
                    Point3f( halfwidth,  halfheight, 0),   //tr
                    Point3f( halfwidth, -halfheight, 0),   //br
                    Point3f(-halfwidth, -halfheight, 0)    //bl
            };

    Mat rVec = Mat::zeros(3, 1, CV_64FC1);
    Mat tVec = Mat::zeros(3, 1, CV_64FC1);

    /*具体pnp相机姿位结算*/
    if (!solvePnP(obj, img, cameraMatrix, distCoeffs, rVec, tVec, false, SOLVEPNP_P3P))
        return false;

    Mat_<double> rotMat(3, 3);
    /*罗德里格斯变换*/
    Rodrigues(rVec, rotMat);

    /*获得欧拉角*/
    euroangels.push_back(atan2(rotMat[2][1], rotMat[2][2]) * 57.2958);
    euroangels.push_back(atan2(-rotMat[2][0], sqrt(rotMat[2][0] * rotMat[2][0] + rotMat[2][2] * rotMat[2][2])) * 57.2958);
    euroangels.push_back(atan2(rotMat[1][0], rotMat[0][0]) * 57.2958);


    double x = tVec.at<double>(0, 0);
    double y = tVec.at<double>(1, 0);
    double z = tVec.at<double>(2, 0);

    /*获得俯仰角*/
    angels.push_back(atan2(x, z));//angels[0]= atan2(x, z);
    angels.push_back(atan2(y, sqrt(x * x + z * z)));//angels[1]= atan2(y, sqrt(x * x + z * z));

    /*获得距离*/
    distance = sqrt(x * x + y * y + z * z);

    return true;
}

/*cal_angle_distance
 * params @ faces(脸部二维矩形框)
 *          cameraMatrix(相机矩阵) distCoeffs(畸变参数)
 *          frame(进行角度距离标记了的图像) faces_position(获得脸的三维坐标)
 * func @ 通过pnp算法，由图像获得矩形框的三维坐标
 * return @
 * */
bool cal_angle_distance(const vector<Rect>& faces,
                        const Mat& cameraMatrix,const Mat& distCoeffs,
                        Mat & frame,vector<Point3d>& faces_position){

    faces_position.clear();
    /*按照单个脸矩形框进行遍历*/
    for (int i=0;i<faces.size();i++) {
        double distance;
        vector<double> angle;
        vector<double> eruangle;

        /*获取当前脸*/
        Rect cur_face=faces[i];

        /*获得top-left bottom-right 和 center 的点坐标*/
        Point2f tl = cur_face.tl();
        Point2f br = cur_face.br();
        Point2f center = Point2f((tl.x+br.x)/2,(tl.y+br.y)/2);

        /*输入的顺序为 ： tl tr rb lb*/
        vector<Point2f> vertice;
        vertice.push_back(cur_face.tl());
        vertice.push_back(cur_face.br());
        vertice.emplace_back(tl.x+cur_face.width,tl.y);
        vertice.emplace_back(br.x-cur_face.width,br.y);

        /*姿位结算 获得距离、角度、欧拉角等 其实可以直接获得三维坐标，不需要进行额外的运算*/
        if(!PNPsolver(vertice,cameraMatrix,distCoeffs,distance,angle,eruangle)){
            return false;
        }
        /*水平偏角*/
        double HA=angle[0];/*单位是rad，不是°*/
        /*竖直偏角*/
        double VA=angle[1];
        double dist=distance/1000;
        cout<<distance<<"   "<<angle[0]<<"   "<<angle[1]<<endl;

        /*打印相应的竖直*/
        std::string msg = cv::format("D:%.2f", dist);//打印内容
        cv::Point textOrigin1(center.x - 20,center.y+60);//打印位置
        cv::putText(frame, msg, textOrigin1, 1, 1, cv::Scalar(0, 255, 0));//实现打印

        msg = cv::format("HA:%.2f ",HA/3.14*180);
        Point textOrigin2(center.x - 20,center.y+60+20);
        cv::putText(frame, msg, textOrigin2, 1, 1, cv::Scalar(0, 255, 0));

        msg = cv::format("VA:%.2f ", -VA/3.14*180);
        cv::Point textOrigin3(center.x - 20,center.y+60+40);
        cv::putText(frame, msg, textOrigin3, 1, 1, cv::Scalar(0, 255, 0));
        //cout<<"sin"<<sin(3.14)<<endl;
        Point3d face_position(dist*cos(VA),dist*cos(VA)*sin(HA),-dist*sin(VA));

        msg = cv::format("(x,y,z)=(%.2f,%.2f,%.2f) ",face_position.x,face_position.y,face_position.z);
        cv::Point textOrigin4(center.x - 20,center.y+60+40+20);
        cv::putText(frame, msg, textOrigin4, 1, 1, cv::Scalar(0, 255, 0));

        /*返回三维坐标*/
        faces_position.push_back(face_position);
    }

}

/*trans_to_flat
 * params @ img(读入原来的场地图) faces_position(输入的脸三维坐标)
 * func @ 通过三维坐标实现对二维场地的投影
 * return @
 * */
bool trans_to_flat(const Mat & img ,const vector<Point3d>& faces_position){
    Mat court = img.clone();

    /*场地大小*/
    double court_length = 1.4/2;
    double court_width  = 1.5/2;

    /*相机在场地的位置*/
    Point offset(court.cols/2,50);
    circle(court,offset,7,Scalar(0,255,0),3);

    /*按脸进行遍历*/
    for(int i = 0; i<faces_position.size() ; i++){
        /*获取当前脸
         * x 水平的前后 y 表示水平的左右 z表示竖直方向
         */
        Point3d cur_player_3Dpst=faces_position[i];
        /*进行坐标的转换，注意场地图和相机获得的脸的 x,y 的区别*/
        double scale_x=cur_player_3Dpst.y/(court_width/2)*(court.cols/2);
        double scale_y=cur_player_3Dpst.x/(court_length)*(court.rows/2);

        /*投影反馈二维位置*/
        Point current_play(offset.x+scale_x,offset.y+scale_y);
        circle(court,current_play,12,Scalar(0,255,0),8);

    }
    imshow("2d_court",court);
    return true;
}
enum KALMAN_MODEL{
    CV_PATTARN = 0,
    CA_PATTARN = 1
};

/**
 * 我的卡尔曼滤波结构体
 * */
class my_KF{
    int64_t m_lastTimeStamp;///时间戳
public:
    KalmanFilter KF;///cv kalman滤波
    int stateSize ;///状态数
    int measSize ;///观测数
    int contrSize ;///控制数
    KALMAN_MODEL model;///模型选择
    cv::Mat state;///预测结果
    cv::Mat measure;///观测输入
    cv::Point3d velocity;///....

    explicit my_KF(KALMAN_MODEL m): m_lastTimeStamp(0),model(m){
        measSize =  4;///输出[x,y,w,h]
        contrSize = 0;
        if(model == CV_PATTARN){///匀速模型
            stateSize = 8;//10;//
        }
        else{
            stateSize = 10;///匀加速模型
        }
        /**初始化模型*/
        KF.init(stateSize, measSize, contrSize, CV_32F);
        state=cv::Mat(stateSize, 1, CV_32F);
        measure=cv::Mat(measSize, 1, CV_32F);

        if(model == CV_PATTARN){
            /**相应的初始化转移矩阵(随机过程系数)F和测量矩阵(观测方程系数)H*/
            KF.transitionMatrix = (Mat_<float>(stateSize,stateSize) <<
                    1,0,0,0,1,0,0,0,//1,0,
                    0,1,0,0,0,1,0,0,//0,1,
                    0,0,1,0,0,0,1,0,//0,0,
                    0,0,0,1,0,0,0,1,//0,0,
                    0,0,0,0,1,0,0,0,//0,0,
                    0,0,0,0,0,1,0,0,//0,0,
                    0,0,0,0,0,0,1,0,//0,0,
                    0,0,0,0,0,0,0,1);//,0,0);

            KF.measurementMatrix  = (Mat_<float>(measSize,stateSize,CV_32F)<<
                    1,0,0,0,0,0,0,0,//0,0,
                    0,1,0,0,0,0,0,0,//0,0,
                    0,0,1,0,0,0,0,0,//0,0,
                    0,0,0,1,0,0,0,0//,0,0
            );
        }
        else{
            KF.transitionMatrix = (Mat_<float>(stateSize,stateSize) <<
                    1,0,0,0,1,0,0,0,1,0,
                    0,1,0,0,0,1,0,0,0,1,
                    0,0,1,0,0,0,1,0,0,0,
                    0,0,0,1,0,0,0,1,0,0,
                    0,0,0,0,1,0,0,0,0,0,
                    0,0,0,0,0,1,0,0,0,0,
                    0,0,0,0,0,0,1,0,0,0,
                    0,0,0,0,0,0,0,1,0,0,
                    0,0,0,0,0,0,0,0,1,0,
                    0,0,0,0,0,0,0,0,0,1
                    );

            KF.measurementMatrix  = (Mat_<float>(measSize,stateSize,CV_32F)<<
                    1,0,0,0,0,0,0,0,0,0,
                    0,1,0,0,0,0,0,0,0,0,
                    0,0,1,0,0,0,0,0,0,0,
                    0,0,0,1,0,0,0,0,0,0
            );
        }
        /**设对角线为scalar的噪音*/
        setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(KF.errorCovPost, Scalar::all(1));
    }
/**
 * 清空
 * @param
 */
//    void clear_and_init(cv::Point3d &pos, int64_t timeStamp) {
//        /*预测模型*/
//        KF.transitionMatrix = (Mat_<float>(stateSize,stateSize) <<
//                1,0,0,0,1,0,0,0,//1,0,
//                0,1,0,0,0,1,0,0,//0,1,
//                0,0,1,0,0,0,1,0,//0,0,
//                0,0,0,1,0,0,0,1,//0,0,
//                0,0,0,0,1,0,0,0,//0,0,
//                0,0,0,0,0,1,0,0,//0,0,
//                0,0,0,0,0,0,1,0,//0,0,
//                0,0,0,0,0,0,0,1);//,0,0);
//
//        KF.measurementMatrix  = (Mat_<float>(measSize,stateSize,CV_32F)<<
//                1,0,0,0,0,0,0,0,//0,0,
//                0,1,0,0,0,0,0,0,//0,0,
//                0,0,1,0,0,0,0,0,//0,0,
//                0,0,0,1,0,0,0,0//,0,0
//        );
//        setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-5));
//        setIdentity(KF.measurementNoiseCov, cv::Scalar::all(1e-1));
//        setIdentity(KF.errorCovPost, cv::Scalar::all(1));
//
//        //KF.statePost = (cv::Mat_<float>(6, 1) << pos.x, pos.y, pos.z, 0, 0, 0);
//        m_lastTimeStamp = timeStamp;
//    }
    /**
     * 预测值
     * @param timeStamp 秒
     * @func  将预测帧放到state中
     */
    void predict(int64_t timeStamp) {//float delay, cv::Point3d &predictRelativePos
        /*Computes a predicted state.*/
        cout<<"   statePre : "<<KF.statePre.size<<"\n"<< KF.statePre.t()<<endl;
        state = KF.predict();
        cout<<"   state : "<<state.size<<"\n"<< state.t()<<endl;
        return ;
    }
    /**
     *
     * @param bBox 目标框[x,y,w,h],
     * @param timeStamp 当前时间,用于计算dT
     * @func  更新后验方差矩阵(statPost)
     */
    void correct(cv::Rect bBox, int64_t timeStamp) {
        /** 计算时间差 */
        float deltaT = (timeStamp - m_lastTimeStamp) /1000;/// 10.0;  // s
        cout<<"delta : "<<deltaT<<endl;
        assert(deltaT > 0);
        /**更新上一次的时间*/
        m_lastTimeStamp = timeStamp;
        /** 预测模型 */
        if(model == CV_PATTARN){
            KF.transitionMatrix = (cv::Mat_<float>(stateSize,stateSize)
                    <<
                    1,0,0,0,deltaT,0     ,0     ,0     ,//deltaT*deltaT/2,0              ,
                    0,1,0,0,0     ,deltaT,0     ,0     ,//0              ,deltaT*deltaT/2,
                    0,0,1,0,0     ,0     ,deltaT,0     ,//0,0,
                    0,0,0,1,0     ,0     ,0     ,deltaT,//0,0,
                    0,0,0,0,1     ,0     ,0     ,0     ,//0,0,
                    0,0,0,0,0     ,1     ,0     ,0     ,//0,0,
                    0,0,0,0,0     ,0     ,1     ,0     ,//0,0,
                    0,0,0,0,0     ,0     ,0     ,1     );//,0,0);
        }
        else{
            KF.transitionMatrix = (cv::Mat_<float>(stateSize,stateSize)
                    <<
                    1,0,0,0,deltaT,0     ,0     ,0     ,deltaT*deltaT/2,0              ,
                    0,1,0,0,0     ,deltaT,0     ,0     ,0              ,deltaT*deltaT/2,
                    0,0,1,0,0     ,0     ,deltaT,0     ,0,0,
                    0,0,0,1,0     ,0     ,0     ,deltaT,0,0,
                    0,0,0,0,1     ,0     ,0     ,0     ,0,0,
                    0,0,0,0,0     ,1     ,0     ,0     ,0,0,
                    0,0,0,0,0     ,0     ,1     ,0     ,0,0,
                    0,0,0,0,0     ,0     ,0     ,1     ,0,0,
                    0,0,0,0,0     ,0,0,0,1,0,
                    0,0,0,0,0     ,0,0,0,0,1
            );
        }
        cout<<"transitionMatrix :  "<<KF.transitionMatrix.size<<"\n"<<KF.transitionMatrix<<endl;
        measure= (Mat_<float>(measSize,1) <<bBox.x,bBox.y,bBox.width,bBox.height);
        //cout<<"measure :  "<<measure.size<<"\n"<<measure<<endl;
        /*Updates the predicted state from the measurement.*/
        KF.correct(measure);
        cout<<"   statePost : "<<KF.statePost.size<<"\n"<< KF.statePost.t()<<endl;
    }

};
int cur_frame = 0;
int err_sum = 0;
/**prediction
 * @func 实现单目标的卡尔曼预测
 * @return
 * */
void prediction(Mat& frame,my_KF&kalman_filter,vector<Rect>& faces){
    for(int i = 0 ;i < faces.size();i++ ){
        /**用于记录当前帧数 (为了计算卡尔曼误差评价平均值*/
        cur_frame++;
        /**用于打印上一帧的位置 @黄色点yellow*/
        if(last_frame.x!=0){
            cv::circle(frame, Point(last_frame.tl().x+last_frame.width/2, last_frame.tl().y+last_frame.height/2),
                       2, CV_RGB(255,255,0), -1);
        }
        /**获取时间,用于计算dT*/
        double precTick =   (double) cv::getTickCount();
        /**先修正方差*/
        kalman_filter.correct(last_frame,precTick);
        /**然后进行预测*/
        kalman_filter.predict(precTick);//
        //   cout << "State post: \n" <<kalman_filter.state.t() << endl;
        /**获取预测位置,[x,y,w,h] 左上点坐标 宽 高*/
        cv::Mat& state = kalman_filter.state;
        /**预测位置偏差太大,筛选不要*/
        if(abs(state.at<float>(0)-faces[i].x)>faces[i].x){
            continue;
        }
        /**用于打印预测框 @红色red*/
        cv::Rect predRect;
        predRect.width = state.at<float>(2);
        predRect.height = state.at<float>(3);

        cv::Point center;
        center.x = state.at<float>(0)+predRect.width / 2;
        center.y = state.at<float>(1)+predRect.height / 2;

        predRect.x = state.at<float>(0);
        predRect.y = state.at<float>(1);

        cv::circle(frame, center, 2, CV_RGB(255,0,0), -1);
        cv::rectangle(frame, predRect, CV_RGB(255,0,0), 2);
        /**评估预测误差,直接采用差的平方和*/
        Mat predict_box = state(Range(0,4),Range(0,1)).clone().t();
        Mat Real_box = (Mat_<float >(1,4)<<faces[i].x,faces[i].y,faces[i].width,faces[i].height);
        cout<<"|predict_box : "<<predict_box<<endl;
        cout<<"|Real_box    : "<<Real_box<<endl;
        Mat result ;
        cv::pow(Mat(predict_box-Real_box),2,result);

        int err= 0;
        for(int i = 0 ; i < 4 ; i++){
            err += pow(predict_box.ptr<float>(0)[i]-Real_box.ptr<float>(0)[i],2);
        }
        err_sum+=err;
        cout<<"|------- err : "<<err<<endl;
        cout<<"|------- aver_err : "<<err_sum/cur_frame<<endl;
        /**更新上一帧的位置*/
        last_frame = faces[i];
    }
}
/*face_detect_dnn
 * func @ 相当于检测主函数 进行准备工作与摄像头读取操作，并实现主体功能
 * return @
 * */
void face_detect_dnn() {
    /*配置文件*/
    String modelDesc = "../dnn_face/deploy.prototxt";//opencv_face_detector.pbtxt
    String modelBinary = "../dnn_face/res10_300x300_ssd_iter_140000_fp16.caffemodel";//opencv_face_detector_uint8.pb
    Mat frame;

    Mat court=imread("../court.jpeg");
    /**初始化卡尔曼滤波*/
    my_KF kalman_filter(CA_PATTARN);
    /**存放由检测返回的二维坐标*/
    vector<Rect> faces;
    /**存放由pnp解算的三维坐标*/
    vector<Point3d> faces_position;

    /*初始化网络*/
    dnn::Net net = readNetFromCaffe(modelDesc, modelBinary);//readNetFromTensorflow(modelBinary, modelDesc);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    if (net.empty()){
        printf("could not load net...\n");
        return;
    }

    /**打开摄像头*/
    VideoCapture capture("/dev/video0");//////sample.avi
    if (!capture.isOpened()) {
        printf("could not load camera...\n");
        return;
    }

    /**读入相机矩阵*/
    Mat_<double > cameraMatrix,distCoeffs;
    if(!inputCalibration("../dnn_face/calibrateLists.txt",cameraMatrix,distCoeffs)){
        cout<<"读取标定参数失败"<<endl;
    }

    VideoWriter vw;
    vw.open("for_show.avi",
            VideoWriter::fourcc('X','2','6','4'),
            30,
            Size(capture.get(CAP_PROP_FRAME_WIDTH),capture.get(CAP_PROP_FRAME_HEIGHT))
            );
    if( capture.isOpened() ){
        cout << "Video capturing has been started ..." << endl;
        for(;;){
            capture >> frame;
            cout<<"@@@@@@@@@@@@@@@@ new frame @@@@@@@@@@@@@@@"<<endl;
            if( frame.empty() )
                break;
            Mat frame1 = frame.clone();

            /**二维检测与回显*/
            detectAndDraw( frame, net,faces);
            cout<<"|-detectAndDraw finished"<<endl;

            /**卡尔曼预测*/
            prediction(frame,kalman_filter,faces);
            cout<<"|-prediction finished"<<endl;

            /**解算获得角度与距离*/
            if(cal_angle_distance(faces,cameraMatrix,distCoeffs,frame,faces_position)){
                return ;
            }
            cout<<"|-cal_angle_distance finished"<<endl;

            /**三维到二维平面图的投影*/
            trans_to_flat(court ,faces_position);
            cout<<"|-trans_to_flat finished"<<endl;

            imshow("dnn_face_detection", frame);
            vw.write(frame);
            cout<<"-------------------------------------"<<endl;
            char c = (char)waitKey(1 );
            if( c == 27 || c == 'q' || c == 'Q' )
                break;
        }

    }
    vw.release();
}