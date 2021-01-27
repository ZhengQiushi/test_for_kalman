## 预测模块汇报
### 1.老代码阅读

####  1.1 执行流程
m_preDetect --> m_classify --> m_match --> m_isEnablePredict --> correctTrajectory_and_calcEuler
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200918101642619.png#pic_center)
#### 1.2 匹配 / m_match :
形成一个历史队列
1.只保留持续时间30帧以内的
```cpp
                if (iter->rTick > 30) {
                    s_historyTargets.erase(iter, s_historyTargets.end());
```
2.击打策略的选择
- **之前历史记录为空:**
找一个离炮台最近的作为`minTarElement`
```cpp
                auto minTarElement = std::min_element(
                        m_targets.begin(), m_targets.end(), [](Target &a_, Target &b_) -> bool {
                            return cv::norm(a_.ptsInGimbal) < cv::norm(b_.ptsInGimbal);
                        });
```
- **之前记录不为空:**
通过`轮廓匹配`和`距离匹配`寻找上一次的击打对象
该轮廓是目标ROI的四个角点, 感觉不是很靠谱...
```cpp
distanceA = cv::matchShapes
```
- **改进方向:**
提取图像特征
通过一个简单的CNN提取图片特征,然后根据特征值进行匹配,参考DEEPSORT

```cpp
/**疑问 不知道什么意思*/
                    if (distanceA > 0.5 ||
                        (m_1.nu11 + m_1.nu30 + m_1.nu12) * (m_2.nu11 + m_2.nu30 + m_2.nu12) < 0)
                        continue;
```
#### 1.3 预测 / m_isEnablePredict :
- **注意的变量:**
炮口坐标系 : 
cv::Point3d ptsInGimbal;  // 物体在云台坐标系下坐标(相机坐标系经过固定变换后得到)
世界坐标系 : 
cv::Point3d ptsInWorld;  // 物体在世界坐标系下坐标

- **主要思想:**
通过kalman filter 对世界坐标系的目标的三维坐标进行预测.
具体流程如下:
>![在这里插入图片描述](https://img-blog.csdnimg.cn/20200918110703484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MTE0OQ==,size_16,color_FFFFFF,t_70#pic_center)
> 获得云台的绝对偏转角度 ```cpp m_communicator.getGlobalAngle(&gYaw, &gPitch); ```
> 转到所有的检测物体世界坐标 ```cpp tar.convert2WorldPts(-gYaw, gPitch); ```
> 对三维坐标点进行预测 ```cpp kalman.correct(s_historyTargets[0].ptsInWorld,
> timeStamp); ```

缺点:
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200918111138570.png#pic_center)

考虑到相机和物体的相对运动,三维坐标转换的预测不是那么准确.

### 2.外校代码阅读
##### 2.1太原理工大学
- **主要思想:**
kalman filter 加上二维预测
对yaw和pitch进行预测!
- **具体流程**
1.人工消除畸变
对ROI的四个角点结合相机矩阵和畸变参数进行畸变矫正.
```cpp
cv::Point2f MyCamera::undistortPoints(cv::Point2f inputPoint)
```
2.等比测距
在分类结束后,进行距离的测量.
```cpp
//利用灯柱高度计算距离
//实际height
//之前测好的数据 65000
	pridectDistance = (ushort)(65000. / height);
```
3.目标的获得绝对偏转角度
```cpp
	/**
	* @函数: caculateTargetPose
	* @描述: 得到目标的世界坐标
	* @输入: Point2f angle			敌方与摄像头成像平面中垂线的角度（x，y）
			 cv::Mat tvecDeviation  枪管到摄像头的平移矩阵
			 float distance			敌方距离
			 double timestamp		此时的时间
	* @输入输出:cv::Point2f &targetPose 敌方世界坐标
	* @返回: bool 成功转换
	*/
	bool caculateTargetPose(cv::Point2f angle, cv::Mat tvecDeviation,  float distance, double timestamp, cv::Point2f &targetPose);
```
以下是`caculateTargetPose`的具体步骤:
- 先二维坐标转云台相对偏转角度
```cpp
cv::Point2f MyCamera::pix2angle(cv::Point2f point, bool shoot42)
```
- 然后运用线性插值预测云台当前绝对偏转情况,存放于`GimblaPose gimblaPose`
```cpp
	/**
	* @函数: interpolateRobotPose
	* @描述: 根据时间对云台值进行插值
	* @输入: double timestamp 此刻时间
			 GimblaPose &pose 云台状态
	*/
	bool interpolateRobotPose(double timestamp, GimblaPose &pose);
```

- 拿距离和二维偏转角度构建**相对三维坐标**与**修正后的相对偏转角度**
```cpp
/**构建相对三维坐标*/
	tvec.at<float>(0, 0) = (float)distance * tan(angle.x / 180 * CV_PI) + tvecDeviation.at<float>(0, 0);
	tvec.at<float>(0, 1) = (float)distance * tan(angle.y / 180 * CV_PI) + tvecDeviation.at<float>(0, 1);
	tvec.at<float>(0, 2) = (float)distance + tvecDeviation.at<float>(0, 2);
/**修正后的偏转角度*/
	Point2f correctedAngle;
	correctedAngle.x = atan2(tvec.at<float>(0, 0), tvec.at<float>(0, 2)) * 180 / CV_PI;
	correctedAngle.y = atan2(tvec.at<float>(0, 1), tvec.at<float>(0, 2)) * 180 / CV_PI;
```
- 然后得到目标的**绝对偏转角度**
```cpp
	targetPose.x = gimblaPose.yaw - correctedAngle.x ;
	targetPose.y = gimblaPose.pitch - short(correctedAngle.y * 8192. / 360.);
```
最后使用kalman对二维pitch 和 yaw进行预测!
```cpp
	/**
	* @函数: kalmanPredict
	* @描述: 对绝对坐标进行预判
	* @输入: float yaw 装甲板所在的yaw角度值
			 float pitch 装甲板所在的pitch编码值
			 float time 预判时长
	* @返回：Point2f 预测后云台的yaw，pitch
	*/
	cv::Point2f kalmanPredict(float yaw, float pitch, float time);
```
- 对比:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200918114624353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MTE0OQ==,size_16,color_FFFFFF,t_70#pic_center)
##### 2.2北京理工大学珠海学院
只对yaw进行了kalman预测

######  2.3交大
无预测...
######  2.4大连交通大学
无预测...
```cpp
/**
  * @brief kalman filter for point. never use
  */
void ArmorPredict::Classical_kalman(Kalman4Point &KF)
```

**但是有许多值得学习的地方**
1.三相机加三线程
```cpp
                std::thread af1(&ArmorFind::process,FinderLeft,contours_left,std::ref(left),std::ref(leftout),std::ref(ArmorLeftResult),false);
                std::thread af2(&ArmorFind::process,FinderRight,contours_right,std::ref(right),std::ref(rightout),std::ref(ArmorRightResult),false);
                af1.join();
                af2.join();

```
双目定位:

```cpp
void StereoXML::Calculation(const Point &left,const Point &right,AbsPosition &Position){
    //std::cout<<"left y position:"<<left.y<<std::endl;
    VisionDis = left.x - right.x;
    Position.z = Param.distance_const/VisionDis;
    Position.x = (left.x - Param.width/2)*Position.z/Param.f;
    Position.y = (Param.lightbase- left.y)*Position.z/Param.f;
}
```

2.比赛中对相机的动态调整

```cpp
            /**根据距离调整分辨度**/
            if(Predictor.Data.z.f > 2500 && videotype == 1280){
                vtrans_count++;
                if(vtrans_count > 10){
                    videotype = 1920;
                    /**开启左右(上 不一定)摄像头*/
CamSetMode(capleft,capright,capup,1920,capstatus[2]);
                    vtrans_count = 0;
                    std::cout<<"format:1920"<<std::endl;
                }
            }else if((Predictor.Data.z.f < 2000) && (Predictor.Data.z.f > 1) && (videotype == 1920)){
                vtrans_count++;
                if(vtrans_count > 20){
                    videotype = 1280;
CamSetMode(capleft,capright,capup,1280,capstatus[2]);
                    vtrans_count = 0;
                    std::cout<<"format:1280"<<std::endl;
                }
            }else{
                vtrans_count = 0;
            }
        }else{
            if(videotype != 1280){
                videotype = 1280;
                CamSetMode(capleft,capright,capup,1280,capstatus[2]);
                std::cout<<"format:1280"<<std::endl;
            }
        }
```
3.根据机器人等级调节弹道补偿

```cpp
void ArmorPredict::AngleFit(const AbsPosition input,float *pitch,float *yaw,float *shootspd,int level){
...
    int shootspeedmax;
    // max bullet speed:
    // level 0 / level 1: 22m/s
    // level 2          : 25m/s
    // level 3          : 28m/s
    switch(level){
        case 0:
        case 1:shootspeedmax = 22;break;
        case 2:shootspeedmax = 25;break;
        case 3:shootspeedmax = 28;break;
    }
}
```

4.自启使用了看门狗WatchDog

#####  2.5 东南大学
1.三线程 产生一个信息队列

依次获取图片 处理图片 

##### 2.6 其他参考资料
和太原理工类似
https://zhuanlan.zhihu.com/p/38745950
###  3.改进方向
借鉴太原理工的思想
试图通过对角度进行预测!

不易测试,需要与电控结合!

