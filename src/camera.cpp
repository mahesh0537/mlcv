#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include "camera.hpp"
#include<iostream>



camera::camera(): it_(nh_){
            image_sub_ = it_.subscribe(this->IMAGE_TOPIC, 1, &camera::imageCb, this);
            // ROS_INFO("whatever");
        }

camera::~camera(){
    cv::destroyWindow(OPENCV_WINDOW);
}


void camera::imageCb(const sensor_msgs::ImageConstPtr& msg){
            try{
                this->img_rgb = (cv_bridge::toCvCopy(msg, "bgr8"))->image;
                std::cout<<"whatever"<<std::endl;
                ROS_INFO(" lol");
                cv::imshow("OPENCV_WINDOW", this->img_rgb);
    cv::waitKey(3);
                


            }
            catch(cv_bridge::Exception& e){
                ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
            }
            
        }



int main(int argc, char **argv){
    ros::init(argc, argv, "image_sub");
    std::cout<<"ros node init done"<<std::endl;
    ROS_INFO(" lol");
    camera zed = camera();
    ROS_INFO(" zed started");
    
                    ros::spin();



    
}
