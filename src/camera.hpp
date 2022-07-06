#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>



class camera{
    public:
        ros::NodeHandle nh_;
        image_transport::ImageTransport it_;
        image_transport::Subscriber image_sub_;
        cv::Mat img_rgb;
        std::string OPENCV_WINDOW = "Image window";
        std::string IMAGE_TOPIC = "/zed2/zed_node/left/image_rect_color";
        std::string DEPTH_TOPIC = "/zed2/zed_node/depth/depth_registered";



    public:
        camera();
        ~camera();
        void imageCb(const sensor_msgs::ImageConstPtr& msg);
};