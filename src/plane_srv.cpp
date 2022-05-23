#include <sensor_msgs/Image.h>
#include <iostream>
#include <pcl/io/io.h>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float64.h>
#include <ros/ros.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/common/centroid.h>
#include "pcl/point_cloud.h"
#include <pcl/filters/filter.h>

#include "mlcv/plane.h"


namespace sm = sensor_msgs;
typedef pcl::PointXYZ point_type;
typedef pcl::PointCloud<point_type> pointcloud_type;

ros::Publisher pc_pub;

pcl::ModelCoefficients::Ptr plane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients (true);
    seg.setMaxIterations(1000);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);
    return coefficients;
}

bool plane_seg_callback(mlcv::plane::Request &req,
                        mlcv::plane::Response &res){


    // data from service call
    bool debug, find_plane;
    debug = req.debug; find_plane = req.find_plane;
    uint32_t x1, y1, x2, y2;
    x1 = req.x1; y1 = req.y1; x2 = req.x2; y2 = req.y2;
    double cx, cy, fx, fy; 
    cx = req.cx; cy = req.cy; fx = req.fx; fy = req.fy;
    const float* depth_buffer = reinterpret_cast<const float*>(&req.depth[0]);




    /// pcl cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr organizedCloud(new pcl::PointCloud< pcl::PointXYZ>);
    int depth_idx = 0;
    organizedCloud->width = (uint32_t)((int)y2 - (int)y1);
    organizedCloud->height = (uint32_t)((int)x2 - (int)x1) ;
    organizedCloud->is_dense = false;
    organizedCloud->points.resize(organizedCloud->height*organizedCloud->width);
    int n_nan = 0;
    for(std::size_t v = 0; v < organizedCloud->width; v++){
        depth_idx = (int)organizedCloud->width*((int)y1 + v) + (int)x1;
        for(std::size_t u = 0; u < organizedCloud->height; u++){
            float Z = depth_buffer[depth_idx + u];
            if (std::isnan (Z)){
                n_nan++ ;
                organizedCloud->at(v,u).x = Z;
                organizedCloud->at(v,u).y = Z;
                organizedCloud->at(v,u).z = Z;
            }
            else {
                organizedCloud->at(v,u).x = ((double)x1 + u - cx) * Z * fx;
                organizedCloud->at(v,u).y = ((double)y1 + v - cy) * Z * fy;
                organizedCloud->at(v,u).z = Z;}
        }
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr organizedCloud_noNaN(new pcl::PointCloud< pcl::PointXYZ>);
    organizedCloud_noNaN->is_dense = false;
    std::vector<int> ind;
    pcl::removeNaNFromPointCloud(*organizedCloud, *organizedCloud_noNaN, ind);

    if (find_plane){
        if(organizedCloud_noNaN->size() > 0){
            pcl::ModelCoefficients::Ptr coefficients = plane(organizedCloud_noNaN);
            pcl::PointXYZ c1;
            pcl::computeCentroid(*organizedCloud_noNaN, c1);
            res.if_found = true;
            res.a = (double)coefficients->values[0];
            res.b = (double)coefficients->values[1];
            res.c = (double)coefficients->values[2];
            res.d = (double)coefficients->values[3];
            res.x = (double)c1.x;
            res.y = (double)c1.y;
            res.z = (double)c1.z;
        if (debug){

            std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3];

            std::cout<<"No of nan points : "<<n_nan;
            ROS_INFO(" ");
        }}
    }

    if (debug){
        organizedCloud_noNaN->header.frame_id = "/map";
        pcl_conversions::toPCL(ros::Time::now(), organizedCloud_noNaN->header.stamp);
        sm::PointCloud2 cloudMessage;
        pcl::toROSMsg(*organizedCloud_noNaN,cloudMessage);
        pc_pub.publish(cloudMessage);
    }
    return true;
}










int main(int argc, char **argv){
    ros::init(argc, argv, "plane_seg_node");
    ros::NodeHandle nh;
    pc_pub = nh.advertise<sm::PointCloud2> ("/cloud_out", 10);
    ros::ServiceServer service = nh.advertiseService("plane_seg", plane_seg_callback);

    ROS_INFO("Plane segmentation service ready !");
    ros::spin();

    return 0;
}