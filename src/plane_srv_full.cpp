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
#include "pcl/point_cloud.h"

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

    if (find_plane){
        pcl::ModelCoefficients::Ptr coefficients = plane(organizedCloud);
        res.if_found = true;
        res.a = (double)coefficients->values[0];
        res.b = (double)coefficients->values[1];
        res.c = (double)coefficients->values[2];
        res.d = (double)coefficients->values[3];
        if (debug){

            std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3];

            std::cout<<"No of nan points : "<<n_nan;
            ROS_INFO(" ");
        }
    }

    if (debug){
        organizedCloud->header.frame_id = "/map";
        pcl_conversions::toPCL(ros::Time::now(), organizedCloud->header.stamp);
        sm::PointCloud2 cloudMessage;
        pcl::toROSMsg(*organizedCloud,cloudMessage);
        pc_pub.publish(cloudMessage);
    }
    return true;
}










int main(int argc, char **argv){
    ros::init(argc, argv, "point_cloud_node");
    ros::NodeHandle nh;
    pc_pub = nh.advertise<sm::PointCloud2> ("/full_cloud_out", 10);
    ros::ServiceServer service = nh.advertiseService("depth_to_cloud", plane_seg_callback);

    ROS_INFO("Plane segmentation service ready !");
    ros::spin();

    return 0;
}
