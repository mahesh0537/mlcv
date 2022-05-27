#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <ros/ros.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/io.h>
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include <pcl_conversions/pcl_conversions.h>
// #include "pcl-1.8/pcl/conversions.h"
namespace sm = sensor_msgs;
namespace mf = message_filters;
typedef pcl::PointXYZ point_type;
typedef pcl::PointCloud<point_type> pointcloud_type;
typedef mf::sync_policies::ApproximateTime<sm::Image, sm::CameraInfo> NoCloudSyncPolicy;


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
    ROS_INFO(" working fine");
    return coefficients;
}

pointcloud_type::Ptr createPointCloud (const sm::ImageConstPtr& depth_msg, 
                                   const sm::CameraInfoConstPtr& cam_info) 
{
pointcloud_type::Ptr cloud (new pointcloud_type() );
// cloud->header.stamp     = depth_msg->header.stamp;
ros::Time time_st = ros::Time::now ();
cloud->header.stamp     = time_st.toNSec()/1e3;
// cloud->header.frame_id  = depth_msg->header.frame_id;
cloud->header.frame_id  = "/map";
cloud->is_dense         = true; //single point of view, 2d rasterized

float cx, cy, fx, fy;//principal point and focal lengths

cloud->height = depth_msg->height;
cloud->width = depth_msg->width;
cx = cam_info->K[2]; //(cloud->width >> 1) - 0.5f;
cy = cam_info->K[5]; //(cloud->height >> 1) - 0.5f;
fx = 1.0f / cam_info->K[0]; 
fy = 1.0f / cam_info->K[4]; 

cloud->points.resize (cloud->height * cloud->width);

const float* depth_buffer = reinterpret_cast<const float*>(&depth_msg->data[0]);

  int depth_idx = 0;

  pointcloud_type::iterator pt_iter = cloud->begin ();
  for (int v = 0; v < (int)cloud->height; ++v)
  {
    for (int u = 0; u < (int)cloud->width; ++u, ++depth_idx, ++pt_iter)
    {
      point_type& pt = *pt_iter;
      float Z = depth_buffer[depth_idx];

      // Check for invalid measurements
      if (std::isnan (Z))
      {
        pt.x = pt.y = pt.z = Z;
      }
      else // Fill in XYZ
      {
        pt.x = (u - cx) * Z * fx;
        pt.y = (v - cy) * Z * fy;
        pt.z = Z;
      }

    }
  }
  



pcl::ModelCoefficients::Ptr coefficients = plane(cloud);
// std::cout<< coefficients->values[0] ;
ROS_INFO(" working fine");

std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3];
  return cloud;
}


void depth_callback(const sm::ImageConstPtr& dimage, 
                    const sm::CameraInfoConstPtr& cam_info,
                    ros::Publisher* cloud_pub)
{
  pointcloud_type::Ptr pc = createPointCloud(dimage, cam_info);
  sm::PointCloud2 cloudMessage;
  pcl::toROSMsg(*pc,cloudMessage);
  cloud_pub->publish(cloudMessage);
  // delete pc;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "depth2cloud");
  ros::NodeHandle nh;
  ros::Publisher pc_pub = nh.advertise<sm::PointCloud2>("/cloud_out_whatever", 10);
  ROS_INFO("To reconstruct point clouds from openni depth images call this tool as follows:\nrosrun depth2cloud depth2cloud depth_image:=/camera/depth/image camera_info:=/camera/depth/camera_info");

  //Subscribers
  mf::Subscriber<sm::Image> depth_sub(nh, "/camera/depth/image_raw", 2);
  mf::Subscriber<sm::CameraInfo> info_sub(nh, "/camera/rgb/camera_info", 5);

  //Syncronize depth and calibration
  mf::Synchronizer<NoCloudSyncPolicy> no_cloud_sync(NoCloudSyncPolicy(2), depth_sub, info_sub);
  no_cloud_sync.registerCallback(boost::bind(&depth_callback, _1, _2, &pc_pub));

  ros::spin();
  return 0;
}
