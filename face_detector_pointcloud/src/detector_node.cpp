#include "ros/ros.h"
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>

#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

#include <Eigen/Core>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/centroid.h>

#include <vector>
#include <cmath>
#include "mhri_social_msgs/FaceDetection3D.h"
#include "geometry_msgs/PointStamped.h"
#include "sensor_msgs/Image.h"


class FaceDetectorPointCloud
{
public:
    FaceDetectorPointCloud()
    {
        image_sub_ =  new message_filters::Subscriber<sensor_msgs::Image>(nh_, "camera/rgb/image_rect_color", 10);
		pointcloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, "camera/depth_registered/points", 10);
        sync_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *image_sub_, *pointcloud_sub_);
		sync_->registerCallback(boost::bind(&FaceDetectorPointCloud::callback, this, _1, _2));

        result_pub_ = nh_.advertise<mhri_social_msgs::FaceDetection3D>("face_detected", 10);

        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
        std::string packagePath = ros::package::getPath("face_detector_pointcloud");

        cascade_gpu_ = cv::cuda::CascadeClassifier::create(packagePath + "/data/haarcascade_frontalface_alt.xml");
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    }

    ~FaceDetectorPointCloud()
    {
        delete sync_;
        delete image_sub_;
        delete pointcloud_sub_;
    }

public:
	void callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::PointCloud2ConstPtr& pointcloud)
    {
        // Get RGB888 image from sensor_msgs::Image
		cv_bridge::CvImagePtr cv_ptr;
		try
        {
            cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
        }
		catch(cv_bridge::Exception& e)
        {
			ROS_ERROR("cv_bridge exception: %s", e.what());
			return;
    	}

        // Get PointCloudXYZRGB from sensor_msgs::PointCloud2
    	pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    	pcl::fromROSMsg(*pointcloud, pcl_cloud);


        // Find the faces
        cv::cuda::GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;
        std::vector<cv::Rect> faces;
        std::vector<cv::Rect> faces_roi;
        double scale = 1.0;

        frame_gpu.upload(cv_ptr->image);
        cv::cuda::cvtColor(frame_gpu, gray_gpu, cv::COLOR_BGR2GRAY);
        cv::Size sz(cvRound(gray_gpu.cols * scale), cvRound(gray_gpu.rows * scale));
        cv::cuda::resize(gray_gpu, resized_gpu, sz);


        bool findLargestObject = true; // Find Multi Faces
        bool filterRects = true;

        cascade_gpu_->setFindLargestObject(findLargestObject);
        cascade_gpu_->setScaleFactor(1.2);
        cascade_gpu_->setMinNeighbors((filterRects || findLargestObject) ? 4 : 0);
        cascade_gpu_->detectMultiScale(resized_gpu, facesBuf_gpu);
        cascade_gpu_->convert(facesBuf_gpu, faces);

        faces_roi.resize(faces.size());
        for (size_t i = 0; i < faces.size(); ++i)
        {
            double xc = faces[i].x + faces[i].width / 2.0;
            double yc = faces[i].y + faces[i].height / 2.0;

            faces_roi[i] = cv::Rect(xc - 10 , yc - 10, 20, 20);

            rectangle(cv_ptr->image, faces[i], CV_RGB(255, 255, 255), 2);
            rectangle(cv_ptr->image, faces_roi[i], CV_RGB(0, 255, 0), 2);
        }

        // Find the centroid points
        std::vector<Eigen::Vector4f>faces_centroid;
        // faces_centroid.resize(faces.size());

        for (size_t i = 0; i < faces.size(); ++i)
        {
            pcl::PointCloud<pcl::PointXYZRGB> pcl_centroid_points;
            for(int y = 0; y < faces_roi[i].height; y++)
            {
                for(int x = 0; x < faces_roi[i].width; x++)
                {
                    cv::Point pt;
                    pt.x = faces_roi[i].x + x;
                    pt.y = faces_roi[i].y + y;
                    pcl_centroid_points.push_back(pcl_cloud(pt.x, pt.y));
                }
            }

            Eigen::Vector4f centroid;
            unsigned int ret = pcl::compute3DCentroid(pcl_centroid_points, centroid);
            faces_centroid.push_back(centroid);
        }

        // if(faces_centroid.size() > 0)
        //     printf("%f %f %f\n", faces_centroid[0][0], faces_centroid[0][1], faces_centroid[0][2]);

        // Publish
        mhri_social_msgs::FaceDetection3D msg;

        int sum = 0;
        for(size_t i = 0; i < faces_centroid.size(); i++)
        {
            if(std::isnan(faces_centroid[i][0]) || std::isnan(faces_centroid[i][1]) || std::isnan(faces_centroid[i][2]))
                continue;

            geometry_msgs::PointStamped pt;
            pt.header.stamp = ros::Time::now();
            pt.header.frame_id = "camera_depth_optical_frame";
            pt.point.x = faces_centroid[i][0];
            pt.point.y = faces_centroid[i][1];
            pt.point.z = faces_centroid[i][2];

            msg.faces_pose.push_back(pt);

            cv::Mat subimage(cv_ptr->image, faces[i]);
            resize(subimage, subimage, cv::Size(200, 200));
            sensor_msgs::ImagePtr image = cv_bridge::CvImage(std_msgs::Header(), "bgr8", subimage).toImageMsg();

            msg.faces_image.push_back(*image);
            sum++;
        }

        msg.num_of_detected = sum;
        result_pub_.publish(msg);

        //
        cv::imshow("Display window", cv_ptr->image);
        cv::waitKey(1);
    }

private:
    ros::NodeHandle nh_;
    message_filters::Subscriber<sensor_msgs::Image> *image_sub_;
	message_filters::Subscriber<sensor_msgs::PointCloud2> *pointcloud_sub_;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;
	message_filters::Synchronizer<MySyncPolicy> *sync_;
    ros::Publisher result_pub_;

    cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu_;
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "face_detector_pointcloud");
	FaceDetectorPointCloud hd;
	ros::spin();
    return 0;
}
