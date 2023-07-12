#!/usr/bin/env python3
#-*- coding:utf-8   -*-
# by zzn txy
import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from robot_msgs.msg import position
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import sys
import numpy as np
from PIL import Image as PILImage
import pyrealsense2 as rs2
from yolo import YOLO

class ImageDepthListener:

    # 类初始化
    def __init__(self, color_image_topic,depth_image_topic, depth_info_topic,camera_state_topic):
        # 初始化cv_bridge
        self.bridge = CvBridge()
        # 订阅彩色图
        self.sub_color = rospy.Subscriber(color_image_topic, msg_Image, self.imageColorCallback)
        # 订阅深度图
        self.sub_depth = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        # 订阅相机信息
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        # 订阅相机状态
        self.sub_cam = rospy.Subscriber(camera_state_topic, String, self.CameraStateCallback)
        # 初始化内参
        self.intrinsics = None
        # 初始化深度图
        self.depthimage = None
        self.camera_state = False

    # 计算得到物体在相机坐标系下的三维坐标
    def get_3d_camera_coordinate(self,x,y):
        pix_x=int(x)
        pix_y=int(y)
        if self.intrinsics:
            # 获取深度值
            depth = self.depthimage[pix_y, pix_x]/1000.0 
            # 计算三维坐标
            result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix_x, pix_y], depth)
        return result
    
    # 深度图回调函数，存储
    def imageDepthCallback(self, data):
        try:
            # 类型转化
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.depthimage=cv_image
        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return
        
    # 彩色图回调函数
    def imageColorCallback(self,data):
        try:
            # 类型转化
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # 转变成Image
            img = PILImage.fromarray(np.uint8(img))
            # 进行检测
            if self.camera_state==True:
                img, result_label = yolo.detect_image(img)
                if not result_label.empty:
                    for i in result_label.values:
                        # 计算识别中心点转换后的坐标
                        result=self.get_3d_camera_coordinate(i[1],i[2])
                        p.x = result[0]
                        p.y = result[1]
                        p.z = result[2]
                        # 坐标点发布
                        if (p.x**2+p.y**2+p.z**2>0.1) :
                            pub.publish(p)
                            rospy.loginfo('%s:(%f,%f,%f)' %(i[0],p.x,p.y,p.z))
                        else:
                            rospy.logwarn('%s:(%f,%f,%f)' %(i[0],p.x,p.y,p.z))
                        
            img = np.array(img)
            # RGBtoBGR满足opencv显示格式
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return

    # 相机信息回调函数，获得相机内参
    def imageDepthInfoCallback(self, cameraInfo):
        try:
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return
 
    def CameraStateCallback(self,msg):
        if msg.data == "camera_open":
            self.camera_state = True
        elif msg.data == "camera_close":
            self.camera_state = False

if __name__ == "__main__": 
    yolo = YOLO()
    # 节点初始化
    rospy.init_node('listen_rgb_and_depth')

    # 订阅话题
    color_image_topic = '/camera/color/image_raw'
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'
    listener = ImageDepthListener(color_image_topic,depth_image_topic, depth_info_topic)

    # 发布节点初始化
    pub = rospy.Publisher("/position_camera",position,queue_size=10)
    p = position()

    rospy.spin()





