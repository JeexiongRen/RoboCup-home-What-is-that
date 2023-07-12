#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from PIL import Image
from yolo import YOLO
import pandas as pd
import math
import rospy
from std_msgs.msg import String
from robot_msgs.msg import object_position
# 一个相机识别并发布坐标
# written by zzn
camera_state = False

# 计算像素点深度与三维坐标
def get_3d_camera_coordinate(x, y, aligned_depth_frame, depth_intrin):
    pix_x=int(x)
    pix_y=int(y)
    dis = aligned_depth_frame.get_distance(pix_x,pix_y)        # 获取该像素点对应的深度
    #print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [pix_x,pix_y], dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

# 接收信号，决定是否开始检测
def CameraStateCallback(msg):
    global camera_state
    if msg.data == "camera_open":
        camera_state = True
    elif msg.data == "camera_close":
        camera_state = False

if __name__ == "__main__":
    # ROS节点初始化
    rospy.init_node('tidyup_camera')
    # 创建ROS话题发布者
    pub = rospy.Publisher("/detect_result2",object_position,queue_size=10)
    # 订阅话题，决定是否进行检测
    sub = rospy.Subscriber("/camera_state", String, CameraStateCallback)
    # 发布坐标信息
    p = object_position()
    rate = rospy.Rate(10) # 10hz

    fps = 0.0
    yolo = YOLO()
    #################### 配置相机 ########################3
    # 相机RGB和深度图对齐
    align = rs.align(rs.stream.color)
    # 配置视频流
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 创建一个上下文对象。该对象拥有所有连接的Realsense设备的句柄
    pipeline = rs.pipeline()
    # 开启视频流
    profile = pipeline.start(config)    

    try:
        while not rospy.is_shutdown():
            pTime=time.time()
            ##################### 相机启动 ########################
            frames = pipeline.wait_for_frames()
            # 将RGBD对齐
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参
            
            img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # 转变成Image
            img = Image.fromarray(np.uint8(img))
            if camera_state == True:
                # 进行检测
                img, result_label = yolo.detect_image(img)
                if not result_label.empty:
                    for i in result_label.values:
                        # 计算识别中心点转换后的坐标
                        dist, result=get_3d_camera_coordinate(i[1],i[2],depth_frame, depth_intrin)
                        p.label = i[0]
                        p.x = result[0]
                        p.y = result[1]
                        p.z = result[2]

                        # 坐标点发布，不发布识别结果为0的点
                        if (p.x**2+p.y**2+p.z**2>0.1) :
                            pub.publish(p)
                            rospy.loginfo('%s:(%f,%f,%f)' %(p.label,p.x,p.y,p.z))
                        else:
                            rospy.logwarn('%s:(%f,%f,%f)' %(p.label,p.x,p.y,p.z))

            img = np.array(img)
            # RGBtoBGR满足opencv显示格式
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 显示帧率
            cTime=time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)

            if cv2.waitKey(1) == ord("q"):
                break
    except rospy.ROSInterruptException:
        pipeline.stop()
        pass