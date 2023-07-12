#! /usr/bin/env python
import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from PIL import Image
from yolo_P import YOLO as YOLO_P
from yolo_F import YOLO as YOLO_F
import pandas as pd
import math
import rospy
from std_msgs.msg import String
from robot_msgs.msg import object_position

# 两个相机同时使用
# written by zzn
camera_state = False
camera_use = 1
camera_change = False

# 计算像素点深度与三维坐标
def get_3d_camera_coordinate(x, y, aligned_depth_frame, depth_intrin):
    pix_x=int(x)
    pix_y=int(y)
    dis = aligned_depth_frame.get_distance(pix_x,pix_y)        # 获取该像素点对应的深度
    #print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [pix_x,pix_y], dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

# 用来决定使用哪个相机，以及是否进行检测
def CameraStateCallback(msg):
    global camera_state
    global camera_use
    global camera_change
    if msg.data == "camera_open":
        camera_state = True
        camera_use = 1
        camera_change = False
    elif msg.data == "camera_close":
        camera_state = False
        # camera_use = 1
        camera_change = False
    elif msg.data =="camera_1" and camera_use==0:
        camera_use = 1
        camera_change = True
        # camera_state = True
    elif msg.data =="camera_0" and camera_use==1:
        camera_use = 0
        camera_change = True
        camera_state = True

# 切换相机
def change_camera(pipeline):
    global camera_use
    global camera_change
    if camera_use==0:
        pipeline.stop()
        pipeline = rs.pipeline()
        pipeline.start(config0)
        camera_use = 0
        camera_change = False
    elif camera_use==1:
        pipeline.stop()
        pipeline = rs.pipeline()
        pipeline.start(config1)
        camera_use=1
        camera_change = False
    return pipeline

if __name__ == "__main__":
    # ROS节点初始化
    rospy.init_node('tidyup_camera')
    # 创建ROS话题发布者，两个相机发布话题不同
    pub = rospy.Publisher("/detect_result", object_position, queue_size=10)
    pub2 = rospy.Publisher("/detect_result2", object_position, queue_size=10)
    # 订阅
    sub = rospy.Subscriber("/camera_state", String, CameraStateCallback)
    p = object_position()
    rate = rospy.Rate(10) # 10hz

    fps = 0.0
    # 载入两个模型
    yolo_p = YOLO_P()
    yolo_f=YOLO_F()
    #################### 配置相机 ########################3
    #realsense双摄像机设置，0是俯视摄像头（上），1是平视摄像头（下）
    connect_device = []
    for d in rs.context().devices:
        print('Found device: ',
                d.get_info(rs.camera_info.name), ' ',
                d.get_info(rs.camera_info.serial_number))
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            connect_device.append(d.get_info(rs.camera_info.serial_number))
    if len(connect_device) < 2:
        print('Registrition needs two camera connected.But got one.')
        exit()

    pipeline = rs.pipeline()
    config0 = rs.config()
    config0.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config0.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config0.enable_device(connect_device[0])

    config1 = rs.config()
    config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config1.enable_device(connect_device[1])

    # 默认使用平视相机与平视模型
    pipeline.start(config1)
    align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)
    yolo=yolo_p

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
            # 是否切换相机
            if camera_change==True:
                pipeline = change_camera(pipeline)
            # 是否检测
            if camera_state==True:
                # 两个相机使用两个模型，并分别发布
                if camera_use == 0:
                    PUB=pub2
                    yolo=yolo_f
                elif camera_use == 1:
                    PUB=pub
                    yolo=yolo_p

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

                        # 坐标点发布
                        if (p.x**2+p.y**2+p.z**2>0.1) :
                            PUB.publish(p)
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