#!/usr/bin/env python3
#-*- coding:utf-8   -*-
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from yolo import YOLO
import os
import numpy as np
from PIL import Image as PILImage
def image_callback(rgb):
    # global img
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(rgb, "bgr8")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转变成Image
    img = PILImage.fromarray(np.uint8(img))
    # 进行检测
    img, result_label = yolo.detect_image(img)
    # pix=np.array(result_label[0:1]['midx'],result_label[0:1]['midy'])
    # print(pix[0:2])
    # print(type(result_label.values))
    pix=[]
    pix=result_label.values
    for i in pix:
        # i=i.append(0)
        #a=np.array(100)
        i=np.append(i, values=[100.0,50.0], axis=None)

        #np.insert(i,0,[100.0])

        print(i)
    img = np.array(img)
    # RGBtoBGR满足opencv显示格式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    #cv2.imwrite("/home/zzn/dobot_ws/src/realsense/scripts/img/1.jpg", img)
    # rospy.sleep(10)
    # cv2.imshow("rgb",img)

 
if __name__ == "__main__": 
    yolo = YOLO()
    #节点初始化
    rospy.init_node('listen_rgb')
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)
    rospy.spin()
