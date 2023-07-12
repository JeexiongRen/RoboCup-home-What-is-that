#!/usr/bin/env python3
#-*- coding:utf-8   -*-
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from PIL import Image as PILImage
import pyrealsense2 as rs2
from yolo import YOLO

minrot=0.1
minvel=0.01

class ImageDepthListener:

    def __init__(self, color_image_topic,depth_image_topic, depth_info_topic):
        self.bridge = CvBridge()
        self.x=0
        self.dis=0
        self.vel=Twist()
        self.vel.linear.x=0
        self.vel.linear.y=0
        self.vel.linear.z=0
        self.vel.angular.x=0
        self.vel.angular.y=0
        self.vel.angular.z=0
        self.velpub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
        self.sub_color = rospy.Subscriber(color_image_topic, msg_Image, self.imageColorCallback)
        self.sub_depth = rospy.Subscriber(depth_image_topic, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber(depth_info_topic, CameraInfo, self.imageDepthInfoCallback)
        # confidence_topic = depth_image_topic.replace('depth', 'confidence')
        # self.sub_conf = rospy.Subscriber(confidence_topic, msg_Image, self.confidenceCallback)
        self.intrinsics = None
        # self.pix = []
        # self.pix_grade = None
        # self.data = []
        self.depthimage = None

    def get_3d_camera_coordinate(self,x,y):
        pix_x=int(x)
        pix_y=int(y)
        if self.intrinsics:
            depth = self.depthimage[pix_y, pix_x]/1000.0 
            # result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [pix_x, pix_y], depth)
        return depth
    
    def imageDepthCallback(self, data):
        try:
            # print(self.pix)
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            self.depthimage=cv_image
            # if len(self.pix)!=0:
            #     for i in self.pix:
            #         pix=i[1:3]       
            #         # print(type(self.pix))
            #         pix[0]=int(i[1])
            #         pix[1]=int(i[2])
            #     # pick one pixel among all the pixels with the closest range:
            #     # indices = np.array(np.where(cv_image == cv_image[cv_image > 0].min()))[:,0]
                    
            #             # line += '  Coordinate: %8.5f %8.5f %8.5f.' % (result[0], result[1], result[2])
            #         # if (not self.pix_grade is None):
            #         #     # line += ' Grade: %2d' % self.pix_grade
            #         # line += '\r'
            #         # sys.stdout.write(line)
            #         # sys.stdout.flush()
            #         # self.data.append(self.func(cv_image,pix[0],pix[1]))
            #         data=self.func(pix[0],pix[1])
            #         print(data)

        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            return
        
    def imageColorCallback(self,data):
        # global img
        try:
            self.x=320
            self.dis=0.7
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        
            img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # 转变成Image
            img = PILImage.fromarray(np.uint8(img))
            # 进行检测
            img, result_label = yolo.detect_image(img)
            
            if not result_label.empty:
                for i in result_label.values:
                    depth=self.get_3d_camera_coordinate(i[1],i[2])
                    print(i,depth)
                self.x=i[1]
                self.dis=depth
                # print(result_label)
            
            if self.x > 335:
                print('Right')
                self.vel.linear.x=0
                self.vel.angular.z=-(self.x-320)/800-minrot
            elif self.x < 305:
                print('Left')
                self.vel.linear.x=0
                self.vel.angular.z=-(self.x-320)/800+minrot
            else:
                self.vel.angular.z=0
                if self.dis >0.80:
                    print('Front')
                    self.vel.linear.x=(self.dis-0.7)/2+minvel
                    self.vel.angular.z=0
                elif self.dis <0.60:
                    print('Back')
                    self.vel.linear.x=(self.dis-0.7)/2-minvel
                    self.vel.angular.z=0
                else:
                    print('Wait')
                    self.vel.linear.x=0
                    self.vel.angular.z=0
            
            #self.velpub.publish(self.vel)
            
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
 
if __name__ == "__main__": 
    yolo = YOLO()
    #节点初始化
    rospy.init_node('listen_rgb_and_depth')
    color_image_topic = '/camera/color/image_raw'
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'

    listener = ImageDepthListener(color_image_topic,depth_image_topic, depth_info_topic)

    rospy.spin()
    # rospy.Subscriber('/camera/color/image_raw', msg_Image, image_callback)
    # rospy.spin()
