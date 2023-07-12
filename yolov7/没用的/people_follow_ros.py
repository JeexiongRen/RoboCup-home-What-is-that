#!/usr/bin/env python3
#-*- coding:utf-8   -*-
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import sys
import numpy as np
from PIL import Image as PILImage
import pyrealsense2 as rs2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

minrot=0.1
minvel=0.01
def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图
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
        self.comd = None

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
            #print("imgcallback")
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        
            imgRGB = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # 转变成Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
            gesture_recognition_result = recognizer.recognize(mp_image)
            bg_image = np.zeros((720,1280,3), dtype=np.uint8)
            bg_image[:] = BG_COLOR
            #print(gesture_recognition_result)
            if len(gesture_recognition_result.gestures):
                gesture=gesture_recognition_result.gestures[0][0].category_name
                #print(gesture)
                cv2.putText(bg_image,str(gesture_recognition_result.gestures[0][0].category_name), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                if gesture=='Thumb_Up':
                    self.comd = 'Follow'
                elif gesture =='Thumb_Down':
                    self.comd = 'Detect'
            if len(gesture_recognition_result.hand_landmarks):
                    #print(results.multi_hand_landmarks[8])
                    for handLms in gesture_recognition_result.hand_landmarks:
                        #print(handLms)
                        for id, lm in enumerate(handLms):
                            #print(id,lm)
                            h, w, c = cv_image.shape
                            cx, cy = int(lm.x *w), int(lm.y*h)
                                #if id ==0:
                            #cv2.circle(cv_image, (cx,cy), 3, (255,0,255), cv2.FILLED)
            #cv_image=self.follow(cv_image,yolo)
            if self.comd=='Follow':
                self.x=640
                self.dis=0.7
                imgRGB = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # 转变成Image
                imgRGB=cv2.resize(imgRGB,(256,256))
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
                segmented_masks = segmenter.segment(mp_image)
                
                image_data1 = segmented_masks[0].numpy_view()
                bg_image=cv2.resize(bg_image,(256,256))
                if not np.all(image_data1==0):
                    for i in range(256):
                        for j in range(256):
                            if image_data1[i][j]==1:
                                bg_image[i][j]=MASK_COLOR
                    bg_image=cv2.resize(bg_image,(1280,720))
                    mask=cv2.cvtColor(bg_image,cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    bboxs = mask_find_bboxs(mask)
                    i = 0  # 保存的是需要画的框的个数
                    smax=0.0
                    midx=0
                    midy=0
                    for b in bboxs:
                        x0, y0 = b[0], b[1]
                        x1 = b[0] + b[2]
                        y1 = b[1] + b[3]
                        start_point, end_point = (x0, y0), (x1, y1)
                        color = (0, 0, 255)  # Red color in BGR；红色：rgb(255,0,0)
                        thickness = 2  # Line thickness
                        if (x1 - x0) >= 30:  # 连通区域太小时不画框，忽略不计
                            if i == 0:
                                mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道图
                                bg_image = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
                                i += 1
                            else:
                                bg_image = cv2.rectangle(bg_image, start_point, end_point, color, thickness)
                                i += 1
                            s=(x1-x0)*(y1-y0)
                            if smax<s:
                                smax=s
                                midx=int((x0+x1)/2)
                                midy=int((y0+y1)/2)
                    if midx!=0 and midy!=0:
                        x=midx
                        y=midy
                        cv2.circle(bg_image,(x,y),20,(0,0,255),-1)
                        depth=self.get_3d_camera_coordinate(x,y)
                        self.x=x
                        self.dis=depth
                        if x > 680:
                            print('Right')
                            #vel.linear.x=0
                            self.vel.angular.z=-(x-640)/1600-minrot
                        elif x < 600:
                            print('Left')
                            #vel.linear.x=0
                            self.vel.angular.z=-(x-640)/1600+minrot
                        else:
                            self.vel.angular.z=0

                        
                        if self.dis >1.20:
                            print('Front')
                            self.vel.linear.x=(self.dis-1.0)/2+minvel
                            #vel.angular.z=0
                        elif self.dis <0.80:
                            print('Back')
                            self.vel.linear.x=(self.dis-1.0)/2-minvel
                            #vel.angular.z=0
                        else:
                            print('Wait')
                            self.vel.linear.x=0
                            #vel.angular.z=0
            else:
                self.vel.linear.x=0
                self.vel.angular.z=0
            
            cv2.imshow("Image", bg_image)
            cv2.waitKey(1) 
            self.velpub.publish(self.vel)
        except CvBridgeError as e:
            print(e)
            return
        except ValueError as e:
            print(e)
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

    #节点初始化
    rospy.init_node('hand_gesture_ros')
    color_image_topic = '/camera/color/image_raw'
    depth_image_topic = '/camera/aligned_depth_to_color/image_raw'
    depth_info_topic = '/camera/aligned_depth_to_color/camera_info'

    listener = ImageDepthListener(color_image_topic,depth_image_topic, depth_info_topic)
    print("baseoption")
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='./realsense/scripts/yolov7/model/gesture_recognizer.task'),
        running_mode=VisionRunningMode.IMAGE)
    recognizer=GestureRecognizer.create_from_options(options)
    model_path = './realsense/scripts/yolov7/model/selfie_segmenter.tflite'
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    BG_COLOR = (0, 0, 0) # gray
    MASK_COLOR = (255, 255, 255) # white
    # Create a image segmenter instance with the image mode:
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
        )
    segmenter=ImageSegmenter.create_from_options(options) 
    
    
    rospy.spin()
    
    # rospy.Subscriber('/camera/color/image_raw', msg_Image, image_callback)
    # rospy.spin()


