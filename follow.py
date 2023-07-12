import rospy
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image
from yolo import YOLO
import pandas as pd

minvel=0.01
minrot=0.1
velpub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
rospy.init_node('follow')
vel=Twist()

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
yolo = YOLO()
pipeline.start(config)
align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)  
while True:
    result_label=pd.DataFrame(columns=['Label','midx','midy'])
    frames = pipeline.wait_for_frames()
    
    aligned_frames = align.process(frames)      # 获取对齐帧，将深度框与颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()       
    if not depth_frame or not color_frame:
        continue
        # Convert images to numpy arrays 把图像转换为numpy data
    depth_image = np.asanyarray(depth_frame.get_data())
    #color_image = np.asanyarray(color_frame.get_data())
    img = np.asanyarray(color_frame.get_data())

    #img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #imgRGBnp=np.array(img)
    img  = Image.fromarray(np.uint8(imgRGB))
            # 进行检测
    img,result_label=yolo.detect_image(img)
    #print(result_label)
    img = np.array(img)
            # RGBtoBGR满足opencv显示格式
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if not result_label.empty:
        #if 'fan' in result_label['Label']:
        re=result_label.loc[result_label['Label'] == 'fan' ].index.tolist()
        if not len(re)==0:
            r=re[0]
            x=result_label['midx'][r]
            y=result_label['midy'][r]
            if x > 335:
                print('Right')
                #vel.linear.x=0
                vel.angular.z=-(x-320)/800-minrot
            elif x < 305:
                print('Left')
                #vel.linear.x=0
                vel.angular.z=-(x-320)/800+minrot
            else:
                vel.angular.z=0

            dis = depth_frame.get_distance(int(x), int(y))
            if dis >0.70:
                print('Front')
                vel.linear.x=(dis-0.6)/2+minvel
                #vel.angular.z=0
            elif dis <0.50:
                print('Back')
                vel.linear.x=(dis-0.6)/2-minvel
                #vel.angular.z=0
            else:
                print('Wait')
                vel.linear.x=0
                #vel.angular.z=0
    else:
        vel.linear.x=0
        vel.angular.z=0
    velpub.publish(vel)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break