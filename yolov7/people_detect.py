import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pyrealsense2 as rs

# ros相关内容
import rospy
from geometry_msgs.msg import Twist

minvel=0.01
minrot=0.1
velpub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
rospy.init_node('follow')
vel=Twist()
# end


def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图
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
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)  
comd=None
while True:
    
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()       
    if not depth_frame or not color_frame:
        continue
        # Convert images to numpy arrays 把图像转换为numpy data
    depth_image = np.asanyarray(depth_frame.get_data())
    #color_image = np.asanyarray(color_frame.get_data())
    img = np.asanyarray(color_frame.get_data())

    #img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_image=cv2.cvtColor(imgRGB,cv2.COLOR_RGB2BGR)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    gesture_recognition_result = recognizer.recognize(mp_image)
    #print(gesture_recognition_result)
    if len(gesture_recognition_result.gestures):
        gesture=gesture_recognition_result.gestures[0][0].category_name
        cv2.putText(img,str(gesture_recognition_result.gestures[0][0].category_name), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        if gesture=='Thumb_Up':
            comd='Follow'
        if gesture=='Thumb_Down':
            comd='Detect'
    # Perform image segmentation on the provided single image.
# The image segmenter must be created with the image mode.
        result_image=cv2.putText(imgRGB,str(gesture_recognition_result.gestures[0][0].category_name), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    if comd=='Follow':
        imgRGB=cv2.resize(imgRGB,(256,256))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        segmented_masks = segmenter.segment(mp_image)
        image_data = mp_image.numpy_view()
        image_data1 = segmented_masks[0].numpy_view()
        
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        for i in range(256):
            for j in range(256):
                if image_data1[i][j]==1:
                    bg_image[i][j]=MASK_COLOR
        #print((miny,maxy))
        #cv2.circle(bg_image,(int((miny+maxy)/2),128),2,(0,0,255),-1)
        
        result_image=cv2.resize(bg_image,(640,480))
        
        mask=cv2.cvtColor(result_image,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        bboxs = mask_find_bboxs(mask)
        i = 0  # 保存的是需要画的框的个数
        point_to_follow=(0,0) # 要跟的点 
        smax=0 # 最大面积
        for b in bboxs:
            x0, y0 = b[0], b[1]
            x1 = b[0] + b[2]
            y1 = b[1] + b[3]
            start_point, end_point = (x0, y0), (x1, y1)
            s=(x1-x0)*(y1-y0)
            if s>smax:
                smax=s
                point_to_follow=((x0+x1)/2,(y0+y1)/2)
            color = (0, 0, 255)  # Red color in BGR；红色：rgb(255,0,0)
            thickness = 2  # Line thickness
            if (x1 - x0) >= 15:  # 连通区域太小时不画框，忽略不计
                if i == 0:
                    mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道图
                    result_image = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
                    i += 1
                else:
                    result_image = cv2.rectangle(result_image, start_point, end_point, color, thickness)
                    i += 1
        result_image=cv2.putText(result_image,str(comd), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        x=int(point_to_follow[0])
        y=int(point_to_follow[1])
        
        if x!=0 and y!=0:
            result_image=cv2.circle(result_image,(x,y),20,(0,0,255),-1)
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

            dis = depth_frame.get_distance(x, y)
            #print(dis)

            if dis >1.20:
                print('Front')
                vel.linear.x=(dis-1.)/2+minvel
                #vel.angular.z=0
            elif dis <0.8:
                print('Back')
                vel.linear.x=(dis-1.0)/2-minvel
                #vel.angular.z=0
            else:
                print('Wait')
                vel.linear.x=0
                #vel.angular.z=0
        else:
            vel.linear.x=0
            vel.angular.z=0
    else:
        vel.linear.x=0
        vel.angular.z=0
    velpub.publish(vel)

    cv2.imshow('Result',result_image)
    cv2.waitKey(1)
    

    
