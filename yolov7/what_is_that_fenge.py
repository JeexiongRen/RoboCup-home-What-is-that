#mediapipe相关内容
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pyrealsense2 as rs

#识别相关内容
from PIL import Image
from yolo_P import YOLO
import pandas as pd
import math
from speak_rjj import *
import time
import threading
#from concurrent.future import ThreadPoolExecutor
# ros相关内容
import rospy
from geometry_msgs.msg import Twist
from fpdf import FPDF

minvel=0.01
minrot=0.1
velpub=rospy.Publisher('cmd_vel',Twist,queue_size=10)
rospy.init_node('follow')
vel=Twist()
# end

class my_Thread(threading.Thread):
    def __init__(self,add):
        threading.Thread.__init__(self)
        self.add = add
    # 重写run()方法
    def run(self):
        speakout(self.add)

class Process():
    def __init__(self):
        self.result_hand=None
        self.result_people=None
    def deal_hand(self,result:mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.result_hand=result
    def deal_people(self,result, output_image: mp.Image, timestamp_ms: int):
        self.result_people=result

def mask_find_bboxs(mask):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)  # connectivity参数的默认值为8
    stats = stats[stats[:, 4].argsort()]
    return stats[:-1]  # 排除最外层的连通图

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    print((x,y))
    if x>639 or y>479 or x<0 or y<0:
        dis = -1
        camera_coordinate=(0,0,0)
        return dis,camera_coordinate
        
    dis = aligned_depth_frame.get_distance(int(x),int(y))        # 获取该像素点对应的深度
    #print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

def change_camera(camera_using,pipeline,comd):
    if camera_using=='Cam1' and comd=='Detect':
        pipeline.stop()
        pipeline = rs.pipeline()
        pipeline.start(config0)
        camera_using='Cam0'
        
    elif camera_using=='Cam0' and comd=='Stop':
        pipeline.stop()
        pipeline = rs.pipeline()
        pipeline.start(config1)
        camera_using='Cam1'
    return camera_using,pipeline
    
def yolo_hands(img,depth_frame,depth_intrin,yolo,hands,mpDraw,process):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 转变成Image
    img = Image.fromarray(np.uint8(img))
    # 进行检测
    img, result_label = yolo.detect_image(img)
    img = np.array(img)
    # RGBtoBGR满足opencv显示格式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first) 在深度图上用颜色渲染
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = process.result_hand
    # print(results.multi_hand_landmarks)
    landmark_list = []
    # # 用来存储手掌范围的矩形坐标
    # paw_x_list = []
    # paw_y_list = []
    # for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
    #     landmark_list.append([landmark_id, finger_axis.x, finger_axis.y,finger_axis.z])
    #     paw_x_list.append(finger_axis.x)
    #     paw_y_list.append(finger_axis.y)
    # if landmark_list:
    #     # 比例缩放到像素
    #     ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
    #     ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)
    aimx1 = -1
    aimy1 = -1
    aimx2 = -1
    aimy2 = -1
    if len(results.hand_landmarks):
        # print(results.multi_hand_landmarks[8])
        for handid in range(len(results.hand_landmarks)):
            if results.handedness[handid][0].category_name=='Left':
        #for handLms in results.landmarks:
                for id, lm in enumerate(results.hand_landmarks[handid]):
                    # print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # if id ==0:
                    if id == 8:
                        aimx1 = cx
                        aimy1 = cy
                    elif id == 7:
                        aimx2 = cx
                        aimy2 = cy
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                #mpDraw.draw_landmarks(img, results.hand_landmarks[handid], mpHands.HAND_CONNECTIONS)
    # print(result_label)
    dist = []
    locate = []
    label=''
    location=[]
    if not result_label.empty:
        if aimx1 != -1 and aimy1 != -1 and aimx2 != -1 and aimy2 != -1:
            for row in result_label.itertuples():
                #print(aimx1,aimy1,aimx2,aimy2)
                dis_finger1, locate_finger1 = get_3d_camera_coordinate([aimx1, aimy1], depth_frame, depth_intrin)
                dis_finger2, locate_finger2 = get_3d_camera_coordinate([aimx2, aimy2], depth_frame, depth_intrin)
                dis_obj, locate_obj = get_3d_camera_coordinate([getattr(row, 'midx'), getattr(row, 'midy')],
                                                               depth_frame, depth_intrin)
                #print(dis_obj)
                if dis_finger1==-1 or dis_finger2==-1 or dis_obj==-1:
                    return True,0,0,0
                # print(type(locate_finger))
                #print(locate_finger1,locate_finger2,locate_obj)
                v1 = np.array([locate_finger2[0] - locate_finger1[0], locate_finger2[1] - locate_finger1[1],
                               locate_finger2[2] - locate_finger1[2]])
                v2 = np.array([locate_finger1[0] - locate_obj[0], locate_finger1[1] - locate_obj[1],
                               locate_finger1[2] - locate_obj[2]])
                #print(v1,v2)
                dist.append(np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v1))
                # dist.append(math.sqrt((locate_finger1[0]-locate_obj[0])**2+(locate_finger1[1]-locate_obj[1])**2+(locate_finger1[2]-locate_obj[2])**2))
                # print(dist)
                locate.append(locate_obj)
            result_label.insert(1, 'dist', dist)
            # print(result_label)
            idmin = pd.DataFrame(result_label['dist']).idxmin()[0]
            # print(idmin)
            # print(type(idmin))
            if not np.isnan(idmin):
                # print(result_label['Label'][idmin])
                #print(idmin)
                #print(np.isnan(idmin))
                label=result_label['Label'][idmin]
                # print(locate[idmin])
                location=locate[idmin]
    
    del depth_frame
    
    return False,img,label,location

if __name__ == "__main__":

    #记录pdf初始化
    pdf  = FPDF()
    pdf.add_page()

    #设置字体的大小和字体
    pdf.set_font('Arial', size=15)

    #回调函数初始化
    process=Process()

    #手势识别初始化
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path='/home/zzn/robocup_ws/src/realsense/scripts/yolov7/model/gesture_recognizer.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process.deal_hand,
        num_hands=2)
    recognizer=GestureRecognizer.create_from_options(options)

    #人体检测初始化
    model_path = '/home/zzn/robocup_ws/src/realsense/scripts/yolov7/model/selfie_segmenter.tflite'
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    BG_COLOR = (0, 0, 0) # gray
    MASK_COLOR = (255, 255, 255) # white
    # Create a image segmenter instance with the image mode:
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process.deal_people
        )
    segmenter=ImageSegmenter.create_from_options(options) 

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

    #特征字典
    attr_dic={'Orange juice':'Orange liquid.','Cola':'Black liquid.','Cookie':'Black box.','Biscuit':'Light blue box.','Shampoo':'Dark blue.','Chip':'Yellow box.','Bread':'Plastic bag.','Handwash':'White bottle.','Sprite':'Green cap.','Water':'Red and white label.','Dishsoap':'Orange and white bottle.','Lays':'Green jar.'}

    pipeline = rs.pipeline()
    config0 = rs.config()
    config0.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config0.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config0.enable_device(connect_device[0])

    config1 = rs.config()
    config1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config1.enable_device(connect_device[1])

    pipeline.start(config1)
    align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)

    comd=None
    camera_using='Cam1'

    #yolo及手部关键点识别初始化
    mpHands = mp.solutions.hands
    label1=''
    hands = mpHands.Hands(static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    fps = 0.0
    yolo = YOLO()
    gesture_times=0
    gesture_old=None
    label_times=0
    label_old=None
    line=1
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
        recognizer.recognize_async(mp_image, int(time.time()*1000))
        gesture_recognition_result = process.result_hand
        
        if gesture_recognition_result:

            if len(gesture_recognition_result.gestures):
                for handid in range(len(gesture_recognition_result.gestures)):
                    if gesture_recognition_result.handedness[handid][0].category_name=='Right':#mediapipe左右手弄反，不知道因为啥
                        gesture=gesture_recognition_result.gestures[handid][0].category_name
                        cv2.putText(img,str(gesture_recognition_result.gestures[handid][0].category_name), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                        #print (gesture)
                        #手势识别部分，确认当前任务
                        if gesture_old==gesture:
                            gesture_times+=1
                        else :
                            gesture_times=0
                        gesture_old=gesture
                        if gesture_times>=5:
                            gesture_times
                            if gesture=='ILoveYou' and (comd==None or comd=='Stop'):
                                #大拇指向上手势，开始跟随
                                comd='Follow'
                                p=my_Thread('I have seen your I Love You gesture, begin to follow.')
                                p.start()
                                pdf.cell(200, 10, txt='I have seen your ILoveYou gesture, begin to follow.'+ '     '+time.asctime(time.localtime(time.time())), ln=line, align='C')
                                line+=1
                            if gesture=='Thumb_Down' and comd=='Follow':
                                #大拇指向下手势，结束跟随
                                comd='Stop'
                                p=my_Thread('I have seen your thumb down gesture, stop to follow.')
                                p.start()
                                pdf.cell(200, 10, txt='I have seen your thumb down gesture, stop to follow.'+ '     '+time.asctime(time.localtime(time.time())), ln=line, align='C')
                                line+=1
                            if gesture=='Victory' and comd=='Stop':
                                #胜利手势，开始识别
                                comd='Detect'
                                p=my_Thread('I have seen your victory gesture, begin to detect.')
                                p.start()
                                pdf.cell(200, 10, txt='I have seen your victory gesture, begin to detect.'+ '     '+ time.asctime(time.localtime(time.time())), ln=line, align='C')
                                line+=1
                                label_times=0
                                label_old=None
                                label1=None
                                #print(comd)
                            if gesture=='Thumb_Up' and comd=='Detect':
                                comd='Stop'
                                p=my_Thread('I have seen your thumb up gesture, stop to detect.')
                                p.start()
                                pdf.cell(200, 10, txt='I have seen your thumb up gesture, stop to detect.'+ '     '+ time.asctime(time.localtime(time.time())), ln=line, align='C')
                                line+=1
                            if gesture=='Open_Palm' and comd=='Stop':
                                p=my_Thread('I have seen your open palm gesture, exit.')
                                p.start()
                                pdf.cell(200, 10, txt='I have seen your open palm gesture, exit.'+ '     '+ time.asctime(time.localtime(time.time())), ln=line, align='C')
                                line+=1
                                pdf.output('/home/zzn/robocup_ws/src/realsense/scripts/yolov7/output.pdf')
                                #time.sleep(5)
                                exit()
                            
                    # Perform image segmentation on the provided single image.
                # The image segmenter must be created with the image mode.
                        result_image=cv2.putText(imgRGB,str(gesture_recognition_result.gestures[handid][0].category_name), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        
        
        #检测人体并跟随
        if comd=='Follow':
            imgRGB=cv2.resize(imgRGB,(256,256))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
            segmenter.segment_async(mp_image, int(time.time()*1000))
            segmented_masks = process.result_people
            if segmented_masks:
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
                        vel.angular.z=-(x-320)/600-minrot
                    elif x < 305:
                        print('Left')
                        #vel.linear.x=0
                        vel.angular.z=-(x-320)/600+minrot
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
        #开始识别
        if comd=='Detect':
            camera_using,pipeline=change_camera(camera_using,pipeline,comd)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()       
            if not depth_frame or not color_frame:
                continue
                # Convert images to numpy arrays 把图像转换为numpy data
            depth_image = np.asanyarray(depth_frame.get_data())
            #color_image = np.asanyarray(color_frame.get_data())
            img = np.asanyarray(color_frame.get_data())
            pTime=time.time()
            result_label=pd.DataFrame(columns=['Label','midx','midy'])
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参
            erro,result_image,label,location=yolo_hands(img,depth_frame, depth_intrin, yolo, hands, mpDraw,process)
            if erro:
                continue
            cTime=time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            cv2.putText(result_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if label!='':
                if label_old==label:
                    label_times+=1
                else:
                    label_times=0
                    label_old=label
                if label_times>=5:
                    label_times=0
                    if label1!=label:
                        t=my_Thread(label+'.'+attr_dic[label])

                        t.start()
                        pdf.cell(200, 10, txt=label+'.'+attr_dic[label]+ '     '+ time.asctime(time.localtime(time.time())), ln=line, align='C')
                        line+=1
                        label1=label
        if comd=='Stop':
            camera_using,pipeline=change_camera(camera_using,pipeline,comd)
        cv2.imshow('Result',result_image)
        cv2.waitKey(1)
        

    
