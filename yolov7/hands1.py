import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from PIL import Image
from yolo import YOLO
import pandas as pd
import math

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(int(x), int(y))        # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

def yolo_hands(img,color_frame,depth_frame,depth_intrin,yolo,align,hands,mpDraw,pTime):
    fps = 0.0
    # if not ref:
    # raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    # pTime=0
    
    # 读取某一帧
    # 格式转变，BGRtoRGB
    # img = cv2.flip(img, 1)
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
    results = hands.process(imgRGB)
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
    aimx1 = -1.0
    aimy1 = -1.0
    aimx2 = -1.0
    aimy2 = -1.0
    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks[8])
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
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
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    # print(result_label)
    dist = []
    locate = []
    if not result_label.empty:
        if aimx1 != -1.0 and aimy1 != -1.0 and aimx2 != -1.0 and aimy2 != -1.0:
            for row in result_label.itertuples():
                dis_finger1, locate_finger1 = get_3d_camera_coordinate([aimx1, aimy1], depth_frame, depth_intrin)
                dis_finger2, locate_finger2 = get_3d_camera_coordinate([aimx2, aimy2], depth_frame, depth_intrin)
                dis_obj, locate_obj = get_3d_camera_coordinate([getattr(row, 'midx'), getattr(row, 'midy')],
                                                               depth_frame, depth_intrin)
                # print(locate_finger)
                # print(type(locate_finger))
                v1 = np.array([locate_finger2[0] - locate_finger1[0], locate_finger2[1] - locate_finger1[1],
                               locate_finger2[2] - locate_finger1[2]])
                v2 = np.array([locate_finger1[0] - locate_obj[0], locate_finger1[1] - locate_obj[1],
                               locate_finger1[2] - locate_obj[2]])
                dist.append(np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v1))
                # dist.append(math.sqrt((locate_finger1[0]-locate_obj[0])**2+(locate_finger1[1]-locate_obj[1])**2+(locate_finger1[2]-locate_obj[2])**2))
                # print(dist)
                locate.append(locate_obj)
            result_label.insert(1, 'dist', dist)
            # print(result_label)
            idmin = pd.DataFrame(result_label['dist']).idxmin()[0]
            # print(idmin)
            # print(type(idmin))
            if idmin or idmin == 0:
                print(result_label['Label'][idmin])
                print(locate[idmin])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    del depth_frame
    del color_frame
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img


mpHands = mp.solutions.hands

hands = mpHands.Hands(static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

yolo = YOLO()
# pTime = 0
# cTime = 0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# Start streaming
pipeline.start(config)

align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)      # rs.align 执行深度帧与其他帧的对齐


while True:
        result_label=pd.DataFrame(columns=['Label','midx','midy'])
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)      # 获取对齐帧，将深度框与颜色框对齐
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参
       
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays 把图像转换为numpy data
        depth_image = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        #print(color_image.shape)
        #print(type(depth_frame))
        #print(depth_frame.get_distance(240,320))
        #cap=color_image
        img=yolo_hands(color_img, color_frame, depth_frame, depth_intrin, yolo, align, hands, mpDraw, time.time())
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord("q"):
            break
        