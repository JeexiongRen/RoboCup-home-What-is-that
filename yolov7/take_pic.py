import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import os
# 拍数据集用
# written by zzn
save_path = '/home/zzn/dataset/'
name = 'capigu3/capigu3_'

if __name__ == "__main__":
    num = 0
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
    align = rs.align(align_to)      # rs.align 执行深度帧与其他帧的对齐


    while True:
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
        color_image = np.asanyarray(color_frame.get_data())
        
        path = save_path + name 
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imshow("Image", color_image)
        cv2.imwrite(path+ str(num) + '.jpg',color_image)
        num = num + 1
        time.sleep(0.2)
        
        if cv2.waitKey(1) == ord("q"):
            break
    pipeline.stop()
            