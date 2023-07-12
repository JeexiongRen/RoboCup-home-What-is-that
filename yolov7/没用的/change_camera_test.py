import pyrealsense2 as rs
import cv2
import numpy as np


def change_camera(camera_using,pipeline):
    if camera_using=='Cam1':
        pipeline.stop()
        pipeline = rs.pipeline()
        pipeline.start(config0)
        camera_using='Cam0'
        
    elif camera_using=='Cam0':
        pipeline.stop()
        pipeline = rs.pipeline()
        pipeline.start(config1)
        camera_using='Cam1'
    return camera_using,pipeline


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



pipeline.start(config1)
align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)  

camera_using='Cam1'
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames1 = align.process(frames)
        color_frame1 = aligned_frames1.get_color_frame()
        depth_frame1 = aligned_frames1.get_depth_frame()
        color_image1 = np.asanyarray(color_frame1.get_data())
        depth_image1 = np.asanyarray(depth_frame1.get_data())
        depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image1, alpha=0.03), cv2.COLORMAP_JET)
        images1 = np.hstack((color_image1, depth_colormap1))
        cv2.imshow('RealSense1', images1)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('c'):
            camera_using,pipeline=change_camera(camera_using,pipeline)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
        pipeline.stop()