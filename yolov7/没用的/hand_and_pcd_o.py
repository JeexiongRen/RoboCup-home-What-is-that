import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np
from PIL import Image
from yolo import YOLO
import pandas as pd
import math
import open3d as o3d 
import sys

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(int(x), int(y))        # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

def yolo_hands(frames,yolo,align,hands,mpDraw,pTime):
    fps = 0.0
    # if not ref:
    # raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    # pTime=0
    global label
    global location
    # 读取某一帧
    # 格式转变，BGRtoRGB
    # img = cv2.flip(img, 1)
    aligned_frames = align.process(frames)      # 获取对齐帧，将深度框与颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参

    if not depth_frame or not color_frame:
        return
    # Convert images to numpy arrays 把图像转换为numpy data
    depth_image = np.asanyarray(depth_frame.get_data())
    img = np.asanyarray(color_frame.get_data())
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
                # print(result_label['Label'][idmin])
                label=result_label['Label'][idmin]
                # print(locate[idmin])
                location=locate[idmin]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    del depth_frame
    del color_frame
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img

def pcd_process(frames,pinhole_camera_intrinsic):
    
    # 将RGBD对齐
    aligned_frames = align.process(frames)

    # 获取RGB图像，并转为np格式数据
    color_frame = aligned_frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    # 获取深度信息
    depth_frame = aligned_frames.get_depth_frame()
    depth_image = np.asanyarray(depth_frame.get_data()) 

    # 利用中值核进行滤波
    depth_frame = rs.decimation_filter(1).process(depth_frame)
    # 从深度表示转换为视差表示，反之亦然
    depth_frame = rs.disparity_transform(True).process(depth_frame)
    # 空间滤镜通过使用alpha和delta设置计算帧来平滑图像。
    depth_frame = rs.spatial_filter().process(depth_frame)
    # 时间滤镜通过使用alpha和delta设置计算多个帧来平滑图像。
    depth_frame = rs.temporal_filter().process(depth_frame)
    # 从视差表示转换为深度表示
    depth_frame = rs.disparity_transform(False).process(depth_frame)
    # depth_frame = rs.hole_filling_filter().process(depth_frame)

    # 将深度图转化为RGB准备显示
    # depth_color_frame = rs.colorizer().colorize(depth_frame)
    # depth_color_image = np.asanyarray(depth_color_frame.get_data())

    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # # cv2.imwrite("depth_colormap.jpg",depth_colormap)           

    # color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR) # 由于cv中的显示是BGR所以需要再转换出一个

    # 在cv2中显示RGB-D图
    # cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('color image', color_image1)
    # cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('depth image', depth_color_image)

    # 图像类存储具有可自定义的宽度，高度，通道数和每个通道字节数的图像。
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(color_image)

    # 从颜色和深度图像制作RGBDImage的功能
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,depth_scale=4000.0,
                                                                    depth_trunc=5.0,convert_rgb_to_intensity=False)
    # 通过RGB-D图像和相机创建点云，并导入摄像机的固有参数
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
    # 将转换（4x4矩阵）应用于几何坐标。
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # pcd = voxel_down_sample(pcd, voxel_size = 0.003)

    #pointcloud += pcd  # 添加点云

    # 第一次提取平面
    plane_1, inliers_1 = pcd.segment_plane(distance_threshold=0.02, ransac_n=4, num_iterations=1200)

    # 剔除第一个平面的点云
    plane1 = pcd.select_by_index(inliers_1)
    pcd = pcd.select_by_index(inliers_1,invert=True)

    # 第二次提取平面
    plane_2, inliers_2 = pcd.segment_plane(distance_threshold=0.02, ransac_n=4, num_iterations=1200)
    plane2 = pcd.select_by_index(inliers_2)

    # 剩余点云
    others = pcd.select_by_index(inliers_2,invert=True)

    # 统一设置平面点云的颜色
    plane1.paint_uniform_color([1, 0, 0])
    plane2.paint_uniform_color([0, 0, 1])

    # 可视化点云
    #o3d.visualization.draw_geometries([plane1, plane2, others])

    # 使用DBSCAN算法进行聚类
    labels = np.array(others.cluster_dbscan(eps=0.01, min_points=50, print_progress=False))

    # others = others.remove_non_finite_points(True, True)
    # print(len(others.points))

    # 筛除标签为-1的点，即离群点
    others=others.select_by_index(np.where(labels==-1)[0],invert=True)
    labels=labels[labels!=-1]
    #print(len(others.points))

    # 统计聚类数量
    n_clusters = labels.max() + 1
    #print(f"Number of clusters: {n_clusters}")

    # 统计每个聚类中的点的数量
    sizes = np.bincount(labels)
    #print(sizes)

    #定义阈值，保留聚类中点数量大于等于该值的点
    threshold = 1000
    centers = np.zeros((sizes.size, 3))
    indexs=[]
    for i in range(np.max(labels) + 1):
        # 对每个部分进行处理
        mask = labels == i
        cluster_pcd = others.select_by_index(np.where(mask)[0])
        
        if len(cluster_pcd.points) < threshold:
            others=others.select_by_index(np.where(mask)[0],invert=True)
            labels=labels[labels!=i]
            centers[i]=[0,0,0]
        else:
            # 计算聚类中心
            centers[i] = np.mean(np.asarray(others.points)[labels==i], axis=0)

            # 通过平面模型计算部分是否在桌面上
            distance = np.dot(centers[i] - plane_2[:3], plane_2[:3])
            #print(distance)
            if(math.fabs(distance+1.5)>0.1):
                others=others.select_by_index(np.where(mask)[0],invert=True)
                labels=labels[labels!=i]
                centers[i]=[0,0,0]
            else:
                indexs.append(i)
            #continue
        
    # print(centers)
    #print(indexs)
    sizes = np.bincount(labels)
    #print(sizes)
    new_pcd=others
    #print(len(new_pcd.points))   

    #new_pcd, _ = others.remove_statistical_outlier(nb_neighbors=2000, std_ratio=0.5)
    #cl, ind = others.remove_radius_outlier(nb_points=100, radius=0.1)
    #display_inlier_outlier(voxel_down_pcd, ind)

    # 可视化聚类结果
    # colors = plt.cm.jet(labels / (n_clusters if n_clusters > 1 else 2))
    # colors[:, 3] = 0.8  # 设置透明度
    # new_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([plane1, plane2, new_pcd])

    # selected_index=indexs[0]
    # # 计算包围框
    # selected_pcd=new_pcd.select_by_index(np.where(labels==selected_index)[0])
    # abox = selected_pcd.get_axis_aligned_bounding_box()
    # abox.color = (1,0,0)
    # obox = selected_pcd.get_oriented_bounding_box()
    # obox.color = (1,1,1)
    # #o3d.visualization.draw_geometries([plane1,plane2,selected_pcd, abox, obox])
    del depth_frame
    del color_frame
    return plane1, plane2, others, labels, centers, indexs

# 退出当前程序
def breakLoop(vis):
    global breakLoopFlag
    breakLoopFlag += 1
    cv2.destroyAllWindows()
    vis.destroy_window()

    sys.exit()

if __name__ == "__main__":
    mpHands = mp.solutions.hands

    hands = mpHands.Hands(static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    yolo = YOLO()
    # pTime = 0
    # cTime = 0
    label=''
    location=[]
    # 相机RGB和深度图对齐
    align = rs.align(rs.stream.color)

    # 配置视频流
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 创建一个上下文对象。该对象拥有所有连接的Realsense设备的句柄
    pipeline = rs.pipeline()

    # 开启视频流
    profile = pipeline.start(config)    

    # 打开相机并获取相机的深度传感器
    depth_sensor = profile.get_device().first_depth_sensor()

    # 设置深度图像的可接受范围
    depth_scale = depth_sensor.get_depth_scale()

    # get camera intrinsics 获取相机内在函数
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # 设置open3d中的针孔相机数据
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

    # 具有自定义按键Callback功能的可视化工具。
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    # 创建显示窗口
    vis.create_window("Pointcloud", 640, 480)
    # 定义点云类
    pointcloud = o3d.geometry.PointCloud()

    # 注册按键事件，触发后执行对应函数
    vis.register_key_callback(ord("Q"), breakLoop)
    while True:
            result_label=pd.DataFrame(columns=['Label','midx','midy'])

            # 清除几何中的所有元素。
            pointcloud.clear()
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            img=yolo_hands(frames, yolo, align, hands, mpDraw, time.time())

            if label!='':
                print(label,location)
                label=''
            
            cv2.imshow("Image", img)
            if cv2.waitKey(1) == ord("q"):
                break

            plane1,plane2,others,labels,centers,indexs=pcd_process(frames,pinhole_camera_intrinsic)
            selected_index=indexs[0]
            # 计算包围框
            selected_pcd=others.select_by_index(np.where(labels==selected_index)[0])
            abox = selected_pcd.get_axis_aligned_bounding_box()
            abox.color = (1,0,0)
            obox = selected_pcd.get_oriented_bounding_box()
            obox.color = (1,1,1)
            #o3d.visualization.draw_geometries([plane1,plane2,selected_pcd, abox, obox])

            pointcloud+=plane1
            pointcloud+=plane2
            pointcloud+=selected_pcd
            
            if not geometrie_added:
                # 第一次将几何体添加到场景并创建相应的着色器的功能，之后只需要更行的就好
                vis.add_geometry(pointcloud)
                vis.add_geometry(abox)
                previous_abox_handle = abox
                geometrie_added = True
                # print("add")
            else:
                
                vis.update_geometry(pointcloud)

                # 更新 abox
                abox = selected_pcd.get_axis_aligned_bounding_box()
                abox.color = (0,0,0)
                vis.remove_geometry(previous_abox_handle)
                vis.add_geometry(abox)
                previous_abox_handle = abox
            
            
            # 轮询事件的功能
            vis.poll_events()
            # 通知渲染需要更新的功能
            vis.update_renderer()
            
            