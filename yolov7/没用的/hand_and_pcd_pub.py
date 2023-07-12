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
from speak_rjj import *
import threading
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from message.msg import goal
# import sys
# import os

# o_path=os.getcwd()
# sys.path.append(o_path)
# from pcd1 import *

class my_Thread(threading.Thread):
    def __init__(self,add):
        threading.Thread.__init__(self)
        self.add = add
    # 重写run()方法
    def run(self):
        speakout(self.add)

class Thread_pub(threading.Thread):
    def __init__(self,add):
        threading.Thread.__init__(self)
        self.add = add
    # 重写run()方法
    def run(self):
        speakout(self.add)

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(int(x),int(y))        # 获取该像素点对应的深度
    # print ('depth: ',dis)       # 深度单位是m
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    # print ('camera_coordinate: ',camera_coordinate)
    return dis, camera_coordinate

def yolo_hands(img,depth_frame,depth_intrin,yolo,hands,mpDraw):
    
    # if not ref:
    # raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    # pTime=0
    # global label
    # global location
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
    aimx1 = -1
    aimy1 = -1
    aimx2 = -1
    aimy2 = -1
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
                # print(locate_finger)
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
            if idmin or idmin == 0:
                # print(result_label['Label'][idmin])
                #print(idmin)
                label=result_label['Label'][idmin]
                # print(locate[idmin])
                location=locate[idmin]
    
    del depth_frame
    
    return img,label,location

# 点云聚类
def pcd_process(color_image,depth_frame,pinhole_camera_intrinsic):
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
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
    depth_image = np.asanyarray(depth_frame.get_data()) 

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
    # o3d.visualization.draw_geometries([pcd])
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

    # 使用DBSCAN算法进行聚类
    labels = np.array(others.cluster_dbscan(eps=0.01, min_points=50, print_progress=False))

    # others = others.remove_non_finite_points(True, True)
    # print(len(others.points))

    # 筛除标签为-1的点，即离群点
    others=others.select_by_index(np.where(labels==-1)[0],invert=True)
    labels=labels[labels!=-1]
    #print(len(others.points))

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
    # sizes = np.bincount(labels)
    # #print(sizes)
    # new_pcd=others
    #print(len(new_pcd.points))   

    # 可视化聚类结果
    # colors = plt.cm.jet(labels / (n_clusters if n_clusters > 1 else 2))
    # colors[:, 3] = 0.8  # 设置透明度
    # new_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([plane1, plane2, new_pcd])

    del depth_frame
    return plane1, plane2, others, labels, centers, indexs

# 退出当前程序
def breakLoop(vis):
    global breakLoopFlag
    breakLoopFlag += 1
    cv2.destroyAllWindows()
    vis.destroy_window()

    sys.exit()

def pc2_from_open3d(open3d_pc):
    """
    Convert Open3D PointCloud to ROS PointCloud2 message
    """
    ros_pc = PointCloud2()
    ros_pc.header.stamp = rospy.Time.now()
    ros_pc.header.frame_id = "camera_color_optical_frame"

    ros_pc.height = 1
    ros_pc.width = len(open3d_pc.points)
    ros_pc.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]
    ros_pc.is_bigendian = False
    ros_pc.point_step = 24
    ros_pc.row_step = ros_pc.point_step * ros_pc.width
    ros_pc.is_dense = True

    points_list = []
    for point, color in zip(open3d_pc.points, open3d_pc.colors):
        r, g, b = color
        points_list.append([point[0], -point[1], -point[2], r, g, b])

    ros_pc.data = np.asarray(points_list, np.float32).tobytes()
    return ros_pc

if __name__ == "__main__":
    mpHands = mp.solutions.hands
    label1=''
    hands = mpHands.Hands(static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    fps = 0.0
    yolo = YOLO()
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
    # get camera intrinsics 获取相机内在函数
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # 设置open3d中的针孔相机数据
    
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    # 具有自定义按键Callback功能的可视化工具。
    # geometrie_added = False
    # vis = o3d.visualization.VisualizerWithKeyCallback()
    # # 创建显示窗口
    # vis.create_window("Pointcloud", 640, 480)
    # 定义点云类
    pointcloud = o3d.geometry.PointCloud()
    # 注册按键事件，触发后执行对应函数
    # vis.register_key_callback(ord("Q"), breakLoop)
    try:
        rospy.init_node('real_time_pcd_pub')
        # 创建ROS话题发布者
        pub = rospy.Publisher('pointcloud_real', PointCloud2, queue_size=10)
        pub2 = rospy.Publisher("/msg_topic_real",goal,queue_size=10)
        p = goal()
        # 循环发布ROS消息
        rate = rospy.Rate(5) # 10hz
        while not rospy.is_shutdown():
            pTime=time.time()
            result_label=pd.DataFrame(columns=['Label','midx','midy'])

            # 清除几何中的所有元素。
            pointcloud.clear()
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            # 将RGBD对齐
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            

            # if not depth_frame or not color_frame:
            #     continue

            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics     # 获取深度参数（像素坐标系转相机坐标系会用到）
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics     # 获取相机内参

            plane1,plane2,others,labels,centers,indexs=pcd_process(color_image,depth_frame,pinhole_camera_intrinsic)
            
            img,label,location=yolo_hands(color_image,depth_frame, depth_intrin, yolo, hands, mpDraw)
            
            
            if label!='':
                # print(label,location)
                if label1!=label:
                    t=my_Thread(label)
                    t.start()
                    label1=label
            
            # plane1,plane2,others,labels,centers,indexs=pcd_process(color_image,depth_frame,pinhole_camera_intrinsic)
            pointcloud+=plane1
            pointcloud+=plane2
            pointcloud+=others
            dis = []
            idmin=-1
            center=[0,0,0]
            axis=[0,0,0]
            if not len(location)==0:
                for i in range(centers.shape[0]):
                    # print(centers[i])
                    dis.append((location[0]-centers[i][0])**2+(location[1]+centers[i][1])**2+(location[2]+centers[i][2])**2)
                temp={'dist':dis}
                distant=pd.DataFrame(temp)
                # print(distant)
                idmin=distant.idxmin()[0]
            
            if len(indexs)==0:
                continue
            elif idmin>=0:
                # print(idmin)
                # print(centers[idmin])
                selected_index=idmin
                center=centers[idmin]
                # 计算包围框
                selected_pcd=others.select_by_index(np.where(labels==selected_index)[0])
                # abox = selected_pcd.get_axis_aligned_bounding_box()
                # abox.color = (1,0,0)
                # obox = selected_pcd.get_oriented_bounding_box()
                # obox.color = (1,1,1)
                #o3d.visualization.draw_geometries([plane1,plane2,selected_pcd, abox, obox])
                pointcloud+=selected_pcd
                # 将点云数据转换为numpy数组
                points = np.asarray(selected_pcd.points)

                # 计算主轴方向
                cov = np.cov(points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                axis = eigenvectors[:, np.argmax(eigenvalues)]

            p.x = center[0]
            p.y = -center[1]
            p.z = -center[2]

            p.roll = axis[0]
            p.yaw = -axis[1]
            p.pitch = -axis[2]
            ros_cloud=pc2_from_open3d(pointcloud)
            pub.publish(ros_cloud)
            if(math.fabs(p.x**2+p.y**2+p.z**2)>0.1):
                pub2.publish(p)
                rospy.loginfo("x:%f, y:%f, z:%f, roll:%f, yaw:%f, pitch:%f", \
                    p.x, p.y, p.z, p.roll, p.yaw, p.pitch)
            # rate.sleep()
            # if not geometrie_added:
            #     # 第一次将几何体添加到场景并创建相应的着色器的功能，之后只需要更行的就好
            #     vis.add_geometry(pointcloud)
            #     # vis.add_geometry(abox)
            #     # previous_abox_handle = abox
            #     geometrie_added = True
            #     # print("add")
            # else:
                
            #     vis.update_geometry(pointcloud)

                # # 更新 abox
                # abox = selected_pcd.get_axis_aligned_bounding_box()
                # abox.color = (0,0,0)
                # vis.remove_geometry(previous_abox_handle)
                # vis.add_geometry(abox)
                # previous_abox_handle = abox
            # 轮询事件的功能
            # vis.poll_events()
            # 通知渲染需要更新的功能
            # vis.update_renderer()

            cTime=time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.imshow("Image", img)
           
            
            if cv2.waitKey(1) == ord("q"):
                break
    except rospy.ROSInterruptException:
        pipeline.stop()
        pass
            