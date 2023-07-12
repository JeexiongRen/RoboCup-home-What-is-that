import mediapipe as mp
import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
#cap = cv2.VideoCapture(0)
pipeline.start(config)
align_to = rs.stream.color      # align_to 是计划对齐深度帧的流类型
align = rs.align(align_to)      # rs.align 执行深度帧与其他帧的对齐
mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='./model/gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)
recognizer=GestureRecognizer.create_from_options(options)
  # The detector is initialized. Use it here.
  # ...
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
    #imgRGBnp=np.array(img)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    gesture_recognition_result = recognizer.recognize(mp_image)
    #print(gesture_recognition_result)
    if len(gesture_recognition_result.gestures):
        print(gesture_recognition_result.gestures[0][0].category_name)
        cv2.putText(img,str(gesture_recognition_result.gestures[0][0].category_name), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    if len(gesture_recognition_result.hand_landmarks):
            #print(results.multi_hand_landmarks[8])
            for handLms in gesture_recognition_result.hand_landmarks:
                #print(handLms)
                for id, lm in enumerate(handLms):
                    #print(id,lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                        #if id ==0:
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    #imgcv=cv2.cvtColor(imgRGB,cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
