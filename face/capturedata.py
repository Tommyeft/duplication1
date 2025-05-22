import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# 加载 YOLO 模型
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml").load(
    "C:\\Users\\16481\\Desktop\\duplication1\\best.pt")

# 设置视频路径（0 表示摄像头，或者填写视频文件路径）
video_path = 0  # 使用摄像头
video_path = 'C:\\Users\\16481\Desktop\duplication1\\UI\【niko的颜值鉴赏】.mp4'  # 替换为你的视频文件路径

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("无法打开视频")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("视频结束或无法读取帧")
        break

    # 将 OpenCV 图像转换为 PIL 格式
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 运行 YOLO 进行人脸检测
    result = model(image)

    # 处理检测结果
    for box in result[0].boxes:
        if box.cls == 0:  # 假设类别 0 代表人脸
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])  # 获取边框坐标
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框

    # 显示结果
    cv2.imshow("YOLO Face Detection", frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
