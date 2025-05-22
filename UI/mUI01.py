import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QListWidget, QFileDialog, QMessageBox, QInputDialog)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import face_recognition
import cv2
from ultralytics import YOLO
from PIL import Image
import time
from queue import Queue
import threading
from concurrent.futures import ThreadPoolExecutor


def match_face(database_path, image_path):
    """从数据库中查找匹配的人脸"""
    try:
        # 读取数据库
        face_data = np.load(database_path, allow_pickle=True).item()
        known_encodings = face_data.get("encodings", [])
        known_ids = face_data.get("ids", [])

        if not known_encodings or not known_ids:
            return "数据库为空"

        # 读取并编码新图像
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        if not unknown_encoding:
            return "未检测到人脸"

        unknown_encoding = unknown_encoding[0]

        # 进行人脸比对
        results = face_recognition.compare_faces(known_encodings, unknown_encoding)

        if True in results:
            matched_index = results.index(True)
            return known_ids[matched_index]  # 返回匹配到的 ID

        return "未匹配到已知人脸"

    except Exception as e:
        return f"匹配失败: {str(e)}"


class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise ValueError("无法打开摄像头")
        # 设置摄像头帧率
        self.stream.set(cv2.CAP_PROP_FPS, 60)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update_time = time.time()

        self.queue = Queue(maxsize=10)  # 限制队列大小防止内存溢出
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                break

            if self.queue.full():
                self.queue.get()  # 丢弃旧帧保持队列大小

            (self.grabbed, frame) = self.stream.read()
            self.queue.put(frame)

            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            if elapsed_time >= 0.2:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.last_update_time = current_time
            self.frame_count += 1

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()


class FaceRecognitionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸识别系统")
        self.setFixedSize(1280, 800)
        self.npz_file = 'E:\\desk\\biyesheji\\duplication1\\UI\\face_data.npy'  # 假设你的npy文件名为faces.npy
        self.face_data = self.load_data()  # 加载npy文件中的字典
        self.face_ids = self.get_face_ids()  # 获取字典中'IDs'键对应的内容
        self.yaml_path = "E:\\desk\\biyesheji\\duplication1\\ultralytics\\cfg\\models\\v8\\yolov8_01.yaml"
        self.pt_path = "E:\\desk\\biyesheji\\duplication1\\best.pt"
        self.camera = None
        self.init_ui()

    def load_data(self):
        """加载npy文件中的字典"""
        try:
            data = np.load(self.npz_file, allow_pickle=True).item()  # 加载为字典
            return data
        except FileNotFoundError:
            return {}  # 如果文件不存在，返回空字典

    def get_face_ids(self):
        """从字典中获取'IDs'键对应的内容"""
        return self.face_data.get('ids', [])  # 获取字典中'IDs'对应的内容，如果没有则返回空列表

    def save_data(self):
        """保存数据到npy文件"""
        np.save(self.npz_file, self.face_data)

    def init_ui(self):
        # 设置窗口背景颜色为浅灰色，更显现代化
        self.setStyleSheet("background-color: #f4f4f9;")

        # 标题
        self.title = QLabel("人脸识别系统", self)
        self.title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        # 标题使用深紫色背景，搭配白色文字，突出显示
        self.title.setStyleSheet("""
            background-color: #6a5acd;
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        """)

        # 人脸管理列表
        self.face_list = QListWidget(self)
        # 列表使用白色背景，增加可读性
        self.face_list.setStyleSheet("""
            background-color: white;
            color: #333333;
            border: none;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            padding: 10px;
        """)
        self.update_face_list()  # 更新ID列表显示

        # 按钮
        self.button_import = QPushButton("导入人脸信息")
        self.button_switch = QPushButton("切换算法")
        self.button_recognize = QPushButton("识别")
        self.button_delete = QPushButton("删除ID")  # 新增删除按钮

        # 设置按钮样式，使用紫色系，更显美观和现代化
        button_style = """
            QPushButton {
                background-color: #8a7ec9;
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                transition: background-color 0.3s ease;
            }
            QPushButton:hover {
                background-color: #7b68ee;
            }
            QPushButton:pressed {
                background-color: #6a5acd;
            }
        """
        self.button_import.setStyleSheet(button_style)
        self.button_switch.setStyleSheet(button_style)
        self.button_recognize.setStyleSheet(button_style)
        self.button_delete.setStyleSheet(button_style)

        # 设置按钮事件
        self.button_import.clicked.connect(self.import_id)
        self.button_delete.clicked.connect(self.delete_id)
        self.button_recognize.clicked.connect(self.recognize_face)
        self.button_switch.clicked.connect(self.choose_algorithm)

        # 按钮布局
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.button_import)
        button_layout.addWidget(self.button_switch)
        button_layout.addWidget(self.button_recognize)
        button_layout.addWidget(self.button_delete)  # 添加删除按钮
        button_layout.setSpacing(20)

        # 总体布局
        main_layout = QHBoxLayout()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.face_list)
        main_layout.setSpacing(30)

        # 总布局
        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(main_layout)
        layout.setSpacing(30)
        layout.setContentsMargins(30, 30, 30, 30)

        self.setLayout(layout)

    def choose_algorithm(self, default_model="YOLOv8-attention"):
        """弹窗选择不同的 YOLO 模型进行加载，默认使用 `default_model`"""

        models = {
            "YOLOv8-attention": {
                "yaml": "E:\\desk\\biyesheji\\duplication1\\ultralytics\\cfg\\models\\v8\\yolov8_01.yaml",
                "pt": "E:\\desk\\biyesheji\\duplication1\\best.pt"
            },
            "YOLOv11": {
                "yaml": "E:\\desk\\biyesheji\\duplication1\\ultralytics\\cfg\\models\\11\\yolo11.yaml",
                "pt": "E:\\desk\\biyesheji\\duplication1\\yolo11n.pt"
            }
            # 可以继续添加更多模型
        }

        model_names = list(models.keys())  # 获取可用的模型名称

        # 弹出对话框选择模型，默认选中 `default_model`
        model_choice, ok = QInputDialog.getItem(
            self, "选择识别算法", "请选择 YOLO 模型:", model_names,
            model_names.index(default_model) if default_model in model_names else 0, False
        )

        if ok and model_choice:  # 用户选择了模型
            yaml_path = models[model_choice]["yaml"]
            pt_path = models[model_choice]["pt"]

            try:
                # 加载模型
                self.model = YOLO(yaml_path).load(pt_path)
                QMessageBox.information(self, "模型加载成功", f"已成功加载 {model_choice} 模型！")
                return yaml_path, pt_path  # 返回模型路径
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"模型加载失败: {str(e)}")
                return None, None
        return None, None

    def update_face_list(self):
        """更新人脸ID列表显示"""
        self.face_list.clear()
        for face_id in self.face_ids:
            self.face_list.addItem(f"{face_id}")

    def import_id(self):
        """导入人脸信息并更新npy文件"""
        # 1. 获取用户输入的 ID
        new_id, ok = QInputDialog.getText(self, "输入人脸ID", "请输入人脸ID：")
        if not ok or not new_id.strip():  # 用户取消或输入为空
            return

        new_id = new_id.strip()

        if new_id in self.face_ids:
            QMessageBox.warning(self, "警告", "该ID已经存在！")
            return

        # 2. 让用户选择人脸照片
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "导入人脸照片", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)",
                                              options=options)

        if not file:  # 用户未选择文件
            return

        # 3. 读取人脸图片
        try:
            image = face_recognition.load_image_file(file)  # 读取图片
            face_encodings = face_recognition.face_encodings(image)  # 提取人脸特征

            if not face_encodings:
                QMessageBox.warning(self, "警告", "未检测到人脸，请选择包含人脸的图片！")
                return

            face_encoding = face_encodings[0]  # 只取第一张脸

            # 4. 确保 face_data["encodings"] 初始化
            if "encodings" not in self.face_data:
                self.face_data["encodings"] = []
            if "ids" not in self.face_data:
                self.face_data["ids"] = []

            # 5. 添加 ID 和编码，并保存
            self.face_data["encodings"].append(face_encoding.tolist())  # 转换为列表
            self.face_data["ids"].append(new_id)

            self.save_data()  # 保存数据
            self.update_face_list()  # 更新列表

            QMessageBox.information(self, "成功", f"成功导入人脸ID: {new_id}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入人脸失败: {str(e)}")

    def delete_id(self):
        """删除选中的ID并更新npy文件"""
        selected_item = self.face_list.currentItem()
        if selected_item:
            selected_id = selected_item.text().strip()  # 直接获取ID，而不是使用 split

            if selected_id in self.face_ids:
                index = self.face_ids.index(selected_id)  # 获取ID的索引
                self.face_ids.pop(index)  # 删除ID
                self.face_data["ids"] = self.face_ids  # 更新字典中的ID列表

                if "encodings" in self.face_data and index < len(self.face_data["encodings"]):
                    self.face_data["encodings"].pop(index)  # 删除对应的编码

                self.save_data()  # 更新npy文件
                self.update_face_list()  # 更新列表
            else:
                QMessageBox.warning(self, "警告", "该ID不存在！")
        else:
            QMessageBox.warning(self, "警告", "请先选择一个ID删除！")

    def recognize_face(self):
        """弹窗选择输入来源：摄像头、视频或图片"""
        options = ["摄像头", "视频文件", "图片"]
        choice, ok = QInputDialog.getItem(self, "选择输入方式", "请选择输入来源：", options, 0, False)

        if not ok:  # 用户取消
            return

        if choice == "摄像头":
            self.recognize_face_video(mode="camera")
        elif choice == "视频文件":
            self.recognize_face_video(mode="video")
        elif choice == "图片":
            self.recognize_face_image()

    def recognize_face_video(self, mode="camera"):
        try:
            # 选择更快的 YOLO 模型
            yaml_path, pt_path = self.choose_algorithm(default_model="yolov8n")
            model = YOLO(yaml_path).load(pt_path)

            if mode == "camera":
                self.camera = CameraStream()
            else:
                options = QFileDialog.Options()
                video_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "",
                                                            "Video Files (*.mp4 *.avi *.mov *.mkv)",
                                                            options=options)
                if not video_path:
                    return
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    QMessageBox.critical(self, "错误", "无法打开视频文件")
                    return
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_delay = int(1000 / fps)

            cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
            # 创建线程池，设置最大工作线程数为 8
            with ThreadPoolExecutor(max_workers=8) as executor:
                last_confidence_update = time.time()
                while True:
                    if mode == "camera":
                        if self.camera is None:
                            break
                        frame = self.camera.read()
                        if frame is None:
                            break
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            break

                    start_time = time.time()

                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    # 提交模型推理任务到线程池
                    future = executor.submit(model, image, imgsz=320, half=False)
                    results = future.result()

                    # 绘制FPS并居中显示
                    fps_text = f"FPS: {self.camera.fps:.1f}" if mode == "camera" else f"FPS: {fps:.1f}"
                    text_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = 30
                    cv2.putText(frame, fps_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 绘制检测结果
                    for box in results[0].boxes:
                        if box.cls == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # 计算字体大小
                            face_width = x2 - x1
                            font_scale = min(face_width / 200, 1.5)
                            font_scale = max(font_scale, 0.5)

                            # 获取置信度
                            confidence = box.conf.cpu().numpy()[0]

                            # 每 0.2 秒更新一次置信度显示
                            if time.time() - last_confidence_update >= 0.2:
                                last_confidence_update = time.time()

                            # 显示置信度
                            text = f"{confidence * 100:.2f}%"
                            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),
                                        2)

                    cv2.imshow("Face Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Face Recognition",
                                                                                  cv2.WND_PROP_VISIBLE) < 1:
                        break

        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别失败: {str(e)}")
        finally:
            if mode == "camera" and self.camera:
                self.camera.stop()
            elif mode != "camera":
                cap.release()
            cv2.destroyAllWindows()

    def recognize_face_image(self):
        """使用 YOLO 进行图片人脸识别"""
        options = QFileDialog.Options()
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                    "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if not image_path:
            return

        try:
            yaml_path, pt_path = self.choose_algorithm(default_model="YOLOv8-attention")
            if not yaml_path or not pt_path:
                return

            model = YOLO(yaml_path).load(pt_path)
            image = cv2.imread(image_path)
            result = model(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

            faces = []
            for box in result[0].boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    faces.append([x1, y1, x2, y2])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 计算字体大小
                    face_width = x2 - x1
                    font_scale = min(face_width / 200, 1.2)
                    font_scale = max(font_scale, 0.5)

                    # 获取置信度
                    confidence = box.conf.cpu().numpy()[0]

                    # 显示置信度
                    text = f"{confidence * 100:.2f}%"
                    cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)

            cv2.imshow("Face Recognition", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"人脸识别失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionWindow()
    window.show()
    sys.exit(app.exec_())