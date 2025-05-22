import face_recognition
import numpy as np


class FaceRecognitionDatabase:
    def __init__(self, file_path="face_data.npy"):
        # 初始化文件路径和数据
        self.file_path = file_path
        self.face_data = self.load_data()

    def load_data(self):
        """加载现有的面部识别数据"""
        try:
            # 加载.npy文件
            return np.load(self.file_path, allow_pickle=True).item()
        except FileNotFoundError:
            # 文件不存在时返回空数据结构
            return {"encodings": [], "ids": []}

    def save_data(self):
        """保存当前的数据到文件"""
        np.save(self.file_path, self.face_data)

    def add_data(self, image_path, person_id):
        """添加新的人脸数据"""
        # 加载图片
        image = face_recognition.load_image_file(image_path)
        # 提取该人脸的编码
        face_encoding = face_recognition.face_encodings(image)

        if face_encoding:
            # 如果提取到人脸编码，则添加到数据中
            self.face_data["encodings"].append(face_encoding[0].tolist())  # 转换为列表形式
            self.face_data["ids"].append(person_id)
            self.save_data()  # 保存数据
            print(f"已添加 {person_id} 的人脸数据")
        else:
            print("无法提取人脸编码，请检查图像")

    def delete_data(self, person_id):
        """删除指定ID的人脸数据"""
        if person_id in self.face_data["ids"]:
            # 找到该ID的索引
            index_to_delete = self.face_data["ids"].index(person_id)
            # 删除该ID和相应的特征向量
            del self.face_data["ids"][index_to_delete]
            del self.face_data["encodings"][index_to_delete]
            self.save_data()  # 保存数据
            print(f"已删除 {person_id} 的人脸数据")
        else:
            print(f"未找到ID为 '{person_id}' 的数据")

    def display_data(self):
        """显示当前保存的所有ID和人脸编码"""
        for person_id, encoding in zip(self.face_data["ids"], self.face_data["encodings"]):
            print(f"ID: {person_id}, 编码: {encoding}")

    import face_recognition


    def match_face(self, image_path):
        """与现有的数据库进行人脸匹配"""
        # 加载新图像
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        if not unknown_encoding:
            print("未能提取到该图像的人脸编码")
            return "Unknown"

        # 提取新图像的人脸编码
        unknown_encoding = unknown_encoding[0]

        # 使用 face_distance 计算人脸之间的距离
        distances = face_recognition.face_distance(self.face_data["encodings"], unknown_encoding)

        # 设置匹配阈值（可以降低这个值来提高匹配的灵敏度，建议范围是 0.4 到 0.6）
        threshold = 0.4  # 可以调整这个值来降低阈值

        # 如果某个距离小于阈值，则认为是匹配
        matches = [i for i, distance in enumerate(distances) if distance < threshold]

        if matches:
            matched_index = matches[0]  # 获取最小距离对应的索引
            matched_id = self.face_data["ids"][matched_index]
            print(f"匹配成功! 该人是: {matched_id}")
            return matched_id
        else:
            print("未能匹配到任何已知的人脸")
            return "Unknown"


# 示例使用
if __name__ == "__main__":
    # 创建人脸数据库对象，使用.npy文件格式
    db = FaceRecognitionDatabase("face_data.npy")


    # 显示所有数据
    db.display_data()

    # 匹配新图像
    db.match_face("D:\zz\work\\ultralytics-main\\ultralytics\\assets\huge.jpg")
