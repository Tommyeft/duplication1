import psutil
import time
from ultralytics import YOLO
import os


def log_hardware_data(cpu_log_path, gpu_log_path, mem_log_path):
    cpu_percent = psutil.cpu_percent(interval=1)
    mem_percent = psutil.virtual_memory().percent
    # 由于 psutil 不直接支持 GPU 监控，这里简单占位，可使用 pynvml 库监控 NVIDIA GPU
    gpu_percent = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_percent = info.gpu
        pynvml.nvmlShutdown()
    except ImportError:
        print("pynvml 未安装，无法监控 GPU 使用率。")

    with open(cpu_log_path, 'a') as f:
        f.write(f"{time.time()},{cpu_percent}\n")
    with open(gpu_log_path, 'a') as f:
        f.write(f"{time.time()},{gpu_percent}\n")
    with open(mem_log_path, 'a') as f:
        f.write(f"{time.time()},{mem_percent}\n")


if __name__ == '__main__':
    # 构建和加载模型
    model = YOLO(r"E:\desk\Yolotrain\ultralytics\cfg\models\v8\yolov8.yaml")  # 从 YAML 构建新模型
    model = YOLO("yolov8n.pt")  # 加载预训练模型
    model = YOLO(r"E:\desk\Yolotrain\ultralytics\cfg\models\v8\yolov8.yaml").load(
        "yolov8n.pt")  # 结合 YAML 和预训练权重

    # 定义硬件数据记录路径
    cpu_log_path = r"E:\desk\Yolotrain\load\cpu"
    gpu_log_path = r"E:\desk\Yolotrain\load\gpu"
    mem_log_path = r"E:\desk\Yolotrain\load\mem"

    # 创建日志文件目录
    for path in [os.path.dirname(cpu_log_path), os.path.dirname(gpu_log_path), os.path.dirname(mem_log_path)]:
        if not os.path.exists(path):
            os.makedirs(path)

    # 训练模型
    train_results = model.train(
        data="data.yaml",  # 数据集 YAML 文件路径
        epochs=10,  # 适中的训练轮数
        imgsz=640,  # 训练图像尺寸
        device=0  # 使用 GPU 设备
    )

    # 在训练过程中记录硬件数据
    start_time = time.time()
    while time.time() - start_time < train_results.training_time:
        log_hardware_data(cpu_log_path, gpu_log_path, mem_log_path)
        time.sleep(5)  # 每 5 秒记录一次

    # 在验证集上评估模型性能
    metrics = model.val()

    # 对图像进行目标检测
    results = model(r"E:\desk\Yolotrain\ultralytics\assets\huge.jpg")
    results[0].show()

    # 将模型导出为 ONNX 格式
    path = model.export(format="onnx")  # 返回导出模型的路径
    print(f"Model exported to: {path}")
