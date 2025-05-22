import numpy as np

# 加载npy文件时启用allow_pickle
data = np.load('face_data.npy', allow_pickle=True)

# 打印加载的数组内容
print(data)

# 提取数组中包含'IDs'键的元素
for item in data:
    if isinstance(item, dict) and 'IDs' in item:
        face_ids = item['IDs']
        print("IDs内容：", face_ids)
