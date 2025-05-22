import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置中文字体


def plot_yolo_model():
    # 使用完整路径指定文件列表
    filenames = [
        r'E:\desk\biyesheji\duplication1\runs\detect\train13\results.csv'

    ]

    # 创建文件路径列表并检查文件是否存在
    files = [Path(filename) for filename in filenames if Path(filename).exists()]
    assert len(files) > 0, 'No specified csv files found, nothing to plot.'

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    colors = ['orange', 'olive', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'blue', 'cyan']

    # 处理每个文件
    for f, color in zip(files, colors):
        try:
            data = pd.read_csv(f)
            x = data.iloc[:, 0]
            y = data.iloc[:, 1].astype('float')  # 假设你关注的是第二列数据
            ax.plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8, color=color)
            ax.set_title('改进的Loss曲线对比', fontsize=14)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')

    ax.legend(fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    # 设置保存路径
    fig.savefig(Path('E:\\  desk\\biyesheji\\duplication1\\picture').joinpath('改进的loss曲线对比.png'),
                dpi=1000)  # 保存图表
    plt.close()


if __name__ == '__main__':
    plot_yolo_model()

