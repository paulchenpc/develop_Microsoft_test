import matplotlib.pyplot as plt
import numpy as np

# 定义画散点图的函数
def draw_scatter(n, s):
    """
    :param n: 点的数量，整数
    :param s: 点的大小，整数
    :return: None
    """
    # 加载数据
    data = np.loadtxt(r'F:\exercise_2.txt', encoding='utf-8', delimiter=',')
    # 通过切片获取横坐标x1
    x = data[:, 0]
    # 通过切片获取纵坐标R
    y = data[:, 1]

    # 横坐标x2
    # x2 = np.random.uniform(0, 5, n)
    # 纵坐标y2
    # y2 = np.array([3] * n)
    # 创建画图窗口
    fig = plt.figure()
    # 将画图窗口分成1行1列，选择第一块区域作子图
    ax = fig.add_subplot(1, 1, 1)
    # 设置标题
    ax.set_title('Result Analysis')
    # 设置横坐标名称
    ax.set_xlabel('x(average_consuming_level)')  # 人均消费水平
    # 设置纵坐标名称
    ax.set_ylabel('y(average_GDP)')  # 人均GDP
    # 画散点图
    ax.scatter(x, y, s=s, c='k', marker='.')
    # 画直线图
    # ax1.plot(x2, y2, c='b', ls='--')
    # 调整横坐标的上下界
    plt.xlim(xmax=100000, xmin=0)
    # 显示
    plt.show()

# 主模块
if __name__ == "__main__":
    # 运行
    # n: 点的数量，整数
    # s: 点的大小，整数
    draw_scatter(n=7, s=100)
