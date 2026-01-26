import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,10)
# print(x)
y = np.sin(x)

x2 = np.linspace(0,10,1000)
y2 = np.sin(x2)

# # sin函数图
# plt.plot(x,y)
# plt.title('y = sin(x)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# # 散点图 -> 拟合结果
# plt.scatter(x,y,marker = '*',label = 'samples')
# plt.plot(x2,y2,linestyle = '--',label = 'fitted curve')
# plt.legend()
# plt.show()

# # 一张窗口显示两个图
# fig, axes = plt.subplots(1,2)
# axes[0].scatter(x,y,marker = '*', label = 'sample')
# axes[0].set_xlabel('x')
# axes[0].set_ylabel('y')
# axes[0].set_title('sample')
# axes[1].plot(x,y,linestyle = '--', label = 'fitted curve')
# axes[1].set_xlabel('x')
# axes[1].set_ylabel('y')
# axes[1].set_title('fitted curve')
# plt.show()

# # 柱状图
# x = [1,2,3]
# y = [4,6,10]
# plt.bar(x,y)
# plt.show()