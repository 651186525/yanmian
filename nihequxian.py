# import numpy as np
# import scipy as sp
# from scipy.optimize import leastsq
# import matplotlib.pyplot as plt
#
# # 1.目标函数,进行拟合的数据点都分布在这条正弦曲线附近
# def real_func(x):
#     return np.sin(2*np.i*x)
#
# # 2.多项式  numpy.poly1d([1,2,3]) 生成1𝑥^2+2𝑥^1+3𝑥^0，p为多项式的参数,x为下面linspace(0, 1, 10)取得的点
# def fit_func(p, x):
#     f = np.poly1d(p)
#     return f(x)
#
# # 3.残差 误差函数，所谓误差就是指我们拟合的曲线的值对应真实值的差
# def residuals_func(p, x, y):
#     ret = fit_func(p, x) - y
#     return ret
#
# # 4.十个点
# x = np.linspace(0, 1, 10) #linspace均分计算指令，用于产生x1,x2之间的N点行线性的矢量 0到10以1为步长或者可以表达为(0,10,10):0到10输出10个值
# x_points = np.linspace(0, 1, 1000)
# # 加上正态分布噪音的目标函数的值
# y_ = real_func(x)
# y = [np.random.normal(0, 0.1) + y1 for y1 in y_]
#
# # 5.关于拟合的曲线的函数
# def fitting(M=0):
#     # M为多项式的次数，随机初始化多项式参数，生成M+1个随机数的列表，这样poyld函数返回的多项式次数就是M
#     p_init = np.random.rand(M + 1)
#     # 最小二乘法,三个参数：误差函数、函数参数列表，数据点
#     p_lsq = leastsq(residuals_func, p_init, args=(x, y))
#     print('Fitting Parameters:', p_lsq[0])
#
#     # 可视化
#     plt.plot(x_points, real_func(x_points), label='real')
#     plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
#     plt.plot(x, y, 'bo', label='noise')
#     plt.legend() #图例
#     plt.show()
#     return p_lsq
#
# p_lsq_9= fitting(M=9)
# # for i in range(10):
# #     fitting(i)
#
# # #结果显示过拟合， 引入正则化项(regularizer)，降低过拟合
# # regularization = 0.0001
# #
# # def residuals_func_regularization(p, x, y):
# #     ret = fit_func(p, x) - y   #残差
# #     ret = np.append(ret,
# #                     np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
# #     return ret
# #
# # # 最小二乘法,加正则化项
# # p_init = np.random.rand(9 + 1)
# # p_lsq_regularization = leastsq(
# #     residuals_func_regularization, p_init, args=(x, y))
# #
# # plt.plot(x_points, real_func(x_points), label='real')
# # plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
# # plt.plot(
# #     x_points,
# #     fit_func(p_lsq_regularization[0], x_points),
# #     label='regularization')
# # plt.plot(x, y, 'bo', label='noise')
# # plt.legend()
# # plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
y = np.array([5760, 3600, 1620, 1260, 1080, 900, 1080, 1800, 3060, 4680, 2880, 5040, 4140, 5580, 5040, 4860, 3780,
   3420, 4860, 3780, 4860, 5220, 4860, 3600])
z1 = np.polyfit(x, y, 6) # 用4次多项式拟合
p1 = np.poly1d(z1)
print(p1) # 在屏幕上打印拟合多项式
yvals=p1(x) # 也可以使用yvals=np.polyval(z1,x)
plot1=plt.plot(x, y, '*',label='original values')
plot2=plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend(loc=4) # 指定legend的位置,读者可以自己help它的用法
plt.title('polyfitting')
plt.show()


