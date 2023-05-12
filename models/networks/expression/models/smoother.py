from scipy import signal
import numpy as np
import math
import time


class Smoother():
    def __init__(self, window_width):
        self.window = signal.windows.hamming(window_width)
        self.window /= self.window.sum()
        # print(self.window)
        self.cache = []
        self.cache_len = window_width

    # 对模型输出的37个参数进行平滑
    def getValue(self, data: list):
        res_data = []
        for index, value in enumerate(data):
            # 当缓存列表为空的时候，将一维列表扩展为二维列表
            if len(self.cache) != len(data):
                self.cache.append([value])
                res_data.append(value)
            # 当缓存列表第二维不足窗函数长度时，扩展缓存列表第二维
            elif len(self.cache[index]) < self.cache_len - 1:
                self.cache[index].append(value)
                res_data.append(value)
            # 当缓存列表存第二维存入当前对应值后等于窗函数长度时，与窗函数做卷积并得到平滑后的输出
            elif len(self.cache[index]) == self.cache_len - 1:
                self.cache[index].append(value)
                res_data.append(float(np.convolve(self.cache[index], self.window, 'valid')))
            # 当缓存列表存第二维长度等于窗函数长度时，先将最早入队列的值出列，然后将当前值入列，最后与窗函数做卷积并得到平滑后的输出
            else:
                _ = self.cache[index].pop(0)
                self.cache[index].append(value)
                res_data.append(float(np.convolve(self.cache[index], self.window, 'valid')))
        return res_data
        # return scipy.signal.savgol_filter(data, 7, 3)


class OneEuroFilter:
    def __init__(self, num, min_cutoff=0.5, beta=0.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = 1.0
        # Previous values.
        self.x_prev = np.array([0.0 for x in range(num)])
        self.dx_prev = np.array([0.0 for x in range(num)])
        self.t_prev = time.time()

    @staticmethod
    def smoothing_factor(t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    @staticmethod
    def exponential_smoothing(a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def getValue(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev + 0.0001
        if t_e > 1.0:
            t_e = 0.03

        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        # dx = (x - self.x_prev) / t_e
        dx = np.array([(x[i] - self.x_prev[i])/t_e for i in range(len(x))])
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        # cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        cutoff = np.array([self.min_cutoff + self.beta * abs(dx_hat[i]) for i in range(len(dx_hat))])
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat.tolist()

if __name__ == '__main__':
    data = [
        [-0.00565722054203174, 0.07663204667291472, 0.001358739798888565, 0.0181527771637775, 0.0011176668090878855],
        [-0.0056857180149693575, 0.09247116491730724, 0.010267031811443823, 0.011015630119280624, 0.0010552646958136133],
        [-0.0038717806538833034, 0.09955807136637825, 0.021887937228062322, 0.005103923297221109, 0.0012617339296931668],
        [0.0033256171736866244, 0.10168218645932418, 0.03561716882645019, 0.001891670682068382, 0.0017129109889668012],
        [0.029403816742290353, 0.10766401853678481, 0.053782575530931354, -0.0008503092186791556, 0.0022052806951770826],
        [0.1442721977863195, 0.12462756357022695, 0.06625187323827829, -0.0022065687059823953, 0.0018067859928123653]]
    smoother = Smoother(window_width=10)
    oef = OneEuroFilter(5, 0.5, 0.0)
    for i in data:
        res = oef.getValue(time.time(), i)
        # res = smoother.getValue(i)
        print(res)
