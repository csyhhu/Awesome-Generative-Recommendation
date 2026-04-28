import numpy as np
import matplotlib.pyplot as plt

# 定义GELU精确公式（基于误差函数近似）
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 定义ReLU和Sigmoid作为对比
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成输入范围
x = np.linspace(-4, 4, 1000)
y_gelu = gelu(x)
y_relu = relu(x)
y_sigmoid = sigmoid(x)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(x, y_gelu, label='GELU', linewidth=3, color='blue')
plt.plot(x, y_relu, label='ReLU', linestyle='--', color='red')
plt.plot(x, y_sigmoid, label='Sigmoid', linestyle=':', color='green')
plt.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
plt.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.title('GELU vs ReLU vs Sigmoid Activation Functions', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xlim(-4, 4)
plt.ylim(-1, 4)
plt.show()