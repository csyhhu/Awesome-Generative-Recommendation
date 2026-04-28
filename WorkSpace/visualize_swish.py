import numpy as np
import matplotlib.pyplot as plt

# 定义Swish函数（支持可变的β参数）
def swish(x, beta=1.0):
    return x / (1 + np.exp(-beta * x))

# 定义Swish的导数（用于梯度分析）
def swish_derivative(x, beta=1.0):
    sigmoid = 1 / (1 + np.exp(-beta * x))
    return sigmoid + beta * x * sigmoid * (1 - sigmoid)

# 定义对比函数
def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成输入范围
x = np.linspace(-4, 4, 1000)

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Swish与主流激活函数对比
ax1 = axes[0, 0]
ax1.plot(x, swish(x, beta=1), label='Swish (β=1)', linewidth=3, color='red')
ax1.plot(x, relu(x), label='ReLU', linestyle='--', color='blue')
ax1.plot(x, gelu(x), label='GELU', linestyle='-.', color='green')
ax1.plot(x, sigmoid(x), label='Sigmoid', linestyle=':', color='purple')
ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax1.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
ax1.set_xlabel('Input (x)')
ax1.set_ylabel('Output')
ax1.set_title('Swish vs Other Activation Functions')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-1, 4)

# 2. Swish不同β值的影响
ax2 = axes[0, 1]
betas = [0.1, 0.5, 1.0, 2.0, 5.0]
colors = plt.cm.viridis(np.linspace(0, 1, len(betas)))
for beta, color in zip(betas, colors):
    ax2.plot(x, swish(x, beta), label=f'β={beta}', linewidth=2, color=color)
ax2.plot(x, relu(x), label='ReLU', linestyle='--', color='black', linewidth=1.5)
ax2.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax2.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
ax2.set_xlabel('Input (x)')
ax2.set_ylabel('Output')
ax2.set_title('Swish with Different β Values')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-1, 4)

# 3. Swish的梯度（一阶导数）
ax3 = axes[1, 0]
ax3.plot(x, swish_derivative(x, beta=1), label="Swish' (β=1)", linewidth=3, color='red')
# ReLU的"导数"（实际是次梯度）
relu_grad = np.where(x > 0, 1, 0)
ax3.plot(x, relu_grad, label="ReLU'", linestyle='--', color='blue')
ax3.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax3.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
ax3.set_xlabel('Input (x)')
ax3.set_ylabel('Gradient')
ax3.set_title('Gradients: Swish vs ReLU')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-4, 4)
ax3.set_ylim(-0.1, 1.2)

# 4. Swish与GELU的详细对比（负区放大）
ax4 = axes[1, 1]
ax4.plot(x, swish(x, beta=1), label='Swish (β=1)', linewidth=3, color='red')
ax4.plot(x, gelu(x), label='GELU', linewidth=3, color='green', alpha=0.7)
ax4.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax4.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
ax4.set_xlabel('Input (x)')
ax4.set_ylabel('Output')
ax4.set_title('Detailed Comparison: Swish vs GELU (Negative Region)')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-3, 3)
ax4.set_ylim(-0.3, 2)  # 放大负区观察差异

plt.tight_layout()
plt.show()