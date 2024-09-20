# """
# Author: Coder729
# Date: 2024/9/14
# Description: Rosenbrock：f(x, y) = (a - x)^2 + b(y - x^2)^2, a=1, b=100
# Black point is the start point, yellow point is the optimal point.
# """
# import numpy as np
# import matplotlib.pyplot as plt
#
# def rosenbrock(x, a=1, b=100):
#     return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
#
# def rosenbrock_gradient(x, a=1, b=100):
#     grad_x1 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
#     grad_x2 = 2 * b * (x[1] - x[0]**2)
#     return np.array([grad_x1, grad_x2])
#
# def rosenbrock_hessian(x, a=1, b=100):
#     h11 = 2 - 4 * b * x[1] + 12 * b * x[0]**2
#     h12 = -4 * b * x[0]
#     h22 = 2 * b
#     return np.array([[h11, h12], [h12, h22]])
#
# def steepest_descent_rosenbrock(x0, tol=1e-6, max_iter=1000, alpha=0.002):
#     x = x0
#     trajectory = [x0]
#     for _ in range(max_iter):
#         grad = rosenbrock_gradient(x)
#         if np.linalg.norm(grad) < tol:
#             break
#         x = x - alpha * grad
#         trajectory.append(x)
#     return np.array(trajectory)
#
# def newton_method_rosenbrock(x0, tol=1e-6, max_iter=100):
#     x = x0
#     trajectory = [x0]
#     for _ in range(max_iter):
#         grad = rosenbrock_gradient(x)
#         if np.linalg.norm(grad) < tol:
#             break
#         hessian_inv = np.linalg.inv(rosenbrock_hessian(x))
#         x = x - hessian_inv @ grad
#         trajectory.append(x)
#     return np.array(trajectory)
#
# def conjugate_gradient_rosenbrock(x0, tol=1e-8, max_iter=210):
#     x = x0
#     r = -rosenbrock_gradient(x)
#     p = r
#     trajectory = [x0]
#     for i in range(max_iter):
#         if np.linalg.norm(r) < tol:
#             break
#         grad = rosenbrock_gradient(x)
#         hessian = rosenbrock_hessian(x)
#         alpha = (r.T @ r) / (p.T @ hessian @ p)
#         x = x + alpha * p
#         r_new = -rosenbrock_gradient(x)
#         beta = (r_new.T @ r_new) / (r.T @ r)
#         p = r_new + beta * p
#         r = r_new
#         trajectory.append(x)
#     return np.array(trajectory)
#
#
# x0 = np.array([-1.5, 2.0])
# trajectory_sd = steepest_descent_rosenbrock(x0)
# trajectory_nm = newton_method_rosenbrock(x0)
# trajectory_cg = conjugate_gradient_rosenbrock(x0)
# x1_vals = np.linspace(-2, 2, 400)
# x2_vals = np.linspace(-1, 3, 400)
# X1, X2 = np.meshgrid(x1_vals, x2_vals)
# Z = (1 - X1)**2 + 100 * (X2 - X1**2)**2
#
# plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
# plt.plot(trajectory_sd[:, 0], trajectory_sd[:, 1], 'r.-', label="Steepest Descent", linewidth=2)
# plt.plot(trajectory_nm[:, 0], trajectory_nm[:, 1], 'g.--', label="Newton's Method", linewidth=2)
# plt.plot(trajectory_cg[:, 0], trajectory_cg[:, 1], 'b.-.', label="Conjugate Gradient", linewidth=2)
# plt.scatter([x0[0]], [x0[1]], color='black', label='Start', zorder=5)
# plt.scatter([1], [1], color='yellow', label='Optimal', zorder=5)
# plt.title("Rosenbrock Function Optimization")
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def obj(x, prob):
    switcher = {
        1: allgower_function,
        2: penalty_function,
        3: boundary_value_function,
        4: schittkowski_function,
        5: yang_tridiagonal_function
    }
    return switcher.get(prob)(x)

def allgower_function(x):
    n = len(x)
    ss1 = sum(x)
    f = 0.0
    for i in range(n):
        ss2 = (i + 1) * ss1
        ss3 = np.cos(ss2)
        ss4 = np.exp(ss3)
        f += (x[i] - ss4) ** 2
    return f

def penalty_function(x):
    a = 0.00001
    f = sum((x[i] - 1) ** 2 for i in range(len(x)))
    h = sum(x[i] ** 2 for i in range(len(x)))
    return a * f + (h - 0.25) ** 2

def boundary_value_function(x):
    n = len(x)
    rh1 = 1.0 / (n + 1)
    rh2 = rh1 * rh1 / 2.0
    f = 0.0

    if n > 1:
        f += (2.0 * x[0] - x[1] + rh2 * (x[0] + rh1 + 1.0) ** 3) ** 2

    for i in range(1, n - 1):
        f += (2 * x[i] - x[i - 1] - x[i + 1] + rh2 * (x[i] + (i + 1) * rh1 + 1.0) ** 3) ** 2

    if n > 1:
        f += (2.0 * x[n - 1] - x[n - 2] + rh2 * (x[n - 1] + (n + 1) * rh1 + 1.0) ** 3) ** 2

    return f

def schittkowski_function(x):
    n = len(x)
    ss1 = sum(0.5 * (i + 1) * x[i] for i in range(n))
    ss2 = ss1 ** 2
    f = ss2 * (1.0 + ss2) + sum(x[i] ** 2 for i in range(n))
    return f

def yang_tridiagonal_function(x):
    n = len(x)
    f = 0.0
    for i in range(n):
        if i > 0 and i < n - 1:
            ss1 = x[i - 1] + x[i] + x[i + 1]
        elif i == 0:
            ss1 = x[i] + x[i + 1]
        else:
            ss1 = x[i - 1] + x[i]
        ss2 = (i + 1) * ss1
        ss3 = np.cos(ss2)
        ss4 = np.exp(ss3)
        f += (x[i] - ss4) ** 2
    return f

def schittkowski_function(x):
    n = len(x)
    ss1 = sum(0.5 * (i + 1) * x[i] for i in range(n))
    ss2 = ss1 ** 2
    f = ss2 * (1.0 + ss2) + sum(x[i] ** 2 for i in range(n))
    return f

# 定义梯度计算
def grad_obj(x, prob):
    n = len(x)
    grad = np.zeros(n)
    if prob == 4:  # Schittkowski函数的梯度
        ss1 = sum(0.5 * (i + 1) * x[i] for i in range(n))
        ss2 = ss1 ** 2
        for i in range(n):
            grad[i] = (ss2 * (2 * (i + 1) * ss1 + (1.0 + ss2)) + 2 * x[i])  # 梯度
    return grad

# 定义Hessian矩阵计算
def hessian_obj(x, prob):
    n = len(x)
    hessian = np.zeros((n, n))
    if prob == 4:  # Schittkowski函数的Hessian
        ss1 = sum(0.5 * (i + 1) * x[i] for i in range(n))
        for i in range(n):
            hessian[i, i] = 2 + 2 * ss1 * (i + 1)  # 对角线元素
            for j in range(n):
                if i != j:
                    hessian[i, j] = 0  # 非对角线元素
    return hessian

# 最速下降法
def steepest_descent(x0, func, grad_func, prob, alpha=0.01, tol=1e-6, max_iter=1000):
    x = x0
    path = [x.copy()]
    for _ in range(max_iter):
        gradient = grad_func(x, prob)
        x_new = x - alpha * gradient
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.array(path)

# 牛顿法
def newton_method(x0, func, grad_func, hess_func, prob, tol=1e-6, max_iter=1000):
    x = x0
    path = [x.copy()]
    for _ in range(max_iter):
        gradient = grad_func(x, prob)
        hessian = hess_func(x, prob)
        x_new = x - np.linalg.inv(hessian) @ gradient
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, np.array(path)

# 初始化
x0 = np.array([0.1, 0.1, 0.1, 0.1])  # 更小的初始值
prob = 4  # 选择 Schittkowski 函数

# 运行两种优化方法
optimal_sd, path_sd = steepest_descent(x0, schittkowski_function, grad_obj, prob, alpha=0.1)
optimal_newton, path_newton = newton_method(x0, schittkowski_function, grad_obj, hessian_obj, prob)

# 可视化收敛路径
plt.figure(figsize=(10, 6))
plt.plot(path_sd[:, 0], path_sd[:, 1], 'ro-', label='Steepest Descent Path')
plt.plot(path_newton[:, 0], path_newton[:, 1], 'bo-', label='Newton\'s Method Path')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Convergence Path Comparison (Schittkowski Function)')
plt.legend()
plt.grid()
plt.show()

# 输出结果
print("最速下降法的最优解:", optimal_sd)
print("牛顿法的最优解:", optimal_newton)