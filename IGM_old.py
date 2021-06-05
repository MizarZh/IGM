import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from math import cos, sin, log, exp, asin, acos, atan, sqrt, ceil, pi, degrees, radians
import math
# 初始值
# 地球常数
Gr = 6.67384 * 10**(-11)  # 万有引力常数
M = 5.965 * 10**24  # 地球质量
Re = 12756 / 2 * 10**3  # 地球半径

# 火箭常数
F = 80 * 10**5  # 推力
m = 10000  # 质量
# m0 # 火箭质量
# mf # 燃料质量
m_dt = 42
Ve = F / m_dt  # 发动机喷气速度、特征速度

# 轨道常数
# 最终点重力加速度
g_xi_c = 0  # 垂直向下，为0
g_ita_c = Gr * M / (Re + 300 * 10**3)**2  # 目标点的重力加速度
g_zeta_c = 0  # 垂直向下，为0

# 最终点速度
xi_c_dt = sqrt(Gr * M / (Re + 300 * 10**3))
ita_c_dt = 0
zeta_c_dt = 0  # 与发射方向没有偏差

# 最终点坐标
xi_c = 0
ita_c = Re + 300 * 10**3  # 300km 近地轨道
zeta_c = 0  # 与发射方向没有偏差

cos_theta_HC = 0.999  # 目标点弹道倾角的余弦值  试一个近圆轨道
deltaT = 1  # 运算间隔
epsilon = 0.001  # 判断tc迭代相差的误差值
tc = 11  # tc初始值
tc_pre = tc + 1
# 在xyz坐标系中敏感元件测量的火箭的视加速度
Wx_dt = 0
Wy_dt = F / m
Wz_dt = 0
# xyz坐标系中的速度
x_dt = 0
y_dt = 1
z_dt = 0
# 坐标
x = 0
y = Re
z = 0
# 重力加速度
gx = 0
gy = -Gr * M / Re**2
gz = 0

xs = []
ys = []

# 模拟开始
t = 0
dt = 0.01
while (t < tc - 1):
    # 控制模块
    if round(t / deltaT, 3) == round(t / deltaT):  # 当deltaT正确的时候
        # 控制模块
        # phi 俯仰角
        # a part
        r_star = sqrt(x**2 + y**2)
        beta_e = asin(x / r_star)
        V_star = sqrt(x_dt**2 + y_dt**2)
        cos_theta_H_star = abs(x * y_dt - y * x_dt) / (r_star * V_star)

        # b part
        W_dt_star = sqrt(Wx_dt**2 + Wy_dt**2)
        W_dt = sqrt(Wx_dt**2 + Wy_dt**2 + Wz_dt**2)
        tau = Ve / W_dt
        tau_xi_ita = Ve / W_dt_star
        #         print(tau_xi_ita, tc)

        count = 0
        while (abs(tc - tc_pre) > epsilon):
            if tc > tau_xi_ita:
                tc = tau_xi_ita - 10
            # c part
            A1 = Ve * log(tau_xi_ita / (tau_xi_ita - tc))
            A2 = A1 * tau_xi_ita - Ve * tc
            A3 = A1 * tc - A2
            A4 = A3 * tau_xi_ita - 0.5 * Ve * tc**2

            # d part
            beta_t = (V_star * tc * cos_theta_H_star +
                      A3 * cos_theta_HC) / ita_c
            beta_c = beta_e + beta_t

            # e part
            xyz2xi_ita_zetaMatrix = np.array([[cos(beta_c), -sin(beta_c), 0],
                                              [sin(beta_c),
                                               cos(beta_c), 0], [0, 0, 1]])
            xi, ita, zeta = xyz2xi_ita_zetaMatrix.dot(np.array([x, y, z]))
            xi_dt, ita_dt, zeta_dt = xyz2xi_ita_zetaMatrix.dot(
                np.array([x_dt, y_dt, z_dt]))
            g_xi, g_ita, g_zeta = (
                xyz2xi_ita_zetaMatrix.dot(np.array([gx, gy, gz])) +
                np.array([g_zeta_c, g_ita_c, g_zeta_c])) * 0.5

            # f part
            tc_pre = tc
            deltaV = sqrt((xi_c_dt - xi_dt - g_xi * tc)**2 +
                          (ita_c_dt - ita_dt - g_ita * tc)**2 +
                          (zeta_c_dt - zeta_dt - g_zeta * tc)**2)
            tc = tau * (1 - exp(-deltaV / Ve))
            count += 1
            if count > 500:
                raise Exception('错误')
#             print('tc',tc,'tc_pre',tc_pre)
#         print(round(t,3), tc, math.degrees(beta_c))
# g part
        phi_xi_wave = atan(
            (ita_c_dt - ita_dt - g_ita) / (xi_c_dt - xi_dt - g_xi * tc))
        P = A3 * cos(phi_xi_wave)
        Q = A4 * cos(phi_xi_wave)
        R = ita_c - ita - ita_dt * tc - 0.5 * g_ita * tc**2 - A3 * sin(
            phi_xi_wave)
        delta_k = A1 * Q - A2 * P
        k1 = A2 * R / delta_k
        k2 = A1 * R / delta_k

        # psi 偏航角
        B1 = Ve * log(tau / (tau - tc))
        B2 = B1 * tau - Ve * tc
        B3 = B1 * tc - B2
        B4 = B3 * tau - 0.5 * Ve * tc**2

        psi_zeta_wave = asin((zeta_c_dt + zeta_dt + g_zeta * tc) / B1)
        E = B3 * cos(psi_zeta_wave)
        G = B4 * cos(psi_zeta_wave)
        H = zeta_c - zeta - zeta_dt * tc - 0.5 * g_zeta * tc**2 + B3 * sin(
            psi_zeta_wave)
        delta_e = B2 * E - B1 * G
        e1 = B2 * H / delta_e
        e2 = B1 * H / delta_e

        tc -= deltaT
    phi_xi = phi_xi_wave + k2 * t - k1
    psi_zeta = psi_zeta_wave + e2 * t - e1
    xi_ita_zeta2xyzMatrix = np.array([[cos(beta_c),
                                       sin(beta_c), 0],
                                      [-sin(beta_c),
                                       cos(beta_c), 0], [0, 0, 1]])
    r = sqrt(x**2 + y**2 + z**2)
    gxyz = Gr * M / r**2
    gx, gy, gz = -gxyz * np.array([x / r, y / r, z / r])
    g_xi_simu, g_ita_simu, g_zeta_simu = xyz2xi_ita_zetaMatrix.dot(
        np.array([gx, gy, gz]))

    xi_dt_dt, ita_dt_dt, zeta_dt_dt = F / m * np.array([
        cos(phi_xi) * cos(psi_zeta),
        sin(phi_xi) * cos(psi_zeta), -sin(psi_zeta)
    ]) + np.array([g_xi_simu, g_ita_simu, g_zeta_simu])
    xi_dt, ita_dt, zeta_dt = np.array([
        xi_dt, ita_dt, zeta_dt
    ]) + np.array([xi_dt_dt, ita_dt_dt, zeta_dt_dt]) * dt
    xi, ita, zeta = np.array([xi, ita, zeta
                              ]) + np.array([xi_dt, ita_dt, zeta_dt]) * dt

    Wx_dt, Wy_dt, Wz_dt = xi_ita_zeta2xyzMatrix.dot(
        np.array([xi_dt_dt, ita_dt_dt, zeta_dt_dt]))
    x_dt, y_dt, z_dt = xi_ita_zeta2xyzMatrix.dot(
        np.array([xi_dt, ita_dt, zeta_dt]))
    x, y, z = xi_ita_zeta2xyzMatrix.dot(np.array([xi, ita, zeta]))
    #     print(x,y,z)
    m -= m_dt * dt
    #     print(round(t,3), x,y,z)
    t += dt
    xs.append(x)
    ys.append(y)

print(ys[-1] - Re)
plt.figure(dpi=200, figsize=[4, 4])
# plt.plot(xs,np.array(ys)-Re)

angle = np.linspace(0, 2 * math.pi, 10000)
plt.plot(Re * np.cos(angle), Re * np.sin(angle))
plt.plot(xs, ys)
plt.figure(dpi=200)
plt.plot(xs, np.array(ys) - Re)
