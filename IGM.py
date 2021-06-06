import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp
from math import cos, sin, log, exp, asin, acos, atan, sqrt, ceil, pi, degrees, radians
import math

# 各坐标系的代号：发射点惯性坐标系-icf,升交点轨道坐标系-rcf,入轨点轨道坐标系-ocf

# 初始值
# 地球常数
Gr = 6.67384 * 10**(-11)  # 万有引力常数
M = 5.965 * 10**24  # 地球质量
Re = 12756 / 2 * 10**3  # 地球半径
mu = Gr * M  # 反正是个常数

# 火箭常数
F = 8 * 10**5  # 推力
m = 86500  # 质量
# mf # 燃料质量
m_dt = 272
Ve = F / m_dt  # 发动机喷气速度、特征速度

# 轨道参数
a = 6622785.34  # 半长轴
e = 0  # 偏心率
i = radians(68.846)  # 倾角
Omega = radians(-9.964)  # 升交点赤经
omega = radians(0)  # 近地点角距
f = radians(58.015)  # 入轨点的真近点角
cos_theta_HC = radians(0)  # 目标点弹道倾角的余弦值  试一个近圆轨道

# 发射点
BT = radians(28.64)  # 发射点地理纬度
AT = asin(cos(i) / cos(BT))  # 发射方位角
deltaLambda = radians(-19.964)  # 升交点与发射点经度差


# 三种旋转矩阵
def Mx(A):
    return np.matrix([[1, 0, 0], [0, cos(A), sin(A)], [0, -sin(A), cos(A)]])


def My(B):
    return np.matrix([[cos(B), 0, -sin(B)], [0, 1, 0], [sin(B), 0, cos(B)]])


def Mz(C):
    return np.matrix([[cos(C), sin(C), 0], [-sin(C), cos(C), 0], [0, 0, 1]])


def phipsi(phi, psi):
    return np.array([cos(phi) * cos(psi), sin(phi) * cos(psi), -sin(psi)])


MoOri = My(i) * Mz(-deltaLambda) * My(-pi / 2) * Mz(BT) * My(AT)  # icf->rcf

beta_c = f + omega
MaOri = Mz(-beta_c) * MoOri  # icf->ocf

Mo = np.array(MoOri)
Ma = np.array(MaOri)

# 初始位置,icf
r = np.array([2.9058E5, 92126 + Re, -33401])
# x = 2.9058E5
# y = 92126 + Re
# z = -33401

# 初始速度,icf
V = np.array([2815.75, 494.703, 70.953])
# Vx = 2815.75
# Vy = 494.703
# Vz = 70.953

g = -mu / norm(r)**3 * r  # 初始重力加速度,icf

phi0 = radians(61.3383)  # 初始俯仰角,icf
psi0 = radians(0.3105)  # 初始偏航角,icf
Ib = phipsi(phi0, psi0)

t = 0
dt = 0.01
deltaT = 1  # 运算间隔
epsilon = 0.001  # 判断tc迭代相差的误差值
tc = 11  # tc初始值
tc_pre = tc + 1

rs = []
Vs = []
Ws = []
while (True):

    if round(t / deltaT, 3) == round(t / deltaT):  # 当deltaT正确的时候
        # 坐标转换更新
        MaOri = Mz(-beta_c) * MoOri
        Ma = np.array(MaOri)

        # 从icf转换至rcf和ocf
        rrcf = Mo.dot(r)  # 坐标,rcf
        # xrcf, yrcf, zrcf = Mo.dot(np.array([rrcf[0], r[1], rrcf[2]]))
        Vrcf = Mo.dot(V)  # 速度,rcf
        # Vxrcf, Vyrcf, Vzrcf = Mo.dot(np.array([Vx, Vy, Vz]))
        rocf = Ma.dot(r)  # 坐标,ocf
        # xocf, yocf, zocf = Ma.dot(np.array([rrcf[0], r[1], rrcf[2]]))
        Vocf = Ma.dot(V)  # 速度,ocf
        # Vxocf, Vyocf, Vzocf = Ma.dot(np.array([Vx, Vy, Vz]))
        gocf_0 = Ma.dot(g)  # 初始重力加速度,ocf
        # gxocf_0, gyocf_0, gzocf_0 = Ma * np.array([gx, gy, gz])

        f = beta_c - omega  # 更新入轨点状态
        # 最终点坐标,ocf
        rocf_c = np.array([0, a * (1 - e**2) / (1 + e * cos(f)), 0])
        # xocf_c = 0
        # yocf_c = a * (1 - e**2) / (1 + e * cos(f))
        # zocf_c = 0

        # 最终点速度,ocf
        Vocf_c = np.array([
            sqrt(mu / (a * (1 - e**2))) / rocf_c[1],
            e * sin(f) * sqrt(mu / (a * (1 - e**2))), 0
        ])
        # Vxocf_c = sqrt(mu / (a * (1 - e**2))) / yocf_c
        # Vyocf_c = e * sin(f) * sqrt(mu / (a * (1 - e**2)))
        # Vzocf_c = 0

        # 最终点重力加速度,ocf
        gocf_c = np.array([0, -mu / norm(rocf_c)**2, 0])
        # rocf = sqrt(xocf_c**2 + yocf_c**2 + zocf_c**2)
        # gxocf_c = 0
        # gyocf_c = -mu / rocf**2
        # gzocf_c = 0

        gocf = 0.5 * (gocf_c + gocf_0)  # 中途重力加速度,ocf
        # gxocf, gyocf, gzocf = 0.5 * (gocf_c + gocf_0)

        tc -= deltaT
        # 控制模块
        # phi 俯仰角
        # a part
        r_star = sqrt(rrcf[0]**2 + rrcf[1]**2)
        beta_e = asin(rrcf[0] / r_star)
        V_star = sqrt(Vrcf[0]**2 + Vrcf[1]**2)
        cos_theta_H_star = abs(rrcf[0] * Vrcf[1] -
                               rrcf[1] * Vrcf[0]) / (r_star * V_star)

        # b part
        tau = m / m_dt
        Ibrcf = Mo.dot(Ib)
        tau_o = Ve / (norm(Ibrcf[1:2]) * (F / m))

        count = 0
        # c part
        A1 = Ve * log(tau_o / (tau_o - tc))
        A2 = A1 * tau_o - Ve * tc
        A3 = A1 * tc - A2
        A4 = A3 * tau_o - 0.5 * Ve * tc**2

        # d part
        beta_t = (V_star * tc * cos_theta_H_star + A3 * cos_theta_HC) / rocf[1]
        beta_c = beta_e + beta_t

        while (abs(tc - tc_pre) > epsilon):
            # f part
            tc_pre = tc
            deltaV = sqrt((Vocf_c[0] - Vocf[0] - gocf[0] * tc)**2 +
                          (Vocf_c[1] - Vocf[1] - gocf[1] * tc)**2 +
                          (Vocf_c[2] - Vocf[2] - gocf[2] * tc)**2)
            tc = tau * (1 - exp(-deltaV / Ve))
            count += 1
            if count > 500:
                raise Exception('错误')
        # g part
        phi_ocf_wave = atan((rocf_c[1] - Vocf[1] - gocf[1]) /
                            (rocf_c[0] - Vocf[0] - gocf[0] * tc))
        P = A3 * cos(phi_ocf_wave)
        Q = A4 * cos(phi_ocf_wave)
        R = rocf_c[1] - rocf[
            1] - Vocf[1] * tc - 0.5 * gocf[1] * tc**2 - A3 * sin(phi_ocf_wave)
        delta_k = A1 * Q - A2 * P
        k1 = A2 * R / delta_k
        k2 = A1 * R / delta_k

        # psi 偏航角
        B1 = Ve * log(tau / (tau - tc))
        B2 = B1 * tau - Ve * tc
        B3 = B1 * tc - B2
        B4 = B3 * tau - 0.5 * Ve * tc**2

        psi_ocf_wave = asin((rocf_c[2] + Vocf[2] + gocf[2] * tc) / B1)
        E = B3 * cos(psi_ocf_wave)
        G = B4 * cos(psi_ocf_wave)
        H = rocf_c[2] - rocf[
            2] - Vocf[2] * tc - 0.5 * gocf[2] * tc**2 + B3 * sin(psi_ocf_wave)
        delta_e = B2 * E - B1 * G
        e1 = B2 * H / delta_e
        e2 = B1 * H / delta_e

    phi_ocf = phi_ocf_wave + k2 * t - k1
    psi_ocf = psi_ocf_wave + e2 * t - e1
    MaInv = np.array(MaOri.I)
    Ibicf = MaInv.dot(phipsi(phi_ocf, psi_ocf))
    psi_icf = asin(-Ibicf[2])
    sinPhi = Ibicf[1] / cos(psi_icf)
    if sinPhi >= 0:
        phi_icf = acos(Ibicf[0] / cos(psi_icf))
    else:
        phi_icf = -acos(Ibicf[0] / cos(psi_icf))

    # 算完之后该做的更新
    W = F / m * Ibicf + g
    V += W * dt
    r += V * dt
    m -= m_dt * dt
    g = -mu / norm(r)**3 * r  # 更新重力加速度
    t += dt

    # 记录
    rs.append(r)
    Vs.append(V)
    Ws.append(W)
    if tc < 5:
        break
    elif norm(r) < Re:
        print('坠毁')
        print('t=',t,'tc=',tc)
        break