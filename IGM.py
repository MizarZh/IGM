import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
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
m = 102650  # 质量
# mf # 燃料质量
m_dt = 200
Ve = F / m_dt  # 发动机喷气速度、特征速度

# 轨道参数
a = 6922785.34  # 半长轴
e = 0  # 偏心率
i = radians(90)  # 倾角
Omega = radians(-9.964)  # 升交点赤经
omega = radians(0)  # 近地点角距
f = radians(0)  # 入轨点的真近点角
cos_theta_HC = radians(0)  # 目标点弹道倾角的余弦值  试一个近圆轨道

# 发射点
BT = radians(41.1906)  # 发射点地理纬度
AT = asin(cos(i) / cos(BT))  # 发射方位角
deltaLambda = radians(2.964)  # 升交点与发射点经度差


# 三种旋转矩阵
def Mx(A):
    return np.matrix([[1, 0, 0], [0, cos(A), sin(A)], [0, -sin(A), cos(A)]])


def My(B):
    return np.matrix([[cos(B), 0, -sin(B)], [0, 1, 0], [sin(B), 0, cos(B)]])


def Mz(C):
    return np.matrix([[cos(C), sin(C), 0], [-sin(C), cos(C), 0], [0, 0, 1]])


def phipsi(phi, psi):
    return np.array([cos(phi) * cos(psi), sin(phi) * cos(psi), -sin(psi)])


def F0(t, th):
    return Ve * log(th / (th - t))


def F1(t, th):
    return th * F0(t, th) - Ve * t


def F2(t, th):
    return F0(t, th) * t - F1(t, th)


def F3(t, th):
    return F2(t, th) * th - 0.5 * Ve * t**2


def F0_orbit(t, tho):
    return Ve * log(tho / (tho - t))


def F1_orbit(t, tho):
    return F0_orbit(t, tho) * tho - Ve * t


def F2_orbit(t, tho):
    return F0_orbit(t, tho) * t - F1_orbit(t, tho)


class moveAvg:
    def __init__(self, array, lim):
        self.array = array
        self.lim = lim
        self.sum = sum(array)
        self.n = len(array)

    def append(self, num):
        if self.n < self.lim:
            self.array.append(num)
            self.n += 1
            return sum(self.array) / self.n
        else:
            self.array.append(num)
            del self.array[0]
            return (sum(self.array) - num) / (self.n - 1)

    def avg(self):
        return (sum(self.array) - self.array[-1]) / (self.n - 1)


MoOri = My(i) * Mz(-deltaLambda) * My(-pi / 2) * Mz(BT) * My(AT)  # icf->rcf

beta_c = f + omega
MaOri = Mz(-beta_c) * MoOri  # icf->ocf

Mo = np.array(MoOri)
Ma = np.array(MaOri)

# 初始位置,icf
r = np.array([2.9058E5, 92126 + Re, 0])

# 初始速度,icf
V = np.array([2815.75, 1494.703, 70.953])

# # 初始位置,icf
# r = np.array([0, 100 + Re, 0])

# # 初始速度,icf
# V = np.array([466, 0, 0], dtype=float)

g = -mu / norm(r)**3 * r  # 初始重力加速度,icf

phi0 = radians(61.3383)  # 初始俯仰角,icf
psi0 = radians(0.3105)  # 初始偏航角,icf
Ib = phipsi(phi0, psi0)

t = 0
dt = 0.01
deltaT = 0.5  # 运算间隔
epsilon = 0.001  # 判断tc迭代相差的误差值
tc = 10  # tc初始值
tc_pre = tc + 1
file = open('IGM.txt', 'w')

rs = []
Vs = []
Ws = []
xs = []
ys = []
zs = []
vxs = []
vys = []
vzs = []
Wxs = []
Wys = []
Wzs = []
xocfs = []
yocfs = []
zocfs = []
vxocfs = []
vyocfs = []
vzocfs = []
rocfs = []
rrcfs = []
Vrcfs = []
phis = []
psis = []
lim = 20
phi50 = moveAvg([], lim)
psi50 = moveAvg([], lim)
unstableFlag = 0
psi20s = 0
phi10s = 0
countX = -1
while (True):
    if round(t / deltaT, 3) == round(t / deltaT):  # 当deltaT正确的时候
        countX += 1
        # 坐标转换更新
        MaOri = Mz(-beta_c) * MoOri
        Ma = np.array(MaOri)

        # 从icf转换至rcf和ocf
        rrcf = Mo.dot(r)  # 坐标,rcf
        Vrcf = Mo.dot(V)  # 速度,rcf
        rocf = Ma.dot(r)  # 坐标,ocf
        Vocf = Ma.dot(V)  # 速度,ocf
        gocf_0 = Ma.dot(g)  # 初始重力加速度,ocf
        #         rrcf = r
        #         Vrcf = V
        #         rocf = np.array(Mz(-beta_c)).dot(rrcf)
        #         Vocf = np.array(Mz(-beta_c)).dot(Vrcf)
        #         gocf_0 = np.array(Mz(-beta_c)).dot(g)

        f = beta_c - omega  # 更新入轨点状态
        # 最终点坐标,ocf
        rocf_c = np.array([0, a * (1 - e**2) / (1 + e * cos(f)), 0])

        # 最终点速度,ocf
        Vocf_c = np.array([(1 + e * cos(f)) * sqrt(mu / (a * (1 - e**2))),
                           e * sin(f) * sqrt(mu / (a * (1 - e**2))), 0])

        # 最终点重力加速度,ocf
        gocf_c = np.array([0, -mu / norm(rocf_c)**2, 0])

        gocf = 0.5 * (gocf_c + gocf_0)  # 中途重力加速度,ocf

        # 控制模块
        # phi 俯仰角
        # a part
        r_star = sqrt(rrcf[0]**2 + rrcf[1]**2)
        beta_e = asin(rrcf[0] / r_star)
        V_star = sqrt(Vrcf[0]**2 + Vrcf[1]**2)
        cos_theta_H_star = abs(rrcf[0] * Vrcf[1] -
                               rrcf[1] * Vrcf[0]) / (r_star * V_star)

        # b part
        Ibrcf = Mo.dot(Ib)
        tau = Ve / (norm(Ibrcf) * (F / m))
        tau_o = Ve / (norm(Ibrcf[1:2]) * (F / m))

        count = 0
        # c part
        A1 = Ve * log(tau_o / (tau_o - tc))
        A2 = A1 * tau_o - Ve * tc
        A3 = A1 * tc - A2
        A4 = A3 * tau_o - 0.5 * Ve * tc**2

        # d part

        while (abs(tc - tc_pre) > epsilon):
            # f part
            tc_pre = tc
            deltaV = sqrt((Vocf_c[0] - Vocf[0] - gocf[0] * tc)**2 +
                          (Vocf_c[1] - Vocf[1] - gocf[1] * tc)**2 +
                          (Vocf_c[2] - Vocf[2] - gocf[2] * tc)**2)
            tc = tau * (1 - exp(-deltaV / Ve))
            #             print(deltaV, tc, tau * (1 - exp(-deltaV / Ve)))
            count += 1
            if count > 500:
                raise Exception('错误')
        # g part
        phi_ocf_wave = atan((Vocf_c[1] - Vocf[1] - gocf[1] * tc) /
                            (Vocf_c[0] - Vocf[0] - gocf[0] * tc))
        psi_ocf_wave = -asin((Vocf_c[2] - Vocf[2] - gocf[2] * tc) / deltaV)
        k1 = (rocf_c[1] - F2(tc, tau) * sin(phi_ocf_wave) -
              0.5 * gocf[1] * tc**2 - Vocf[1] * tc - rocf[1]) / (
                  (-F2(tc, tau) + F3(tc, tau) * F0(tc, tau) / F1(tc, tau)) *
                  cos(phi_ocf_wave))
        k2 = (rocf_c[1] - F2(tc, tau) * sin(phi_ocf_wave) -
              0.5 * gocf[1] * tc**2 - Vocf[1] * tc - rocf[1]) / (
                  (-F2(tc, tau) * F1(tc, tau) / F0(tc, tau) + F3(tc, tau)) *
                  cos(phi_ocf_wave))

        e1 = (rocf_c[2] - 0.5 * gocf[2] * tc**2 - Vocf[2] * tc - rocf[2] +
              F2(tc, tau) * sin(psi_ocf_wave)) / (
                  (F2(tc, tau) - F3(tc, tau) * F0(tc, tau) / F1(tc, tau)) *
                  cos(psi_ocf_wave))

        e2 = (rocf_c[2] - 0.5 * gocf[2] * tc**2 - Vocf[2] * tc - rocf[2] +
              F2(tc, tau) * sin(psi_ocf_wave)) / (
                  (F2(tc, tau) * F1(tc, tau) / F0(tc, tau) - F3(tc, tau)) *
                  cos(psi_ocf_wave))
        #         print(phi_ocf_wave,k1,k2,psi_ocf_wave,e1,e2)
        k = (V_star * (cos_theta_H_star - 1) +
             F2_orbit(tc, tau_o) * cos_theta_HC - F2(tc, tau)) / tc**2
        #         beta_t = cos_theta_HC * (norm(V) * tc + F2(tc, tau) - k * tc *
        #                                  (norm(V) + deltaV - norm(Vocf_c))) / rocf_c[1]
        beta_t = (V_star * tc * cos_theta_H_star +
                  A3 * cos_theta_HC) / rocf_c[1]
        beta_c = beta_e + beta_t
        tc -= deltaT
        rocfs.append(rocf.copy())
#         print(phi_ocf_wave,psi_ocf_wave,k1,k2,e1,e2)

#     if tc <= 10:
#         if phi10s == 0:
#             phi10s = phi_ocf_wave
#         phi_ocf = phi10s
#         psi_ocf = psi20s
#     elif tc <= 20:
#         if psi20s == 0:
#             psi20s = psi_ocf_wave
#         phi_ocf = phi_ocf_wave
#         psi_ocf = psi20s
#     elif tc <= 40:
#         phi_ocf = phi_ocf_wave
#         psi_ocf = psi_ocf_wave
#     elif tc <= 60:
#         phi_ocf = phi_ocf_wave - k2 * t - k1
#         psi_ocf = psi_ocf_wave
#      - countX*deltaT
    k2 /= 100
    e2 /= 100
    phi_ocf = phi_ocf_wave + k2 * (t) - k1
    psi_ocf = psi_ocf_wave + e2 * (t) - e1
    MaInv = np.array(MaOri.I)
    Ibicf = MaInv.dot(phipsi(phi_ocf, psi_ocf))
    psi_icf = asin(-Ibicf[2])
    sinPhi = Ibicf[1] / cos(psi_icf)
    if sinPhi >= 0:
        phi_icf = acos(Ibicf[0] / cos(psi_icf))
    else:
        phi_icf = -acos(Ibicf[0] / cos(psi_icf))

#     if unstableFlag == 0:
#         phi_ocf = phi_ocf_wave + k2 * t - k1
#         psi_ocf = psi_ocf_wave + e2 * t - e1
#     else:
#         phi_ocf = phi50.avg()
#         psi_ocf = psi50.avg()
#         deltaT = tc
#     MaInv = np.array(MaOri.I)
#     Ibicf = MaInv.dot(phipsi(phi_ocf, psi_ocf))
#     psi_icf = asin(-Ibicf[2])
#     sinPhi = Ibicf[1] / cos(psi_icf)
#     if sinPhi >= 0:
#         phi_icf = acos(Ibicf[0] / cos(psi_icf))
#     else:
#         phi_icf = -acos(Ibicf[0] / cos(psi_icf))
#     if phi50.n >= lim:
#         phiAvg = phi50.append(phi_icf)
#         psiAvg = psi50.append(psi_icf)
#         if degrees(abs(phi_icf - phiAvg))> 40 or degrees(abs(psi_icf - psiAvg)) > 40:
#             unstableFlag = 1
#     else:
#         phi50.append(phi_icf)
#         psi50.append(psi_icf)

#         phi_icf = 0
#         psi_icf = 0
# 算完之后该做的更新
    W = F / m * Ibicf + g
    V += W * dt
    r += V * dt
    #     print(W,V,r)
    m -= m_dt * dt
    g = -mu / norm(r)**3 * r  # 更新重力加速度
    t += dt
    # 记录
    #     plt.scatter(r[0],r[1], c='blue')
    xs.append(r[0])
    ys.append(r[1])
    zs.append(r[2])

    vxs.append(V[0])
    vys.append(V[1])
    vzs.append(V[2])

    Wxs.append(W[0])
    Wys.append(W[1])
    Wzs.append(W[2])

    xocfs.append(rocf[0])
    yocfs.append(rocf[1])
    zocfs.append(rocf[2])

    vxocfs.append(Vocf[0])
    vyocfs.append(Vocf[1])
    vzocfs.append(Vocf[2])

    rs.append(r.copy())
    Vs.append(V.copy())
    Ws.append(W.copy())
    rrcfs.append(rrcf.copy())
    Vrcfs.append(Vrcf.copy())
    phis.append(phi_icf)
    psis.append(psi_icf)
    file.write('%f,%f,%f\n' % (r[0], r[1], r[2]))
    if tc < 0.001:
        break
    elif norm(r) < Re:
        print('坠毁')
        print('t=', t, 'tc=', tc)
        break

# plt.figure(dpi=200, figsize=[6,6])
# ax1 = plt.axes(projection='3d')
# ax1.plot3D(xs,ys,zs)

plt.figure(dpi=200, figsize=[4, 4])
angle = np.linspace(0, 2 * math.pi, 10000)
plt.plot(Re * np.cos(angle), Re * np.sin(angle))
plt.plot(xs, ys)
m

# 轨道维持
file2 = open('IGMOrbit.txt', 'w')
newxs = []
newys = []
newzs = []
newdt = 0.1
for i in np.arange(50000):
    W = g
    V += W * newdt
    r += V * newdt
    g = -mu / norm(r)**3 * r  # 更新重力加速度
    newxs.append(r[0])
    newys.append(r[1])
    newzs.append(r[2])
    file2.write('%f,%f,%f\n' % (r[0], r[1], r[2]))
    if norm(r) < Re:
        print('坠毁')
        break

# 误差分析
np.set_printoptions(suppress=True)
print(Vocf, Vocf_c, Vocf - Vocf_c)
print(rocf, rocf_c, rocf - rocf_c)
Mi = np.array(
    Mz(pi / 2 - (-deltaLambda + Omega)) * My(pi / 2) * Mz(BT) * My(AT))
r_ = Mi.dot(r)
v_ = Mi.dot(V)


def calOrbitPara(r, v):
    x = np.array([1, 0, 0])  # 地心赤道惯性坐标系中指向春分点的坐标轴单位矢量
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])  # 地心赤道惯性坐标系中指向北极的坐标轴单位矢量

    h = np.cross(r, v)  # 卫星相对于地心的动量矩
    norm_h = norm(h)
    n = np.cross(z, h)  # 升交点地心单位矢量
    norm_n = norm(n)

    e = ((norm(v)**2 - mu / norm(r)) * r -
         (r.dot(v)) * v) / mu  # 矢量方向为从焦点指向近拱点
    norm_e = norm(e)  # 偏心率（离心率）

    # E = norm(v)^2/2 - mu/norm(r)   # 机械能
    # a = -mu/E/2       # 机械能直接决定了轨道的半长轴
    a = norm_h**2 / mu / (1 - norm_e**2)
    i = acos(z.dot(h) / norm_h)  # 轨道倾角

    if y.dot(n) >= 0:
        Omega = acos(x.dot(n) / norm_n)  # 升交点赤经
    else:
        Omega = -acos(x.dot(n) / norm_n)

    if z.dot(e) >= 0:
        omega = acos(e.dot(n) / norm_n / norm_e)  # 近地点幅角
    else:
        omega = -acos(e.dot(n) / norm_n / norm_e)

    norm_r = norm(r)
    if r.dot(v) >= 0:
        f = acos(r.dot(e) / norm_r / norm_e)  # 真近点角
    else:
        f = -acos(r.dot(e) / norm_r / norm_e)

    return [a, norm_e, i, Omega, omega, f]


ah, eh, ih, Omegah, omegah, fh = calOrbitPara(r_, v_)
# print(a,e,i,Omega,omega,f)
# print(ah,eh,ih,Omegah,omegah,fh)
# print(ah-a,eh-e,ih-i,Omegah-Omega,omegah-omega,fh-f)
print((ah - a) / a, eh - e, (ih - i) / i, (Omegah - Omega) / Omega,
      omegah - omega, (fh - f) / f)

# 参数图
ts = np.arange(0, t, 0.01)

plt.figure(dpi=200)
plt.scatter(ts, np.degrees(phis), s=0.01)
