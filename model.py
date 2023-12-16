import math
import matplotlib.pyplot as plt

rho = 1.2
g = 9.81
dt = 0.001


class Plane:
    def __init__(self, m, Jy, Cx, Cz, S, ba, m0, ma, mde, mw, kt, H, V):
        self.m = m
        self.Jy = Jy
        self.S = S
        self.ba = ba

        self.Cx = Cx
        self.Cz = Cz
        self.m0 = m0
        self.ma = ma
        self.mde = mde
        self.mw = mw

        self.kt = kt

        self.H = H
        self.L = 0
        self.V = V
        self.a = 0
        self.pitch = 0.05
        self.alpha = 0.05
        self.theta = 0
        self.q = 0

        self.X = 0
        self.Z = 0
        self.M = 0

    def update(self, de, dp):
        q = rho * self.V**2 / 2

        Xa = self.Cx * self.alpha * q * self.S
        Za = self.Cz * self.alpha * q * self.S
        Ma = (
            (self.m0 + self.ma * self.alpha + self.mde * de + self.mw * self.q)
            * q
            * self.S
            * self.ba
        )

        G = self.m * g

        T = self.kt * dp

        self.X = T * math.cos(self.alpha) + Xa - G * math.sin(self.theta)
        self.Z = T * math.sin(self.alpha) + Za - G * math.cos(self.theta)
        self.M = Ma

        self.a = self.X / self.m
        self.V += self.a * dt
        self.theta += self.Z / (self.m * self.V) * dt
        self.q += self.M / self.Jy * dt
        self.pitch += self.q * dt
        self.alpha = self.pitch - self.theta

        self.H += self.V * math.sin(self.theta) * dt
        self.L += self.V * math.cos(self.theta) * dt


p = Plane(
    m=1.5,
    Jy=0.1,
    Cx=-0.1,
    Cz=2,
    m0=-0.05,
    mw=-0.01,
    ma=-0.1,
    mde=0.05,
    S=0.3,
    ba=1.2,
    kt=10,
    H=1000,
    V=20,
)

t = []
H = []
L = []
V = []
pitch = []
alpha = []
q = []
X = []
Z = []
M = []

t_current = 0
t_max = 100

while t_current < t_max:
    p.update((min((1000 - p.H) * 0.1, 0.25) - p.pitch) * (50), (20 - p.V) * 0.09)

    t.append(t_current)
    H.append(p.H)
    L.append(p.L)
    V.append(p.V)
    pitch.append(p.pitch)
    alpha.append(p.alpha)
    q.append(p.q)
    X.append(p.X)
    Z.append(p.Z)
    M.append(p.M)
    t_current += dt


fig, axs = plt.subplots(2, 3)

axs[0, 0].set_title("Altitude, Longitude, m")
axs[0, 0].plot(L, H)
axs[0, 0].grid()

axs[0, 1].set_title("Airspeed, m/s")
axs[0, 1].plot(t, V)
axs[0, 1].grid()

axs[0, 2].set_title("Forces, N")
axs[0, 2].plot(t, X, label="X")
axs[0, 2].plot(t, Z, label="Z")
axs[0, 2].grid()
axs[0, 2].legend()

axs[1, 0].set_title("Angles, rad")
axs[1, 0].plot(t, pitch, label="pitch")
axs[1, 0].plot(t, alpha, label="alpha")
axs[1, 0].grid()
axs[1, 0].legend()

axs[1, 1].plot(t, q)
axs[1, 1].set_title("Angular rate, rad/s")
axs[1, 1].grid()

axs[1, 2].plot(t, M)
axs[1, 2].set_title("Moment, N*m")
axs[1, 2].grid()

plt.show()
