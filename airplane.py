import gym
from gym import spaces
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
import torch

rho = 1.2
g = 9.81

target_altitude = 1500
target_airspeed = 25
pitch_target = 0.12
max_altitude = 2000
min_altitude = 0
max_airspeed = 30
min_airspeed = 15
max_pitch = 0.4
min_pitch = -0.4

dt = 0.001

t_graph = []
H_graph = []
L_graph = []
V_graph = []
pitch_graph = []
alpha_graph = []
q_graph = []
X_graph = []
Z_graph = []
M_graph = []
de_graph = []
dp_graph = []


class Plane:
    def __init__(self, m, Jy, Cx, Cz, S, ba, m0, ma, mde, mw, kt, H, V, pitch):
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

        self.t = 0

        self.reset(H=H, V=V, pitch=pitch)

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

        t_graph.append(self.t)
        H_graph.append(self.H)
        L_graph.append(self.L)
        V_graph.append(self.V)
        pitch_graph.append(self.pitch)
        alpha_graph.append(self.alpha)
        q_graph.append(self.q)
        X_graph.append(self.X)
        Z_graph.append(self.Z)
        M_graph.append(self.M)
        de_graph.append(de)
        dp_graph.append(dp)
        self.t += dt

    def reset(self, H, V, pitch):
        self.H = H
        self.L = 0
        self.V = V
        self.a = 0
        self.pitch = pitch
        self.alpha = pitch
        self.theta = 0
        self.q = 0

        self.X = 0
        self.Z = 0
        self.M = 0


class PlaneEnv(gym.Env):
    def __init__(self):
        super(PlaneEnv, self).__init__()
        self.observation_space = gym.spaces.Box(
            # low=np.array([min_pitch, min_altitude, min_airspeed]),
            # high=np.array([max_pitch, max_altitude, max_airspeed]),
            low=np.array([min_pitch]),
            high=np.array([max_pitch]),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-0.3, 0.29]), high=np.array([0.3, 0.31]), dtype=np.float32
        )
        self.airplane = Plane(
            m=1.5,
            Jy=0.1,
            Cx=-0.1,
            Cz=2,
            m0=-0.05,
            mw=-0.01,
            ma=-0.1,
            mde=0.5,
            S=0.3,
            ba=1.2,
            kt=10,
            H=1000,
            V=20,
            pitch=0.0,
        )

    def reset(self):
        self.airplane.reset(H=1000, V=20, pitch=0.05)
        # return self.airplane.pitch, self.airplane.H, self.airplane.V
        return self.airplane.pitch

    def step(self, action):
        action = np.clip(action, -1, 1)
        de, dp = action
        self.airplane.update(de, dp)
        # new_state = self.airplane.pitch, self.airplane.H, self.airplane.V
        new_state = self.airplane.pitch

        # altitude_reward = 0 - 2 * abs(target_altitude - self.airplane.H) / max_altitude
        # airspeed_reward = 0 - 0.5 * abs(target_airspeed - self.airplane.V) / max_airspeed
        # pitch_reward = 0 - 0.1 * abs(self.airplane.pitch)
        # reward = altitude_reward + airspeed_reward + pitch_reward
        pitch_reward = 1.0 - abs(pitch_target - self.airplane.pitch) / max_pitch
        reward = pitch_reward

        done = False
        # if (
        #     abs(target_altitude - self.airplane.H) < 1
        #     and abs(target_airspeed - self.airplane.V) < 0.5
        # ):
        if (
            self.airplane.H >= max_altitude - 1
            or self.airplane.H <= min_altitude + 1
            or self.airplane.V >= max_airspeed - 1
            or self.airplane.V <= min_airspeed
            or abs(self.airplane.pitch) >= 0.4
        ):
            reward = -1
            done = True
        print(reward)
        return new_state, reward, done, {}

    def render(self, mode="human"):
        # Optional: Implement rendering for visualization
        pass


# Instantiate and wrap the environment
env = PlaneEnv()
env = DummyVecEnv([lambda: env])

# Instantiate the DDPG agent
action_noise = NormalActionNoise(
    mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape)
)
model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=0.01,
    gamma=0.1,
    action_noise=action_noise,
    verbose=1,
)

# Train the agent
num_steps = 1000
model.learn(total_timesteps=num_steps)
print("Training completed!")

# Test the trained agent
obs = env.reset()
num_test_steps = 1000

# for step in range(num_test_steps):
#     action, _ = model.predict(obs)
#     obs, reward, done, _ = env.step(action)
#     print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Done: {done}")
#     if done:
#         obs = env.reset()

# env.close()

fig, axs = plt.subplots(2, 4)

axs[0, 0].set_title("Altitude, Longitude, m")
axs[0, 0].plot(L_graph, H_graph)
axs[0, 0].grid()

axs[0, 1].set_title("Airspeed, m/s")
axs[0, 1].plot(t_graph, V_graph)
axs[0, 1].grid()

axs[0, 2].set_title("Forces, N")
axs[0, 2].plot(t_graph, X_graph, label="X")
axs[0, 2].plot(t_graph, Z_graph, label="Z")
axs[0, 2].grid()
axs[0, 2].legend()

axs[0, 3].plot(t_graph, dp_graph)
axs[0, 3].set_title("Throttle")
axs[0, 3].grid()

axs[1, 0].set_title("Angles, rad")
axs[1, 0].plot(t_graph, pitch_graph, label="pitch")
axs[1, 0].plot(t_graph, alpha_graph, label="alpha")
axs[1, 0].grid()
axs[1, 0].legend()

axs[1, 1].plot(t_graph, q_graph)
axs[1, 1].set_title("Angular rate, rad/s")
axs[1, 1].grid()

axs[1, 2].plot(t_graph, M_graph)
axs[1, 2].set_title("Moment, N*m")
axs[1, 2].grid()

axs[1, 3].plot(t_graph, de_graph)
axs[1, 3].set_title("Elevator")
axs[1, 3].grid()

plt.show()
