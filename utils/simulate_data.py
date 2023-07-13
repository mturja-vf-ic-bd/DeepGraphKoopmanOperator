import random

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt


class SimplePendulum:
    def __init__(self, m=1, L=1, b=0, h=0, g=9.81):
        """
        Creates an pendulum object with the following physical properties:
        :param m: mass of the bob
        :param L: length of the massless rod
        :param b: damping coefficient
        """
        self.m = m
        self.L = L
        self.b = b
        self.h = h
        self.g = g

    def _int_pendulum_sim(self, theta_init, t):
        theta_dot_1 = theta_init[1]
        theta_dot_2 = -self.b / self.m * theta_init[1] - \
                      self.g / self.L * np.sin(theta_init[0]) - \
                      self.h / self.m * np.cos(theta_init[0])
        return theta_dot_1, theta_dot_2

    def simulate(self, t, pos_init, vel_init):
        return t, integrate.odeint(self._int_pendulum_sim, [pos_init, vel_init], t)

    def plot(self, t, pos_init, vel_init):
        t, state = self.simulate(t, pos_init, vel_init)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, state[:, 0])
        plt.plot(t, state[:, 1])
        plt.subplot(1, 2, 2)
        plt.plot(state[:, 0], state[:, 1])
        plt.show()


class StichedPendulum:
    def __init__(self, params, time_points):
        self.pendulum_systems = []
        for v in params:
            self.pendulum_systems.append(
                SimplePendulum(
                    m=v["m"],
                    L=v["L"],
                    g=v["g"],
                    b=v["b"],
                    h=v["h"]
                )
            )
        self.time_points = time_points

    def simulate(self, pos_init, vel_init):
        states = []
        T = []
        for i, sys in enumerate(self.pendulum_systems):
            t, state = sys.simulate(self.time_points[i], pos_init, vel_init)
            pos_init = state[-1, 0]
            vel_init = state[-1, 1]
            states.append(state)
            T.append(t)
        states = np.concatenate(states, axis=0)
        T = np.concatenate(T, axis=0)
        return T, states

    def plot(self, pos_init, vel_init):
        t, state = self.simulate(pos_init, vel_init)
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, state[:, 0])
        plt.plot(t, state[:, 1])
        plt.subplot(1, 2, 2)
        plt.plot(state[:, 0], state[:, 1])
        plt.tight_layout()
        plt.show()


def generate_random_params(t_max, delta_t, max_event_count, max_force, b=0):
    m = 1
    L = 1
    g = 9.81
    gap = 5
    t_force = 2
    # event_count = random.randint(2, max_event_count)
    event_count = max_event_count
    random_list = [gap * i + x
                   for i, x in
                   enumerate(sorted(random.sample(range(t_force, t_max-t_force - gap * (event_count-1)), event_count)))]
    t_init = 0

    time_points = []
    params = []
    for i, r in enumerate(random_list):
        time_points.append(np.linspace(t_init, r, int((r - t_init) / delta_t)))
        params.append({"m": m, "L": L, "g": g, "b": b, "h": 0})
        time_points.append(np.linspace(r, r+t_force, int(t_force / delta_t)))
        c = random.randint(0, 1)
        force = random.randint(1, max_force)
        if c == 0:
            params.append({"m": m, "L": L, "g": g, "b": b, "h": force})
        else:
            params.append({"m": m, "L": L, "g": g, "b": b, "h": -force})
        t_init = r + t_force
        if t_init >= t_max:
            break
    if t_init < t_max:
        time_points.append(np.linspace(t_init, t_max, int((t_max - t_init) / delta_t)))
    params.append({"m": m, "L": L, "g": g, "b": b, "h": 0})

    return params, time_points


if __name__ == '__main__':
    m = 1
    L = 1
    g = 9.81
    b = 0.00
    delta_t = 0.02
    t_max = 100
    params, time_points = generate_random_params(t_max, delta_t, 6, 2)
    # params = [{"m": m, "L": L, "g": g, "b": b, "h": 0},
    #           {"m": m, "L": L, "g": g, "b": b, "h": 5},
    #           {"m": m, "L": L, "g": g, "b": b, "h": 0},
    #           {"m": m, "L": L, "g": g, "b": b, "h": -5},
    #           {"m": m, "L": L, "g": g, "b": b, "h": 0}]
    #
    # time_points = [np.linspace(0, 10, int(10/delta_t)),
    #                np.linspace(10, 12, int(2/delta_t)),
    #                np.linspace(12, 22, int(10/delta_t)),
    #                np.linspace(22, 24, int(2/delta_t)),
    #                np.linspace(24, 34, int(10/delta_t))]
    sp = StichedPendulum(params, time_points)
    pos_init = 0
    vel_init = 1
    sp.plot(pos_init, vel_init)





