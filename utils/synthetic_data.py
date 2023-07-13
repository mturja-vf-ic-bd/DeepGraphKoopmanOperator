import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np
import quantities as pq
from scipy.integrate import odeint


def integrated_lorenz(dt, num_steps, x0=0, y0=1, z0=1.05,
                      sigma=10, rho=28, beta=2.667, tau=1e3):
    """

    Parameters
    ----------
    dt :
        Integration time step in ms.
    num_steps : int
        Number of integration steps -> max_time = dt*(num_steps-1).
    x0, y0, z0 : float
        Initial values in three dimensional space
    sigma, rho, beta : float
        Parameters defining the lorenz attractor
    tau : characteristic timescale in ms

    Returns
    -------
    t : (num_steps) np.ndarray
        Array of timepoints
    (3, num_steps) np.ndarray
        Integrated three-dimensional trajectory (x, y, z) of the Lorenz attractor
    """

    def _lorenz_ode(point_of_interest, timepoint, sigma, rho, beta, tau):
        """
        Fit the model with `spiketrains` data and apply the dimensionality
        reduction on `spiketrains`.

        Parameters
        ----------
        point_of_interest : tuple
            Tupel containing coordinates (x,y,z) in three dimensional space.
        timepoint : a point of interest in time
        dt :
            Integration time step in ms.
        num_steps : int
            Number of integration steps -> max_time = dt*(num_steps-1).
        sigma, rho, beta : float
            Parameters defining the lorenz attractor
        tau : characteristic timescale in ms

        Returns
        -------
        x_dot, y_dot, z_dot : float
            Values of the lorenz attractor's partial derivatives
            at the point x, y, z.
        """

        x, y, z = point_of_interest
        x_dot = (sigma * (y - x)) / tau
        y_dot = (rho * x - y - x * z) / tau
        z_dot = (x * y - beta * z) / tau
        return x_dot, y_dot, z_dot

    assert isinstance(num_steps, int), "num_steps has to be integer"

    t = dt * np.arange(num_steps)
    poi = (x0, y0, z0)
    return t, odeint(_lorenz_ode, poi, t, args=(sigma, rho, beta, tau)).T


def random_projection(data, embedding_dimension, loc=0, scale=None):
    """
    Parameters
    ----------
    data : np.ndarray
        Data to embed, shape=(M, N)
    embedding_dimension : int
        Embedding dimension, dimensionality of the space to project to.
    loc : float or array_like of floats
        Mean (“centre”) of the distribution.
    scale : float or array_like of floats
        Standard deviation (spread or “width”) of the distribution.

    Returns
    -------
    np.ndarray
       Random (normal) projection of input data, shape=(dim, N)

    See Also
    --------
    np.random.normal()

    """
    if scale is None:
        scale = 1 / np.sqrt(data.shape[0])
    projection_matrix = np.random.normal(loc, scale, (embedding_dimension, data.shape[0]))
    return np.dot(projection_matrix, data)


def generate_lorentz_Ndim(Ndim, trial):
    # set parameters for the integration of the Lorentz attractor
    timestep = 1 * pq.ms
    transient_duration = 10 * pq.s
    trial_duration = 20 * pq.s
    num_steps_transient = int((transient_duration.rescale('ms') / timestep).magnitude)
    num_steps = int((trial_duration.rescale('ms') / timestep).magnitude)

    # set parameters for spike train generation
    np.random.seed(42)  # for visualization purposes, we want to get identical spike trains at any run

    data3d = np.zeros((trial, 3, num_steps_transient + num_steps))
    dataNd = np.zeros((trial, Ndim, num_steps))
    for i in range(trial):
        # calculate the oscillator
        times, lorentz_trajectory_3dim = integrated_lorenz(
            timestep,
            num_steps=num_steps_transient + num_steps,
            x0=0, y0=1, z0=1.25)

        # random projection
        lorentz_trajectory_Ndim = random_projection(
            lorentz_trajectory_3dim[:, num_steps_transient:],
            embedding_dimension=Ndim)
        data3d[i] = lorentz_trajectory_3dim
        dataNd[i] = lorentz_trajectory_Ndim
    return data3d, dataNd


def draw_from_inhomogeneous_gaussian(mu, sigma, duration, interpolation):
    """

    :param mu: numpy array [[mu_lower, mu_upper], ...] where mu_lower is the lowest possible value of mu
    and mu_upper is the highest possible value of mu
    :param sigma: numpy array [[sigma_lower, sigma_upper], ...] where sigma_upper is the lowest possible value of sigma
    and sigma_upper is the highest possible value of sigma
    :param duration: number os samples
    :param interpolation: random/linear. If random, mu is drawn from uniform(mu_lower, mu_upper) for each time.
    If linear, mu is linearly interpolated between (mu_lower, mu_upper).
    :return:
    """
    data = np.zeros((mu.shape[0], duration))
    for i in range(mu.shape[0]):
        if interpolation == 'random':
            data[i] = np.random.normal(np.random.uniform(mu[i, 0], mu[i, 1], duration),
                                       np.random.uniform(sigma[i, 0], sigma[i, 1]))
        else:
            data[i] = np.random.normal(np.linspace(mu[i, 0], mu[i, 1], duration),
                                       np.random.uniform(sigma[i, 0], sigma[i, 1]))
    return data


def draw_from_sequential_inhomogeneous_gaussian(param_list: list):
    """
    :param param_list: A list containing dictionaries of paramters.
    For each of these paramters, samples are drawn from inhomogeneous gaussian
    and they are concatenated to generate a single time series.
    :return: simulated time series
    """
    sim_data = []
    for param in param_list:
        data = draw_from_inhomogeneous_gaussian(
            param["mu"],
            param["sigma"],
            param["duration"],
            param["interpolation"]
        )
        sim_data.append(data)
    return np.concatenate(sim_data, axis=1)


def spikefy_data(lmd):
    pi = 1 - np.exp(-lmd)
    h = bernoulli.rvs(pi)
    return h


def generate_inhomogeneous_gaussian(num_nodes, trial=1):
    param_set_1 = {"mu": np.array([[0.1, 0.4], [0.3, 0.7], [0, 0.1], [0.8, 1]]),
                   "sigma": np.array([[0.01, 0.1], [0.03, 0.06], [0, 0.2], [0.02, 0.03]]),
                   "duration": 1000,
                   "interpolation": 'random'}
    param_set_2 = {"mu": np.array([[0.4, 0.6], [0.7, 0.6], [0.1, 0.3], [1, 0.7]]),
                   "sigma": np.array([[0.01, 0.1], [0.03, 0.05], [0, 0.1], [0.02, 0.04]]),
                   "duration": 20,
                   "interpolation": 'linear'}
    param_set_3 = {"mu": np.array([[0.6, 0.3], [0.6, 0.4], [0.3, 0.2], [0.7, 0.8]]),
                   "sigma": np.array([[0.01, 0.1], [0.03, 0.05], [0, 0.02], [0.02, 0.04]]),
                   "duration": 20,
                   "interpolation": 'linear'}
    param_set_4 = {"mu": np.array([[0.3, 0.5], [0.4, 0.5], [0.2, 0.4], [0.8, 0.9]]),
                   "sigma": np.array([[0.05, 0.08], [0.04, 0.07], [0.1, 0.2], [0.01, 0.07]]),
                   "duration": 1000,
                   "interpolation": 'random'}
    param_set_5 = {"mu": np.array([[0.5, 0.6], [0.5, 0.3], [0.4, 0.3], [0.9, 0.8]]),
                   "sigma": np.array([[0.05, 0.07], [0.01, 0.05], [0.01, 0.05], [0.03, 0.04]]),
                   "duration": 20,
                   "interpolation": 'linear'}
    param_set_6 = {"mu": np.array([[0.6, 0.7], [0.3, 0.45], [0.3, 0.5], [0.8, 0.1]]),
                   "sigma": np.array([[0.01, 0.1], [0.01, 0.2], [0.01, 0.1], [0.01, 0.06]]),
                   "duration": 1000,
                   "interpolation": 'random'}
    param_list = [param_set_1, param_set_2, param_set_3, param_set_4, param_set_5, param_set_6]

    ts_4d = np.zeros((trial, 4, param_set_1["duration"] +
                      param_set_2["duration"] +
                      param_set_3["duration"] +
                      param_set_4["duration"] +
                      param_set_5["duration"] +
                      param_set_6["duration"]))
    ts_nd = np.zeros((trial, num_nodes, param_set_1["duration"] +
                      param_set_2["duration"] +
                      param_set_3["duration"] +
                      param_set_4["duration"] +
                      param_set_5["duration"] +
                      param_set_6["duration"]))
    for i in range(trial):
        ts = draw_from_sequential_inhomogeneous_gaussian(param_list)
        ts_proj = random_projection(ts, num_nodes)
        ts_4d[i] = ts
        ts_nd[i] = ts_proj
    return ts_4d, ts_nd


def kura_ode(cur_theta, t, nat_freq, filter_orders, Kvals, splits, Kfunc):
    """
    Generate phase trajectories of coupled oscillator system following kuramoto model.

    :param cur_theta: numpy array (N,) containing current phase state of N oscillators.
    :param t: time point
    :param nat_freq: numpy array (N,) containing initial frequesncy of N oscillators.
    :param filter_orders: filter orders to control how many neighbors to consider
    :param Kvals: set of coupling strengths
    :param splits: timepoints where coupling strength changes
    :param Kfunc: function to estimate coupling strength K.
    :return: derivative at current time point
    """

    filter = getFilter(t, len(cur_theta), filter_orders, splits)
    coupling_mat = filter * np.sin(cur_theta[:, np.newaxis] - cur_theta[np.newaxis, :])
    K = Kfunc(t, Kvals, splits)
    dydt = nat_freq + K * coupling_mat.mean(axis=0)
    return dydt


def create_diagonal_filter(N, order=1):
    filter = np.diag(np.ones(N))
    for i in range(1, order+1):
        filter += np.diag(np.ones(N - i), k=i)
        filter += np.diag(np.ones(N - i), k=-i)
    return filter


def getFilter(t, N, orders, splits):
    for i in range(len(splits)):
        if t > splits[i]:
            continue
        return create_diagonal_filter(N, orders[i])


def getK(t, vals, splits):
    """
    Find K value depending on where t falls in the time-segments
    defined by splits
    :param t: value
    :param vals: value of K for corresponding time-segment
    :param splits: right boundaries of time-segments
    :return: scalar value of K
    """
    for i in range(len(splits)):
        if t > splits[i]:
            continue
        return vals[i]


class Simulate_Kuramoto:
    def __init__(self, N, T, n_splits, k_range, freq_range):
        """

        :param N: Number of oscillators
        :param T: Timepoints
        :param K: coupling strength for each time point
        """
        self.N = N
        self.T = T
        self.Kvals = np.linspace(k_range[0], k_range[1], n_splits)
        self.splits = np.linspace(0, T, n_splits)
        self.freq_range = freq_range
        self.filter_orders = np.arange(1, n_splits+1, 1)

        self.nat_freq = self.sample_step_natfreq()

    def sample_uniform_natfreq(self):
        return np.random.uniform(self.freq_range[0], self.freq_range[1], self.N)

    def sample_step_natfreq(self):
        return np.linspace(self.freq_range[0], self.freq_range[1], self.N)

    def simulate(self):
        init_theta = np.random.uniform(-np.pi, np.pi, self.N)
        # init_theta.sort()
        t = np.linspace(0, 1, self.T+1)
        self.t = t
        return odeint(kura_ode, init_theta, t=t, args=(self.nat_freq, self.filter_orders, self.Kvals, self.splits, getK))

    def compute_omega(self, theta):
        """

        :param theta: (T, N) array with values of theta
        :return: omega: (T, N) array of angular frequency
        """
        omega = np.zeros_like(theta)
        for i in range(len(theta)):
            omega[i] = kura_ode(cur_theta=theta[i], t=self.t[i],
                         nat_freq=self.nat_freq, filter_orders=self.filter_orders,
                         Kvals=self.Kvals, splits=self.splits,
                         Kfunc=getK)
        return omega


if __name__ == '__main__':
    # for i in range(5):
    #     assert getK(i, [0,1,2,3,4],[0,1,2,3,4]) == i
    lorentz3d, lorentzNd = generate_lorentz_Ndim(20, 1)
    print(f"lorentz3d dim: {lorentz3d.shape}, lorentzNd dim: {lorentzNd.shape}")

    lorentz3d_spikes = np.exp(lorentz3d)
    lorentz3d_spikes = spikefy_data(lorentz3d)
    print(f"lorentz3d spike dim:{lorentz3d_spikes.shape}")

    ts_4d, ts_nd = generate_inhomogeneous_gaussian(20, 3)
    print(f"inhomogeneous gaussian 4d series dim: {ts_4d.shape}, inhomogeneous gaussian Nd series dim: {ts_nd.shape}")
