"""Lorenz 63 example."""

import functools
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from loguru import logger
from scipy.spatial.distance import pdist
from sklearn.gaussian_process.kernels import RBF

import linear_operator_learning as lol

# Configure logger
logger.remove()
logger.add(
    sys.stdout,  # Log to standard output
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
    colorize=True,  # Enable colorization
)

this_folder = Path(__file__).parent.resolve()


# Adapted from https://realpython.com/python-timer/#creating-a-python-timer-decorator
def timer(func):
    """A decorator that times the execution of a function.

    Args:
        func: Callable to be timed.

    Returns:
        Tuple containing the output of `func` and the time it took to execute.
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = perf_counter()
        value = func(*args, **kwargs)
        toc = perf_counter()
        elapsed_time = toc - tic
        return value, elapsed_time

    return wrapper_timer


def L63(x: np.ndarray, t: int = 1, sigma=10, mu=28, beta=8 / 3, dt=0.01):
    """Simulates the Lorenz '63 system using given initial conditions and parameters.

    Args:
        x (np.ndarray): Initial state vector of the system.
        t (int, optional): Total number of time steps for the simulation. Default is 1.
        sigma (float, optional): Prandtl number. Default is 10.
        mu (float, optional): Rayleigh number minus 1. Default is 28.
        beta (float, optional): Geometric factor. Default is 8/3.
        dt (float, optional): Time step size. Default is 0.01.

    Returns:
        np.ndarray: Array containing the state of the system at each time step.
    """
    M_lin = np.array([[-sigma, sigma, 0], [mu, 0, 0], [0, 0, -beta]])

    def D(_, x):
        dx = M_lin @ x
        dx[1] -= x[2] * x[0]
        dx[2] += x[0] * x[1]
        return dx

    sim_time = dt * (t + 1)
    t_eval = np.linspace(0, sim_time, t + 1, endpoint=True)
    t_span = (0, t_eval[-1])
    sol = scipy.integrate.solve_ivp(D, t_span, x, t_eval=t_eval, method="RK45")
    return sol.y.T


def make_dataset():
    """Generates a dataset of the Lorenz '63 system, with the given parameters.

    Returns:
        dict: A dictionary containing the train and test datasets.
    """
    train_samples = 10000
    test_samples = 100
    t = train_samples + 1000 + test_samples
    x = np.ones(3)
    raw_data = L63(x, t=t)
    # raw_data = LogisticMap(N=20).sample(X0 = np.ones(1), T=configs.train_samples + 1000 + configs.test_samples)
    mean = np.mean(raw_data, axis=0)
    norm = np.max(np.abs(raw_data), axis=0)
    # Data rescaling
    data = raw_data - mean
    data /= norm

    train_data = data[: train_samples + 1]
    test_data = data[-test_samples - 1 :]
    return {"train": train_data, "test": test_data}


def center_selection(num_pts: int, num_centers: int, rng_seed: int | None = None):
    """Randomly selects a specified number of center indices from a given number of points.

    Args:
        num_pts (int): Total number of available points to select from.
        num_centers (int): Number of center indices to select.
        rng_seed (int | None, optional): Seed for the random number generator. Defaults to None.

    Returns:
        ndarray: Array of selected center indices.
    """
    rng = np.random.default_rng(rng_seed)
    rand_indices = rng.choice(num_pts, num_centers)
    return rand_indices


def main():
    """Main entry point for the Lorenz 63 benchmarks.

    This script trains multiple reduced rank regression models with different
    training set sizes and plots the training times and root mean squared errors
    (rMSEs).
    """
    data = make_dataset()
    train_data = data["train"]
    test_data = data["test"]
    # Length scale of the kernel: median of the pairwise distances of the dataset
    data_pdist = pdist(train_data)
    kernel = RBF(length_scale=np.quantile(data_pdist, 0.5))

    rank = 25
    num_centers = 250
    tikhonov_reg = 1e-6

    train_stops = np.logspace(3, 4, 10).astype(int)
    timings = defaultdict(list)
    rMSEs = defaultdict(list)

    for stop in train_stops:
        logger.info(f"######## {stop} Training points ########")
        assert stop < len(train_data)
        X = train_data[:stop]
        Y = train_data[1 : stop + 1]
        kernel_X = kernel(X)
        kernel_Y = kernel(Y)
        kernel_YX = kernel(Y, X)
        X_test = test_data[:-1]
        kernel_Xtest_X = kernel(X_test, X)
        Y_test = test_data[1:]

        # Vanilla Reduced Rank Regression
        fit_results, fit_time = timer(lol.kernel.reduced_rank)(
            kernel_X, kernel_Y, tikhonov_reg, rank
        )
        Y_pred = lol.kernel.predict(1, fit_results, kernel_YX, kernel_Xtest_X, Y)
        rMSE = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
        timings["Vanilla RRR"].append(fit_time)
        logger.info(f"Vanilla RRR {fit_time:.1e}s")
        rMSEs["Vanilla RRR"].append(rMSE.item())
        fit_results, fit_time = timer(lol.kernel.pcr)(kernel_X, tikhonov_reg, rank)
        Y_pred = lol.kernel.predict(1, fit_results, kernel_YX, kernel_Xtest_X, Y)
        rMSE = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
        timings["Vanilla PCR"].append(fit_time)
        logger.info(f"Vanilla PCR {fit_time:.1e}s")
        rMSEs["Vanilla PCR"].append(rMSE.item())
        # Nystroem
        center_idxs = center_selection(len(X), num_centers, rng_seed=42)
        fit_results, fit_time = timer(lol.kernel.nystroem_reduced_rank)(
            kernel(X[center_idxs]),
            kernel(Y[center_idxs]),
            kernel_X[:, center_idxs],
            kernel_Y[:, center_idxs],
            tikhonov_reg,
            rank,
        )
        Y_pred = lol.kernel.predict(
            1,
            fit_results,
            kernel(Y[center_idxs], X[center_idxs]),
            kernel_Xtest_X[:, center_idxs],
            Y[center_idxs],
        )
        rMSE = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
        timings["Nystroem RRR"].append(fit_time)
        logger.info(f"Nystroem RRR {fit_time:.1e}s")
        rMSEs["Nystroem RRR"].append(rMSE.item())

        center_idxs = center_selection(len(X), num_centers, rng_seed=42)
        fit_results, fit_time = timer(lol.kernel.nystroem_pcr)(
            kernel(X[center_idxs]),
            kernel(Y[center_idxs]),
            kernel_X[:, center_idxs],
            kernel_Y[:, center_idxs],
            tikhonov_reg,
            rank,
        )
        Y_pred = lol.kernel.predict(
            1,
            fit_results,
            kernel(Y[center_idxs], X[center_idxs]),
            kernel_Xtest_X[:, center_idxs],
            Y[center_idxs],
        )
        rMSE = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
        timings["Nystroem PCR"].append(fit_time)
        logger.info(f"Nystroem PCR {fit_time:.1e}s")
        rMSEs["Nystroem PCR"].append(rMSE.item())
        # Randomized
        fit_results, fit_time = timer(lol.kernel.rand_reduced_rank)(
            kernel_X, kernel_Y, tikhonov_reg, rank
        )
        Y_pred = lol.kernel.predict(1, fit_results, kernel_YX, kernel_Xtest_X, Y)
        rMSE = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
        timings["Randomized RRR"].append(fit_time)
        logger.info(f"Randomized RRR {fit_time:.1e}s")
        rMSEs["Randomized RRR"].append(rMSE.item())

    # Save results to json
    with open(this_folder / "timings.json", "w") as f:
        json.dump(timings, f)
    with open(this_folder / "rMSEs.json", "w") as f:
        json.dump(rMSEs, f)
    # Plot results
    fig, axes = plt.subplots(ncols=2, figsize=(7, 3), dpi=144)

    for name in [
        "Vanilla RRR",
        "Vanilla PCR",
        "Nystroem RRR",
        "Nystroem PCR",
        "Randomized RRR",
    ]:
        axes[0].plot(train_stops, rMSEs[name], ".-", label=name)
        axes[1].plot(train_stops, timings[name], ".-", label=name)

    axes[0].set_title("rMSE")
    axes[1].set_title("Training time (s)")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Training points")
    plt.tight_layout()
    # Save plot
    plt.savefig(this_folder / "reduced_rank_benchmarks.png")


if __name__ == "__main__":
    main()
