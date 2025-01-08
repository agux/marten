import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import math
import logging

import numpy as np

from datetime import datetime

from dask.distributed import Client, LocalCluster

logging.getLogger("distributed.nanny").setLevel(logging.CRITICAL)
logging.getLogger("distributed.scheduler").setLevel(logging.CRITICAL)
logging.getLogger("tornado.application").setLevel(logging.CRITICAL)
logging.getLogger("distributed.worker").setLevel(logging.CRITICAL)

def configure_parser(parser):
    parser.add_argument(
        "--n_workers",
        type=int,
        default=96,
        help="Number of worker processes (default: 96).",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=None,
        help="Number of tasks to submit (default: same as n_workers).",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=1000,
        help="Number of time steps in the RNN (default: 1000).",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=512,
        help="Input size for each time step (default: 512).",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Hidden state size of the RNN (default: 512).",
    )

    parser.set_defaults(func=run_test)


def run_test(args):
    print("Running performance test...")

    n_workers = args.n_workers
    n_tasks = args.n_tasks if args.n_tasks is not None else n_workers
    n_timesteps = args.n_timesteps
    input_size = args.input_size
    hidden_size = args.hidden_size

    # Create a local Dask cluster with the specified number of workers
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    client = Client(cluster)

    start_time = datetime.now()
    # Submit tasks to the Dask cluster
    futures = [
        client.submit(neural_network_computation, n_timesteps, input_size, hidden_size)
        for _ in range(n_tasks)
    ]

    # Gather the results (computation times for each task)
    times_taken = client.gather(futures)

    time_taken = (datetime.now() - start_time).total_seconds()

    # Print the computation time for each task
    for idx, t in enumerate(times_taken):
        print(f"Task {idx + 1} took {t:.2f} seconds.")

    print(f"Total: {time_taken:.2f} seconds.")

    print(f"Stopping Dask client and cluster...")
    try:
        client.shutdown()
    except Exception:
        pass


# Define a CPU-intensive function
def cpu_heavy_function(n):
    start_time = datetime.now()
    result = 0.0
    for i in range(1, n):
        result += math.sqrt(i) * math.log(i + 1)
    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    return time_taken


# Define a function that simulates neural network computations with backpropagation
def neural_network_computation(n_timesteps, input_size, hidden_size):
    start_time = datetime.now()

    # Random initialization of inputs and weights
    np.random.seed(0)  # For reproducibility
    x = np.random.randn(n_timesteps, input_size)  # Input time series data
    h = np.zeros(
        (n_timesteps + 1, hidden_size)
    )  # Hidden states (including initial hidden state)
    y = np.zeros((n_timesteps, input_size))  # Outputs
    targets = np.random.randn(
        n_timesteps, input_size
    )  # Target outputs for loss computation

    Wxh = np.random.randn(hidden_size, input_size)  # Input to hidden weights
    Whh = np.random.randn(hidden_size, hidden_size)  # Hidden to hidden weights
    Why = np.random.randn(input_size, hidden_size)  # Hidden to output weights

    bh = np.random.randn(
        hidden_size,
    )  # Hidden bias
    by = np.random.randn(
        input_size,
    )  # Output bias

    # Forward pass
    for t in range(n_timesteps):
        # Hidden state update (simple RNN)
        h[t] = np.tanh(np.dot(Wxh, x[t]) + np.dot(Whh, h[t - 1]) + bh)
        # Output (this can be thought of as decoding)
        y[t] = np.dot(Why, h[t]) + by

    # Compute loss (mean squared error)
    loss = np.sum(0.5 * (y - targets) ** 2)

    # Backward pass (Backpropagation Through Time)
    dWhy = np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh = np.zeros_like(bh)
    dh_next = np.zeros(hidden_size)

    for t in reversed(range(n_timesteps)):
        dy = y[t] - targets[t]  # Derivative of loss w.r.t. y
        dWhy += np.outer(dy, h[t])  # Gradients w.r.t. Why
        dby += dy  # Gradients w.r.t. by
        dh = np.dot(Why.T, dy) + dh_next  # Backprop into h
        dh_raw = (1 - h[t] ** 2) * dh  # Backprop through tanh nonlinearity
        dbh += dh_raw  # Gradients w.r.t. bh
        dWxh += np.outer(dh_raw, x[t])  # Gradients w.r.t. Wxh
        dWhh += np.outer(dh_raw, h[t - 1])  # Gradients w.r.t. Whh
        dh_next = np.dot(Whh.T, dh_raw)  # Pass gradient to previous time step

    end_time = datetime.now()
    time_taken = (end_time - start_time).total_seconds()
    return time_taken
