import os
import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange, get_num_threads
import numpy as np
import math

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Default Matrix Dimensions & Block Sizes
N = 512
BLOCK_SIZE = 32
BLOCK_SIZE_L2 = 128
BLOCK_SIZE_L1 = 32

#NUMBA_NUM_THREADS=8

EPS = 0.1

# ---------------------------------------------------------
# Baseline 0: Pure Python Naive Correlation
# ---------------------------------------------------------
def correlation_python_0(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    # Column means
    for j in range(m):
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

    # Column standard deviations
    for j in range(m):
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s /= float_n
        s = s ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

# ---------------------------------------------------------
# Baseline 1: apply loop fusion & njit
# ---------------------------------------------------------
@njit
def correlation_python_simplified(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

#----------------------------------------------------------
# Baseline 2: Loop Order Permutations
# ---------------------------------------------------------
@njit
def correlation_2_jii_ij_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ij_ji_ij_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    # Column means
    mean[:] = 0.0
    for i in range(n):
        for j in range(m):
            mean[j] += data[i, j]
    for j in range(m):
        mean[j] /= float_n

    # Column std devs
    for j in range(m):
            s = 0.0
            for i in range(n):
                d = data[i, j] - mean[j]
                s += d * d
            s = (s / float_n) ** 0.5
            stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ji_ij_ij_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    # Column means
    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

    # Column std devs
    stddev[:] = 0.0
    for i in range(n):
        for j in range(m):
            d = data[i, j] - mean[j]
            stddev[j] += d * d
    for j in range(m):
        s = (stddev[j] / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ij_ij_ij_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    # Column means
    mean[:] = 0.0
    for i in range(n):
        for j in range(m):
            mean[j] += data[i, j]
    for j in range(m):
        mean[j] /= float_n

    # Column std devs
    stddev[:] = 0.0
    for i in range(n):
        for j in range(m):
            d = data[i, j] - mean[j]
            stddev[j] += d * d
    for j in range(m):
        s = (stddev[j] / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_jii_ji_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for j in range(m):
        for i in range(n):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ji_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

        # center and reduce data
        for i in range(n):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ji_ij_ijk(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s
    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ji_ij_ikj(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m - 1):
        corr[i, i] = 1.0

        for j in range(i + 1, m):
            corr[i, j] = 0.0

        for k in range(n):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

    corr[m - 1, m - 1] = 1.0

@njit
def correlation_2_ji_ij_jik(data, corr, mean, stddev):
    
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for j in range(1, m):
        for i in range(j):
            s = 0.0
            for k in range(n):
                s += data[k, i] * data[k, j]
            corr[i, j] = s
            corr[j, i] = s

    for i in range(m):
        corr[i, i] = 1.0

@njit
def correlation_2_ji_ij_jki(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    for j in range(1, m):
        for i in range(j):
            corr[i, j] = 0.0

        for k in range(n):
            dkj = data[k, j]
            for i in range(j):
                corr[i, j] += data[k, i] * dkj

        for i in range(j):
            corr[j, i] = corr[i, j]

    for i in range(m):
        corr[i, i] = 1.0

@njit
def correlation_2_ji_ij_kij(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit
def correlation_2_ji_ij_kji(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for j in range(1, m):
            dkj = data[k, j]
            for i in range(j):
                corr[i, j] += data[k, i] * dkj

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

# ---------------------------------------------------------
# Baseline 3: Optimization Flags for Backend
# ---------------------------------------------------------
@njit(fastmath=True)
def correlation_opt_flags_3(data, corr, mean, stddev): #j(ii) ji kij
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for j in range(m):
        for i in range(n):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

# ---------------------------------------------------------
# Baseline 4: Parallel Loop Versions
# ---------------------------------------------------------
@njit(parallel=False, fastmath=True)
def correlation_parallel_jii_ij_kij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_Jii_ij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in prange(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_jIi_ij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in prange(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_jiI_ij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in prange(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_jii_Ij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in prange(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_jii_iJ_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in prange(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_Ijkijij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in prange(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_iJkijij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in prange(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_ijKijij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    # each thread owns a chunk of k and a private accumulator
    nchunks = get_num_threads()
    chunk = (n + nchunks - 1) // nchunks
    partial = np.zeros((nchunks, m, m))

    for c in prange(nchunks):
        for k in range(c * chunk, min((c + 1) * chunk, n)):
            for i in range(m - 1):
                dki = data[k, i]
                for j in range(i + 1, m):
                    partial[c, i, j] += dki * data[k, j]

    # Combine the private copies; each i owns row corr[i, :]
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0
        for c in range(nchunks):
            for j in range(i + 1, m):
                corr[i, j] += partial[c, i, j]

    # Mirror once every upper-triangle row is final
    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_ijkIjij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in prange(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_ijkiJij_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in prange(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_ijkijIj_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in prange(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_ijkijiJ_4(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in range(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k in range(n):
        for i in range(m - 1):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

    for i in range(m - 1):
        for j in prange(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_jii_ij_kij_ikKIjij_4(data, corr, mean, stddev): #jii ij ijKIjij
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    # each thread owns a chunk of k and a private accumulator
    nchunks = get_num_threads()
    chunk = (n + nchunks - 1) // nchunks
    partial = np.zeros((nchunks, m, m))

    for c in prange(nchunks):
        for k in range(c * chunk, min((c + 1) * chunk, n)):
            for i in range(m - 1):
                dki = data[k, i]
                for j in range(i + 1, m):
                    partial[c, i, j] += dki * data[k, j]

    # Combine the private copies; each i owns row corr[i, :]
    for i in prange(m):
        corr[i, i] = 1.0
        for j in range(i + 1, m):
            corr[i, j] = 0.0
        for c in range(nchunks):
            for j in range(i + 1, m):
                corr[i, j] += partial[c, i, j]

    # Mirror once every upper-triangle row is final
    for i in prange(m - 1):
        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

@njit(parallel=True, fastmath=True)
def correlation_parallel_Ijkj_4(data, corr, mean, stddev): #jii ij Ijkj
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in prange(m - 1):
        corr[i, i] = 1.0

        for j in range(i + 1, m):
            corr[i, j] = 0.0

        for k in range(n):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

    corr[m - 1, m - 1] = 1.0

@njit(parallel=True, fastmath=True)
def correlation_parallel_IJkj_4(data, corr, mean, stddev): #jii ij IJkj
    n, m = data.shape
    float_n = float(n)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    for i in prange(m - 1):
        corr[i, i] = 1.0

        for j in prange(i + 1, m):
            corr[i, j] = 0.0

        for k in range(n):
            dki = data[k, i]
            for j in range(i + 1, m):
                corr[i, j] += dki * data[k, j]

        for j in range(i + 1, m):
            corr[j, i] = corr[i, j]

    corr[m - 1, m - 1] = 1.0

# ---------------------------------------------------------
# Baseline 5: Blocked (Tiled) Parallel I Code
# ---------------------------------------------------------
@njit(inline='always', fastmath=True)
def _block_helper(data, corr, b, bs):
    n, m = data.shape
    i0 = b * bs #starting index of row for corr
    i1 = min(i0 + bs, m) #ending index of row for corr

    # Diagonal and zero accumulators for this block's rows
    for i in range(i0, i1):
        corr[i, i] = 1.0 # diagonal cells = 1
        for j in range(i + 1, m):
            corr[i, j] = 0.0 # init cells of upper triangle to 0

    # Tiled i, k, j accumulation (upper triangle only)
    for k0 in range(0, n, bs): # tiled k
        k1 = min(k0 + bs, n) 
        for j0 in range(i0, m, bs): # tiled j; start at i0 bc upper triangle only
            j1 = min(j0 + bs, m) 
            for i in range(i0, i1): # all i (not tiled); start at i0 bc upper triangle only
                js = max(j0, i + 1) # ensures j > i so upper triangle only
                for k in range(k0, k1): #for each item in chunk
                    dki = data[k, i] #load scalar
                    for j in range(js, j1): #for each column
                        corr[i, j] += dki * data[k, j] #multiply and add (i.e. we're building the dot product)

@njit(parallel=True, fastmath=True)
def correlation_blocked_parallel_5(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)
    bs = BLOCK_SIZE

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    nb = (m + bs - 1) // bs #number of blocks

    # Parallel over I blocks: since the rows are uneven as
    # we're working over a triangle; we pair block p (most work) 
    # with block nb-1-p (least work) so every iteration is ~ the same.
    for p in prange((nb + 1) // 2):
        _block_helper(data, corr, p, bs)
        q = nb - 1 - p
        if q != p:
            _block_helper(data, corr, q, bs)

    # Tiled mirror: filling in the lower triangle for matrix correctness
    for b in prange(nb):
        i0 = b * bs
        i1 = min(i0 + bs, m)
        for j0 in range(i0, m, bs):
            j1 = min(j0 + bs, m)
            for i in range(i0, i1):
                for j in range(max(j0, i + 1), j1):
                    corr[j, i] = corr[i, j]

# ---------------------------------------------------------
# Baseline 6: Blocked Parallel I using np.dot for Sub-blocks
# ---------------------------------------------------------
@njit(inline='always', fastmath=True)
def _block_helper_np_dot(data, corr, b, bs):
    n, m = data.shape
    i0 = b * bs
    i1 = min(i0 + bs, m)

    # Zero this block's rows from the diagonal tile rightwards
    for i in range(i0, i1):
        for j in range(i0, m):
            corr[i, j] = 0.0

    for k0 in range(0, n, bs):
        k1 = min(k0 + bs, n)

        # transposed tile as continguous array for np.dot
        # built once per k block and reused across j tiles
        a_t = np.ascontiguousarray(data[k0:k1, i0:i1].T) 

        for j0 in range(i0, m, bs): # move right across the matrix, starting at the diagonal
            j1 = min(j0 + bs, m)
            b_tile = np.ascontiguousarray(data[k0:k1, j0:j1])
            corr[i0:i1, j0:j1] += np.dot(a_t, b_tile)

@njit(parallel=True, fastmath=True)
def correlation_blocked_np_dot_6(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)
    bs = BLOCK_SIZE

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] -= mean[j]
            data[i, j] /= (float_n ** 0.5) * stddev[j]

    # Correlation matrix (upper triangle, mirrored)
    nb = (m + bs - 1) // bs

        # Parallel over I blocks, pairing heavy block p with light block nb-1-p
    for p in prange((nb + 1) // 2):
        _block_helper_np_dot(data, corr, p, bs)
        q = nb - 1 - p
        if q != p:
            _block_helper_np_dot(data, corr, q, bs)

    # Tiled mirror. This also overwrites the stray lower-triangle cells
    # that np.dot computed inside the diagonal tiles.
    for b in prange(nb):
        i0 = b * bs
        i1 = min(i0 + bs, m)
        for i in range(i0, i1):
            corr[i, i] = 1.0
            for j in range(i + 1, m):
                corr[j, i] = corr[i, j]

# ---------------------------------------------------------
# Baseline 7: 
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def correlation_hoist_7(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)
    scale = float_n ** 0.5 # hoisting out this calculation to save m*n operations
    bs = BLOCK_SIZE

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s / float_n

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s / float_n) ** 0.5
        stddev[j] = 1.0 if s <= EPS else s

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] = (data[i, j] - mean[j]) / (scale * stddev[j])

    # Correlation matrix (upper triangle, mirrored)
    nb = (m + bs - 1) // bs

    # Parallel over I blocks, pairing heavy block p with light block nb-1-p
    for p in prange((nb + 1) // 2):
        _block_helper_np_dot(data, corr, p, bs)
        q = nb - 1 - p
        if q != p:
            _block_helper_np_dot(data, corr, q, bs)

    # Tiled mirror. This also overwrites the stray lower-triangle cells
    # that np.dot computed inside the diagonal tiles.
    for b in prange(nb):
        i0 = b * bs
        i1 = min(i0 + bs, m)
        for i in range(i0, i1):
            corr[i, i] = 1.0
            for j in range(i + 1, m):
                corr[j, i] = corr[i, j]

# ----
# 8 - vectorization
# ----
@njit(parallel=True, fastmath=True)
def correlation_vector_8(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)
    inv_float_n = 1 / float_n
    scale = float_n ** 0.5
    bs = BLOCK_SIZE
    inv = np.empty(m)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s * inv_float_n # multiplying vectorizes cleaner than dividing

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s * inv_float_n) ** 0.5 # multiplying vectorizes cleaner than dividing
        stddev[j] = 1.0 if s <= EPS else s
        inv[j] = 1.0 / (scale * stddev[j]) # multiplying vectorizes cleaner than dividing

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] = (data[i, j] - mean[j]) * inv[j] # multiplying vectorizes cleaner than dividing

    # Correlation matrix (upper triangle, mirrored)
    nb = (m + bs - 1) // bs

        # Parallel over I blocks, pairing heavy block p with light block nb-1-p
    for p in prange((nb + 1) // 2):
        _block_helper_np_dot(data, corr, p, bs)
        q = nb - 1 - p
        if q != p:
            _block_helper_np_dot(data, corr, q, bs)

    # Tiled mirror. This also overwrites the stray lower-triangle cells
    # that np.dot computed inside the diagonal tiles.
    for b in prange(nb):
        i0 = b * bs
        i1 = min(i0 + bs, m)
        for i in range(i0, i1):
            corr[i, i] = 1.0
            for j in range(i + 1, m):
                corr[j, i] = corr[i, j]

# ------
# 9 - skip wasted iters in helper
# ------
@njit(inline='always', fastmath=True)
def _block_helper_skip_iters(data, corr, b, bs):
    n, m = data.shape
    i0 = b * bs
    i1 = min(i0 + bs, m)

    for i in range(i0, i1):
        corr[i, i] = 1.0 # set diagonal to 0
        for j in range(i + 1, m):
            corr[i, j] = 0.0

    for k0 in range(0, n, bs):
        k1 = min(k0 + bs, n)
        for j0 in range(i0, m, bs):
            j1 = min(j0 + bs, m)
            for i in range(i0, i1):
                js = max(j0, i + 1) # only perform upper triangle, then mirror
                for k in range(k0, k1):
                    dik = data[k, i]
                    for j in range(js, j1):
                        corr[i, j] += dik * data[k, j]

@njit(parallel=True, fastmath=True)
def correlation_skip_iters_9(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)
    inv_float_n = 1 / float_n
    scale = float_n ** 0.5
    bs = BLOCK_SIZE
    inv = np.empty(m)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s * inv_float_n # multiplying vectorizes cleaner than dividing

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s * inv_float_n) ** 0.5 # multiplying vectorizes cleaner than dividing
        stddev[j] = 1.0 if s <= EPS else s
        inv[j] = 1.0 / (scale * stddev[j]) # multiplying vectorizes cleaner than dividing

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] = (data[i, j] - mean[j]) * inv[j] # multiplying vectorizes cleaner than dividing

    # Correlation matrix (upper triangle, mirrored)
    nb = (m + bs - 1) // bs

        # Parallel over I blocks, pairing heavy block p with light block nb-1-p
    for p in prange((nb + 1) // 2):
        _block_helper_np_dot(data, corr, p, bs)
        q = nb - 1 - p
        if q != p:
            _block_helper_np_dot(data, corr, q, bs)

    # Tiled mirror. This also overwrites the stray lower-triangle cells
    # that np.dot computed inside the diagonal tiles.
    for b in prange(nb):
        i0 = b * bs
        i1 = min(i0 + bs, m)
        for i in range(i0, i1):
            corr[i, i] = 1.0
            for j in range(i + 1, m):
                corr[j, i] = corr[i, j]

# --------------
# 10 - 4 x tiles
# --------------
@njit(parallel=True, fastmath=True)
def correlation_larger_tile_10(data, corr, mean, stddev):
    n, m = data.shape
    float_n = float(n)
    inv_float_n = 1 / float_n
    scale = float_n ** 0.5
    bs = int(BLOCK_SIZE * 4)
    inv = np.empty(m)

    for j in range(m):
        # column means
        s = 0.0
        for i in range(n):
            s += data[i, j]
        mean[j] = s * inv_float_n # multiplying vectorizes cleaner than dividing

        # column std devs
        s = 0.0
        for i in range(n):
            d = data[i, j] - mean[j]
            s += d * d
        s = (s * inv_float_n) ** 0.5 # multiplying vectorizes cleaner than dividing
        stddev[j] = 1.0 if s <= EPS else s
        inv[j] = 1.0 / (scale * stddev[j]) # multiplying vectorizes cleaner than dividing

    # Center and reduce the data
    for i in range(n):
        for j in range(m):
            data[i, j] = (data[i, j] - mean[j]) * inv[j] # multiplying vectorizes cleaner than dividing

    # Correlation matrix (upper triangle, mirrored)
    nb = (m + bs - 1) // bs

        # Parallel over I blocks, pairing heavy block p with light block nb-1-p
    for p in prange((nb + 1) // 2):
        _block_helper_np_dot(data, corr, p, bs)
        q = nb - 1 - p
        if q != p:
            _block_helper_np_dot(data, corr, q, bs)

    # Tiled mirror. This also overwrites the stray lower-triangle cells
    # that np.dot computed inside the diagonal tiles.
    for b in prange(nb):
        i0 = b * bs
        i1 = min(i0 + bs, m)
        for i in range(i0, i1):
            corr[i, i] = 1.0
            for j in range(i + 1, m):
                corr[j, i] = corr[i, j]



baseline_functions = [
    ("1_simplified", correlation_python_simplified),
    ("2_order_1and2_jii", correlation_2_jii_ij_ijk),
    ("2_order_1and2_ij_ji", correlation_2_ij_ji_ij_ijk),
    ("2_order_1and2_ji_ij", correlation_2_ji_ij_ij_ijk),
    ("2_order_1and2_ij_ij", correlation_2_ij_ij_ij_ijk),
    ("2_order_3_ij", correlation_2_jii_ij_ijk),
    ("2_order_3_ji", correlation_2_jii_ji_ijk),
    ("2_order_1and2and3_jiii", correlation_2_ji_ijk),
    ("2_order_4_ijk", correlation_2_ji_ij_ijk),
    ("2_order_4_ikj", correlation_2_ji_ij_ikj),
    ("2_order_4_jik", correlation_2_ji_ij_jik),
    ("2_order_4_jki", correlation_2_ji_ij_jki),
    ("2_order_4_kij", correlation_2_ji_ij_kij),
    ("2_order_4_kji", correlation_2_ji_ij_kji),
    ("3_fastmath", correlation_opt_flags_3),
    ("4_parallel_jii_ij_ijkijij", correlation_parallel_jii_ij_kij_4),
    ("4_parallel_Jii_ij_ijkijij", correlation_parallel_Jii_ij_4),
    ("4_parallel_jIi_ij_ijkijij", correlation_parallel_jIi_ij_4),
    ("4_parallel_jiI_ij_ijkijij", correlation_parallel_jiI_ij_4),
    ("4_parallel_jii_Ij_ijkijij", correlation_parallel_jii_Ij_4),
    ("4_parallel_jii_iJ_ijkijij", correlation_parallel_jii_iJ_4),
    ("4_parallel_jii_ij_Ijkijij", correlation_parallel_Ijkijij_4),
    ("4_parallel_jii_ij_iJkijij", correlation_parallel_iJkijij_4),
    ("4_parallel_jii_ij_ijKijij", correlation_parallel_ijKijij_4),
    ("4_parallel_jii_ij_ijkIjij", correlation_parallel_ijkIjij_4),
    ("4_parallel_jii_ij_ijkiJij", correlation_parallel_ijkiJij_4),
    ("4_parallel_jii_ij_ijkijIj", correlation_parallel_ijkijIj_4),
    ("4_parallel_jii_ij_ijkijiJ", correlation_parallel_ijkijiJ_4),
    ("4_parallel_jii_ij_ijKIjij", correlation_parallel_jii_ij_kij_ikKIjij_4),
    ("4_parallel_jii_ij_Ijkj", correlation_parallel_Ijkj_4),
    ("4_parallel_jii_ij_IJkj", correlation_parallel_IJkj_4),
    ("5_blocked_parallel", correlation_blocked_parallel_5),
    ("6_blocked_np_dot", correlation_blocked_np_dot_6),
    ("7_hoist", correlation_hoist_7),
    ("8_vectorization", correlation_vector_8),
    ("9_skip_iters", correlation_skip_iters_9),
    ("10_larger_tiles", correlation_larger_tile_10),
]

# ---------------------------------------------------------
# Execution & Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(matrix_size=512):
    global N
    N = matrix_size
    M = N  # matrix dimensions. assume square matrix

    np.random.seed(42)
    data = np.random.randn(N, M).astype(np.float64)

    # Compute ground truth reference solution for correctness verification
    corr_expected = np.corrcoef(data, rowvar=False)  # shape (M, M)

    def count_flops(n, m):
        return n * m * (m - 1) + 7.0 * n * m

    total_flops = count_flops(N, M)

    def check(corr, expected):
        return np.allclose(corr, expected, rtol=1e-5, atol=1e-5)

    def measure(fn, warmup=True, reps=9):
        corr = np.zeros((M, M), dtype=np.float64)
        mean = np.zeros(M, dtype=np.float64)
        stddev = np.zeros(M, dtype=np.float64)

        # Kernel mutates `data`, so every call gets a fresh copy.
        if warmup:
            fn(data.copy(), corr, mean, stddev)

        # Correctness check
        corr.fill(0.0)
        fn(data.copy(), corr, mean, stddev)
        is_correct = check(corr, corr_expected)

        total = 0.0
        for _ in range(reps):
            d = data.copy()
            corr.fill(0.0)
            t0 = time.perf_counter()
            fn(d, corr, mean, stddev)
            total += time.perf_counter() - t0
        elapsed = total / reps
        gflops = (total_flops / elapsed) / 1e9
        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (Run on a smaller dimension if N is large to prevent lockups)
    N_py = min(N, 128)
    M_py = N_py
    data_py = data[:N_py, :M_py].copy()
    corr_py_expected = np.corrcoef(data_py, rowvar=False)
    corr_py = np.zeros((M_py, M_py), dtype=np.float64)
    mean_py = np.zeros(M_py, dtype=np.float64)
    stddev_py = np.zeros(M_py, dtype=np.float64)

    t0 = time.perf_counter()
    correlation_python_0(data_py, corr_py, mean_py, stddev_py)
    py_elapsed = time.perf_counter() - t0
    py_gflops = (count_flops(N_py, M_py) / py_elapsed) / 1e9
    py_correct = check(corr_py, corr_py_expected)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = baseline_functions

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    #print("num threads ", get_num_threads())
    #print("cpu cores ", os.cpu_count())

    return results
# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get user input for matrix dimension N
    if len(sys.argv) > 1:
        try:
            N_input = int(sys.argv[1])
        except ValueError:
            N_input = 512
    else:
        N_input = 512

    print(f"\nRunning Matrix Multiplication Benchmarks for Matrix Size {N_input}x{N_input}...\n")
    benchmark_data = run_benchmark(N_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    prev_elapsed = None

    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"

        print(f"| {name:<33} | {gflops:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
        prev_elapsed = elapsed

    print(divider)

    # Plotting Output
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba Matrix Multiplication Benchmark Performance (N={N_input})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()