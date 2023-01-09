from tqdm import tqdm, trange
from libsvm.svmutil import svm_read_problem # https://blog.csdn.net/u013630349/article/details/47323883
from time import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_data(path):
    b, A = svm_read_problem(path)
    rows = len(b)   # 矩阵行数, i.e. sample 数
    cols = max([max(row.keys()) if len(row)>0 else 0 for row in A])  # 矩阵列数, i.e. feature 数
    b = np.array(b)
    A_np = np.zeros((rows,cols))
    for r in range(rows):
        for c in A[r].keys():
            # MatLab 是 1-index, python 则是 0-index
            A_np[r,c-1] = A[r][c]
    # 清楚全 0 features
    effective_row_ids = []
    for idx, row in enumerate(A_np):
        if np.sum(row) > 1e-3:
            effective_row_ids.append(idx)
    return b[effective_row_ids], A_np[effective_row_ids]


def solve_tridiagonal_system(diag: np.ndarray, subdiag: np.ndarray, tau: float, b: np.ndarray, x: np.ndarray, buffer: np.ndarray) -> np.ndarray:
    n = diag.shape[0]
    c = np.zeros(n - 1)
    d = np.zeros(n)

    c[0] = subdiag[0] / (diag[0] + tau)
    d[0] = b[0] / (diag[0] + tau)
    for i in range(1, n - 1):
        w = diag[i] + tau - subdiag[i - 1] * c[i - 1]
        c[i] = subdiag[i] / w
        d[i] = (b[i] - subdiag[i - 1] * d[i - 1]) / w
    d[n - 1] = (b[n - 1] - subdiag[n - 2] * d[n - 2]) / (diag[n - 1] + tau - subdiag[n - 2] * c[n - 2])
    for i in range(n - 2, -1, -1):
        d[i] -= c[i] * d[i + 1]

    return d


def minimize_quadratic_on_l2_ball(g: np.ndarray, H: np.ndarray, R: float, inner_eps: float) -> np.ndarray:
    n = g.shape[0]

    H_tridiag, Q = np.linalg.qr(H)
    diag = np.diag(H_tridiag)
    subdiag = np.diag(H_tridiag, k=-1)
    g_ = Q.T.dot(g)

    tau = 1.0
    S_tau = np.zeros(n)
    buffer = np.zeros(n - 1)
    S_tau_norm = 0.0
    phi_tau = 0.0

    N_LINE_SEARCH_ITERS = 100
    for i in range(N_LINE_SEARCH_ITERS + 1):
        if i == N_LINE_SEARCH_ITERS:
            print("W: Preliminaty line search iterations exceeded in MinimizeQuadraticOnL2Ball")
            break

        S_tau = solve_tridiagonal_system(diag, subdiag, tau, g_, S_tau, buffer)
        S_tau_norm = np.linalg.norm(S_tau)
        phi_tau = 1.0 / S_tau_norm - 1.0 / R
        if phi_tau < inner_eps or tau < inner_eps:
            break
        tau *= 0.5

    if phi_tau < -inner_eps:
        S_tau_grad = np.zeros(n)
        for i in range(N_LINE_SEARCH_ITERS + 1):
            if i == N_LINE_SEARCH_ITERS:
                print("W: 1-D Newton iterations exceeded in MinimizeQuadraticOnL2Ball")
                break

            S_tau_grad = solve_tridiagonal_system(diag, subdiag, tau, g_, S_tau, buffer)
            S_tau_norm = np.linalg.norm(S_tau)
            phi_tau_prime = (1.0 / S_tau_norm**3) * (S_tau.T.dot(S_tau_grad))
            tau -= phi_tau / phi_tau_prime

            S_tau = solve_tridiagonal_system(diag, subdiag, tau, g_, S_tau, buffer)
            S_tau_norm = np.linalg.norm(S_tau)
            phi_tau = 1.0 / S_tau_norm - 1.0 / R

            if abs(phi_tau) < inner_eps or abs(phi_tau_prime) < inner_eps:
                break

    return -Q.dot(S_tau)


def contracting_newton(params, c_0, decrease_gamma, history):
    # start_time = time.perf_counter()
    # last_logging_time = start_time
    # last_display_time = start_time

    n = params['A'].shape[1]
    m = params['A'].shape[0]
    inv_m = 1.0 / m
    data_accesses = m

    x_k = params['x_0'].copy()
    Ax = params['A'].dot(x_k)
    g_k = np.zeros(n)
    H_k = np.zeros((n, n))
    v_k = np.zeros(n)

    gamma_str = f"gamma_k = {c_0}"
    if decrease_gamma:
        gamma_str += " / (3 + k)"
    print(f"Contracting Newton Method, {gamma_str}")
    for k in tqdm(range(params['n_iters'] + 1)):
        to_finish = False
        # update_history(
        #     params,
        #     start_time,
        #     k,
        #     data_accesses,
        #     lambda: float('inf') if x_k.norm() > params['R'] + 1e-5 else inv_m * np.logaddexp(Ax, 0).sum(),
        #     last_logging_time,
        #     last_display_time,
        #     history,
        #     to_finish
        # )
        print(np.linalg.norm(g_k))
        if to_finish or (k>=1 and np.linalg.norm(g_k)<params['outer_eps']):
            break

        gamma_k = c_0
        if decrease_gamma:
            gamma_k /= 3.0 + k

        g_k = inv_m * (params['A'].T.dot(1 / (1 + np.exp(-Ax))))
        H_k = (inv_m * gamma_k) * (params['A'].T.dot((1 / (1 + np.exp(-Ax)) * (1 - 1 / (1 + np.exp(-Ax))))[:, np.newaxis] * params['A']))
        g_k -= H_k.dot(x_k)

        v_k = minimize_quadratic_on_l2_ball(g_k, H_k, params['R'], params['inner_eps'])

        x_k += gamma_k * (v_k - x_k)
        Ax = params['A'].dot(x_k)
        data_accesses += m
    print("Done.")

b, A = read_data('w8a')
# b, A = read_data('ijcnn1.test')
# b, A = read_data('a9a.test')
# b, A = read_data('CINA.test')
m,n = A.shape
m,n

params=dict()
params['A']=A
params['x_0']=np.zeros(n)
c_0 = 3.0
params['R']=10
params['inner_eps']=1e-9
params['outer_eps']=1e-5
params['n_iters']=10000
history=None
decrease_gamma=True
contracting_newton(params, c_0, decrease_gamma, history)