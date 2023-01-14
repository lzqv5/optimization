from tqdm import tqdm, trange
from libsvm.svmutil import svm_read_problem # https://blog.csdn.net/u013630349/article/details/47323883
from time import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import scipy as sp
from scipy.linalg import hessenberg
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
        if   True or np.sum(row) > 1e-3:
            effective_row_ids.append(idx)
    return b[effective_row_ids], A_np[effective_row_ids]


def solve_tridiagonal_system(diag: np.ndarray, subdiag: np.ndarray, tau: float, b: np.ndarray) -> np.ndarray:
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

2 * np.finfo(float).eps
def ss(A, jj):
    """Subfunction for h_trid."""
    return np.sqrt(np.sum(A[jj + 1:, jj] ** 2))

def h_trid(A):
    """
    H_TRID(A) uses Householder method to form a tridiagonal matrix from A.
    Must have a SQUARE SYMMETRIC matrix as the input.
    """
    M, N = A.shape
    if M != N or (A != A.T).any():  # This just screens matrices that can't work.
        raise ValueError("Matrix must be square symmetric only, see help.")

    lngth = len(A)  # Preallocations.
    v = np.zeros(lngth)
    I = np.eye(lngth)
    Aold = A
    finalP=np.eye(lngth)
    for jj in range(lngth - 2):  # Build each vector j and run the whole procedure.
        v[:jj+1] = 0
        S = ss(Aold, jj)
        v[jj + 1] = np.sqrt(0.5 * (1 + abs(Aold[jj + 1, jj]) / (S+2 * np.finfo(float).eps)))
        v[jj + 2:] = Aold[jj + 2:, jj] * np.sign(Aold[jj + 1, jj]) / (2 * v[jj + 1] * S+2 * np.finfo(float).eps )
        P = I - 2 * np.outer(v, v)
        Anew = P @ Aold @ P
        Aold = Anew
        finalP=finalP@P
    # Anew[abs(Anew) < 5e-14] = 0  # Tolerance.

    return Anew,finalP



def minimize_quadratic_on_l2_ball(g: np.ndarray, H: np.ndarray, R: float, inner_eps: float) -> np.ndarray:
    n = g.shape[0]
    # np.savetxt('hess.txt',H)
    # print(np.linalg.norm(H))
    H_tridiag, Q = hessenberg(H,calc_q=True)
    diag = np.diag(H_tridiag)
    subdiag = np.diag(H_tridiag, k=-1)
    # print("Other:",np.sum(H_tridiag)-np.sum(diag)-np.sum(subdiag)*2,flush=True)
    g_ = Q.T.dot(g)

    tau = 1.0
    S_tau = np.zeros(n)
    S_tau_norm = 0.0
    phi_tau = 0.0
    # print(np.linalg.norm(H_tridiag))
    # print(np.linalg.norm(diag),"\t",np.linalg.norm(subdiag),"\t",np.linalg.norm(g_))
    N_LINE_SEARCH_ITERS = 100
    for i in range(N_LINE_SEARCH_ITERS + 1):
        if i == N_LINE_SEARCH_ITERS:
            print("W: Preliminaty line search iterations exceeded in MinimizeQuadraticOnL2Ball")
            break

        S_tau = solve_tridiagonal_system(diag, subdiag, tau, g_)
        S_tau_norm = np.linalg.norm(S_tau)
        phi_tau = 1.0 / S_tau_norm - 1.0 / R
        if phi_tau < inner_eps or tau < inner_eps:
            break
        tau *= 0.5
    # print("phi_tau:",phi_tau)
    if phi_tau < -inner_eps:
        S_tau_grad = np.zeros(n)
        for i in range(N_LINE_SEARCH_ITERS + 1):
            if i == N_LINE_SEARCH_ITERS:
                print("W: 1-D Newton iterations exceeded in MinimizeQuadraticOnL2Ball")
                break

            S_tau_grad = solve_tridiagonal_system(diag, subdiag, tau, S_tau)
            phi_tau_prime = (1.0 / S_tau_norm**3) * (S_tau.T.dot(S_tau_grad))
            tau -= phi_tau / phi_tau_prime

            S_tau = solve_tridiagonal_system(diag, subdiag, tau, g_)
            S_tau_norm = np.linalg.norm(S_tau)
            phi_tau = 1.0 / S_tau_norm - 1.0 / R

            if abs(phi_tau) < inner_eps or abs(phi_tau_prime) < inner_eps:
                break

    return -Q.dot(S_tau)

def Sigmoid(t):
    if t>0:
        return 1.0 / (1 + np.exp(-t))
    else:
        return np.exp(t) / (1 + np.exp(t))
def Log_one_exp(inner):
    if (inner > 0):
        return inner + np.log(1 + np.exp(-inner))
    else:
        return np.log(1 + np.exp(inner))
def contracting_newton(params, c_0, decrease_gamma, variance_reduction):
    # start_time = time.perf_counter()
    # last_logging_time = start_time
    # last_display_time = start_time
    func_val_record = []
    epoch_record=[]
    time_record=[]
    t_s=time()
    n = params['A'].shape[1]
    m = params['A'].shape[0]
    inv_m = 1.0 / m
    data_accesses = m

    x_k = params['x_0'].copy()
    # Ax = params['A'].dot(x_k)
    Ax = params['A']@x_k
    g_k = np.zeros(n)
    H_k = np.zeros((n, n))
    v_k = np.zeros(n)

    gamma_str = f"gamma_k = {c_0}"
    if decrease_gamma:
        gamma_str += " / (3 + k)"
    print(f"Contracting Newton Method, {gamma_str}")
    pbar=tqdm(range(params['n_iters'] ))
    fval_prev=np.average(np.log(1+np.exp(-params['b']*(params['A_o']@x_k))))+inv_m*params['lambda']*np.linalg.norm(x_k)**2
    fval = fval_prev

    for k in pbar:

        if (k>=1 and (grad_norm<params['outer_eps'] or abs(fval-fval_prev)/max(abs(fval_prev),1)<0.1*params['outer_eps'])):
            break

        gamma_k = c_0
        if decrease_gamma:
            gamma_k /= 3.0 + k
            # gamma_k=c_0*(1-(k/(k+1))**3)
            # print("Gamma_k=",gamma_k)
        # print("Round:",k,flush=True)
        if variance_reduction and (k==0 or abs(2**np.floor(np.log2(k)) - k)<1e-6):
            # print(k)
            Ax = params['A'] @ x_k
            full_g_k = inv_m * (params['A'].T.dot(np.array([Sigmoid(ax) for ax in Ax])))
            z_k = x_k.copy()
            data_accesses += m
        
        
        # full_H_k = (inv_m * gamma_k) * (params['A'].T.dot(((1 / (1 + np.exp(-Ax))) * (1 - 1 / (1 + np.exp(-Ax))))[:, np.newaxis] * params['A'])) + (inv_m * gamma_k)*2*params['lambda']*np.diag([1.0]*x_k.size)
        
        obj_indices=np.arange(m)
        k_sqr = (k + 1) * (k + 1)
        batch_size = k_sqr if k_sqr < m else m
        # batch_size=8192
        if batch_size == m:
            print("W: batch_size equals m")
            return
        np.random.shuffle(obj_indices)
        # if k!=0:
        batch = obj_indices[:batch_size].copy()
        A_batch=params['A'][batch,:]
        # print(A_batch.shape)
        A_batchx=A_batch@x_k
        sigax=(np.array([Sigmoid(ax) for ax in A_batchx]))
        if variance_reduction:
            A_batchz=A_batch@z_k
        inv_bs = 1/batch_size
        g_k = inv_bs * (A_batch.T.dot(sigax))+inv_m*2*params['lambda']*x_k
        if variance_reduction:
            g_k+=full_g_k-inv_bs * (A_batch.T.dot(np.array([Sigmoid(az) for az in A_batchz])))
        H_k = (inv_bs * gamma_k) * (A_batch.T.dot(( sigax * (1 - sigax))[:, np.newaxis] * A_batch)) + (inv_m * gamma_k)*2*params['lambda']*np.diag([1.0]*n)
        grad_norm=np.linalg.norm(g_k)
        g_k -= H_k.dot(x_k)

        v_k = minimize_quadratic_on_l2_ball(g_k, H_k, params['R'], params['inner_eps'])

        x_k += gamma_k * (v_k - x_k)
        Ax = params['A'].dot(x_k)
        data_accesses += batch_size
        fval_prev=fval
        fval = np.average(np.array([Log_one_exp(ax) for ax in Ax]))+inv_m*params['lambda']*np.linalg.norm(x_k)**2
        func_val_record.append(fval)
        epoch_record.append(data_accesses//m)
        time_record.append(time()-t_s)
        pbar.set_description(f'Epoch {data_accesses/m} - Function value: %.8f / Grad norm: %.8f'%(fval,grad_norm))
        # print("function value:",np.average(np.log(1+np.exp(-params['b']*(params['A_o']@x_k))))+inv_m*params['lambda']*np.linalg.norm(x_k)**2)
    # print("Done.")
    return x_k,np.asarray(func_val_record),np.asarray(time_record),np.asarray(epoch_record)
