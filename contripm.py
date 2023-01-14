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
# 1/params['lambda'] = 100
def phi(x,D):
    real = D/2-np.linalg.norm(x,ord=2)
    return -np.log(real) if real>0 else float("inf")

def phi_grad(x,D):
    x_norm = np.linalg.norm(x,ord=2)
    return x/(x_norm*(D/2-x_norm))
 
def phi_hessian(x,D):
    x_norm = np.linalg.norm(x,ord=2)
    # print(x_norm)
    xxT = np.matmul(x[:,None],x[None,:])    # x * xT
    return np.eye(x.size)/(x_norm*(D/2-x_norm)) + (2*x_norm-D/2)/(x_norm**3 * (D/2-x_norm)**2)*xxT
def f(x,params):
    b=params['b']
    A=params['A_o']
    m=A.shape[0]
    bAx = b*(A@x)
    exp_mbAx = np.exp(-bAx)
    log1p_exp = np.log(1+exp_mbAx)
    overflow_idxs = np.where(exp_mbAx==float('inf'))
    log1p_exp[overflow_idxs] = -bAx[overflow_idxs]
    return log1p_exp.mean() + 1/(1/params['lambda']*m)* x.T@x

def f_orig(x,params):
    return f(x,params)

def f_grad(x,params):
    m=params['A'].shape[0]
    return 1/m * (params['A'].T.dot(1 / (1 + np.exp(-params['A']@x)))+2*params['lambda']*x)

def f_hessian(x,params):
    A=params['A']
    m=A.shape[0]
    Ax = A@x
    return (1/m ) * (params['A'].T.dot(((1 / (1 + np.exp(-Ax))) * (1 - 1 / (1 + np.exp(-Ax))))[:, np.newaxis] * params['A'])) + 1/m*2*params['lambda']*np.diag([1.0]*x.size)
# Can also be
# def f_grad(x,params):
#     b=params['b']
#     A=params['A_o']
#     m=A.shape[0]
#     return np.ones(m)@(np.expand_dims((-b)/(1+np.exp(b*(A@x))), axis=1)*A)/m + 2/(1/params['lambda']*m)*x

# def f_hessian(x,params):
#     b=params['b']
#     A=params['A_o']
#     m=A.shape[0]
#     Ax = A@x
#     exp_bAx = np.exp(b*Ax)
#     return (A.T @ (np.expand_dims(b*b*exp_bAx/(1+exp_bAx)**2, axis=1)*A) )/m + 2/(1/params['lambda']*m)*np.eye(x.size)
def f_int(x,params,x_k,gamma_k):
    return f_grad(x_k,params).dot(x-x_k)+gamma_k/2*(f_hessian(x_k,params)@(x-x_k)).dot(x-x_k)
def f_int_grad(x,params,x_k,gamma_k):
    return f_grad(x_k,params)+gamma_k*f_hessian(x_k,params)@(x-x_k)
def f_int_hess(x,params,x_k,gamma_k):
    return gamma_k*f_hessian(x_k,params)
def barrier_method(t_init, f, f_grad, f_hessian, phi, phi_grad, phi_hessian, A, b, x0, D, num_constraints, mu,t_s,params,
                        method='newton', epsilon=1e-6, maxIter=20):
    xt = x0
    t = t_init
    duality_gaps = []
    func_val_record = []
    time_record = []
    pbar=tqdm(range(maxIter))
    pbar.set_description('内点法')
    for i in pbar:
        xt,num_newton_step, fvals,times = solve_central(objective=lambda x:f_orig(x,params),
                                f=lambda x:t*f(x)+phi(x,D), 
                                f_grad=lambda x:t*f_grad(x)+phi_grad(x,D), 
                                f_hessian=lambda x:t*f_hessian(x)+phi_hessian(x,D),
                                x0=xt, D=D, method=method, epsilon=epsilon*1e3)
        times=(np.array(times)-t_s).tolist() 
        duality_gaps.extend([num_constraints/t]*num_newton_step)
        func_val_record.extend(fvals)
        time_record.extend(times)
        if num_constraints/t < epsilon:
            break
        t *= mu
    t_e = time()
    return xt, t_e-t_s, np.array(duality_gaps), np.array(func_val_record),np.array(time_record)
def armijo_search(f, f_grad, xk, t_hat, alpha, beta, D, isNewton=False, dk=None):
    if isNewton:
        assert dk is not None
    tk = t_hat*1
    grad = f_grad(xk)
    while True:
        if isNewton:
            if np.linalg.norm(xk+tk*dk,ord=2)<=D/2 and f(xk+tk*dk) <= f(xk) + alpha*tk*grad.T@dk:
                break
        else:
            # if np.linalg.norm(xk-tk*grad,ord=2)<=D/2 and f(xk-tk*grad) <= f(xk)-alpha*tk*grad.T@grad:
            if f(xk-tk*grad) <= f(xk)-alpha*tk*grad.T@grad:
                break
        tk *= beta
    return tk

def solve_central(objective, f, f_grad, f_hessian, x0, D, method='newton', epsilon=1e-6, max_iter=50):
    if method == 'newton':
        return damped_newton(objective, f=f, f_grad=f_grad, f_hessian=f_hessian, x0=x0, D=D, epsilon=epsilon, max_iter=max_iter)
    if method == 'bfgs':
        return bfgs(objective, f=f, f_grad=f_grad, f_hessian=f_hessian, x0=x0, D=D, epsilon=epsilon, max_iter=max_iter)
#* 阻尼牛顿
def damped_newton(objective, f, f_grad, f_hessian, x0, D, epsilon=1e-6, max_iter=50):
    xk = x0
    iter_cnt = 0
    fvals = []
    times=[]
    for idx in range(max_iter):
        iter_cnt += 1
        fvals.append(objective(xk))
        times.append(time())
        grad = f_grad(xk)
        hessian = f_hessian(xk)
        dk = -np.linalg.inv(hessian)@grad
        decrement = (-grad@dk)**0.5
        if decrement**2/2 <= epsilon:
            # print('** End The Loop - Iter Cnt.:',iter_cnt, 'Decrement:',decrement, 'fval:',f(xk))
            return xk, iter_cnt, fvals,times
        tk = armijo_search(f, f_grad, xk, t_hat=1, alpha=0.1, beta=0.5, D=D, isNewton=True, dk=dk)
        # print('Iter Cnt.:',iter_cnt, 'Decrement:',decrement, 'fval:',f(xk), 'tk:',tk)
        xk += tk*dk
    return xk, iter_cnt, fvals, times


def update_approximation_bfgs(mat, sk, yk, mat_type='H'):
    rhok = 1/(yk@sk)
    if mat_type == 'H':
        Hkyk = mat@yk
        ykTHkyk = yk@Hkyk
        HkykskT = Hkyk[:,None]@sk[None,:]
        skskT = sk[:,None]@sk[None,:]
        mat_new = mat + rhok*((rhok*ykTHkyk+1)*skskT - HkykskT - HkykskT.T)
    else:
        Bksk = mat@sk
        skTBksk = sk@Bksk
        mat_new = mat - Bksk[:,None]@Bksk[None,:]/skTBksk + yk[:,None]@yk[None,:]*rhok
    return mat_new

#* 拟牛顿方法 - 选择步长
def wolfe_condition(f, f_grad, xk, pk, D, c1=1e-4, c2=0.9, multiplier=1.2, t0=0, tmax=2):
    ### 
    while (np.linalg.norm(xk+tmax*pk)>=D/2):
        tmax /= 2
        # print('tmax:',tmax)
        if tmax<1e-6:
            # print('too small stepsize')
            return -1
    ###
    ti = tmax/2
    tprev = t0
    i = 1
    fval_cur = f(xk)
    grad_cur = f_grad(xk)
    while True:
        xk_next = xk+ti*pk
        fval_next = f(xk_next)
        if (fval_next > fval_cur + c1*ti*grad_cur@pk) or (fval_next >= fval_cur and i>1):
            return zoom(f, f_grad, xk, pk, fval_cur, grad_cur, c1, c2, tprev, ti)
        grad_next = f_grad(xk_next)
        grad_next_T_pk = grad_next@pk
        if np.abs(grad_next_T_pk) <= -c2*grad_cur@pk:
            return ti
        if grad_next_T_pk >= 0:
            return zoom(f, f_grad, xk, pk, fval_cur, grad_cur, c1, c2, ti, tprev)
        tprev = ti
        ti = tprev*multiplier
        i += 1
def zoom(f, f_grad, xk, pk, fval, grad, c1, c2, t_lo, t_hi):
    while True:
        # print(f"t_lo: {t_lo}\tt_hi: {t_hi}")
        t = (t_lo+t_hi)/2
        xk_next = xk + t*pk
        fval_next = f(xk_next)
        if fval_next > fval + c1*t*grad@pk or fval_next >= f(xk+t_lo*pk):
            t_hi = t
        else:
            grad_next = f_grad(xk_next)
            grad_next_T_pk = grad_next@pk
            if np.abs(grad_next_T_pk) <= -c2*grad@pk:
                return t
            if grad_next_T_pk*(t_hi-t_lo)>=0:
                t_hi = t_lo
            t_lo = t
        if t_lo == t_hi: # 死循环
            return -1
#* 拟牛顿
def bfgs(objective, f, f_grad, f_hessian, x0, D, alpha=0.1, beta=0.5, epsilon=1e-6, max_iter=500):
    xk = x0
    hessian = f_hessian(x0)
    mat_k = np.linalg.inv(hessian) 
    # mat_k = np.eye(n) 
    iter_cnt = 0
    fvals = []
    times=[]
    for idx in range(max_iter):
        iter_cnt += 1
        grad_k = f_grad(xk)
        dk = -mat_k@grad_k 
        tk = wolfe_condition(f, f_grad, xk, dk, D, c1=1e-4, c2=0.9)
        if tk<0:
            return xk, iter_cnt-1, fvals,times
        fvals.append(objective(xk))
        times.append(time())
        sk = tk*dk
        xk_next = xk + sk
        grad_next = f_grad(xk_next)
        # if np.linalg.norm(grad_next, ord=2) <= epsilon:
        # if np.linalg.norm(xk_next)>=D/2-1e-2:
        #     while np.linalg.norm(xk_next)>=D/2-1e-2:
        #         xk_next = xk + tk / 2 * dk
        #         tk /= 2
        #     return xk_next, iter_cnt, fvals, times
        if np.linalg.norm(grad_next, ord=2) <= epsilon or np.linalg.norm(xk_next)>=D/2-1e-2:
            return xk_next, iter_cnt, fvals, times
        
        # print(f'Iteration {iter_cnt} - grad_norm:',np.linalg.norm(grad_next),"tk:",tk, "x_norm:",np.linalg.norm(xk_next))
        # mat_k = np.linalg.inv(f_hessian(xk_next))
        mat_k = update_approximation_bfgs(mat=mat_k, sk=sk, yk=grad_next-grad_k)
        xk = xk_next
    return xk_next, iter_cnt, fvals, times


def minimize_quadratic_on_l2_ball(g: np.ndarray, H: np.ndarray, R: float, inner_eps: float, params: dict,x_k: np.ndarray,gamma_k: float, isbfgs: bool, t_s: float) -> np.ndarray:
    n = g.shape[0]
    x_opt_ipm_damped, t_ipm_damped, duality_gaps_damped, fvals_, times_= barrier_method(t_init=params['t_init'], f=lambda x:f_int(x,params,x_k,gamma_k),
                            f_grad=lambda x:f_int_grad(x,params,x_k,gamma_k), f_hessian=lambda x:f_int_hess(x,params,x_k,gamma_k), phi=phi, phi_grad=phi_grad,
                            phi_hessian=phi_hessian, 
                A=params['A_o'], b=params['b'], x0=x_k, D=2*params['R'], num_constraints=1, params=params, t_s=t_s, method='bfgs' if isbfgs else 'newton', mu=10, epsilon=params['inner_eps'], maxIter=20)
    return x_opt_ipm_damped, fvals_, times_


def contracting_newton(params, c_0, decrease_gamma, isbfgs):
    # start_time = time.perf_counter()
    # last_logging_time = start_time
    # last_display_time = start_time
    t_s = time()
    n = params['A'].shape[1]
    m = params['A'].shape[0]
    inv_m = 1.0 / m
    data_accesses = m
    global num_newton
    num_newton=0
    x_k = params['x_0'].copy()
    fval_prev = np.average(np.log(1+np.exp(-params['b']*(params['A_o']@x_k))))+inv_m*params['lambda']*np.linalg.norm(x_k)**2
    func_val_record = []
    time_record=[]
    # Ax = params['A'].dot(x_k)
    Ax = params['A']@x_k
    g_k = np.zeros(n)
    H_k = np.zeros((n, n))
    v_k = np.zeros(n)

    gamma_str = f"gamma_k = {c_0}"
    if decrease_gamma:
        gamma_str += " / (3 + k)"
    # print(f"Contracting Newton Method, {gamma_str}")
    pbar=tqdm(range(params['n_iters'] ))
    fval = fval_prev
    
    for k in pbar:
        gamma_k = c_0
        if decrease_gamma:
            gamma_k /= 3.0 + k
            # gamma_k=c_0*(1-(k/(k+1))**3)
            # print("Gamma_k=",gamma_k)
        # print("Round:",k,flush=True)
        g_k = inv_m * (params['A'].T.dot(1 / (1 + np.exp(-Ax)))+2*params['lambda']*x_k)
        grad_norm=np.linalg.norm(g_k) 
        
        if (np.linalg.norm(g_k)<params['outer_eps'] or (k>=1 and abs(fval-fval_prev)/max(abs(fval_prev),1)<0.1*params['outer_eps'])):
            break
        H_k = (inv_m ) * (params['A'].T.dot(((1 / (1 + np.exp(-Ax))) * (1 - 1 / (1 + np.exp(-Ax))))[:, np.newaxis] * params['A'])) + inv_m*2*params['lambda']*np.diag([1.0]*x_k.size)

        # g_k -= H_k.dot(x_k)

        v_k, fvals, times = minimize_quadratic_on_l2_ball(g_k, H_k, params['R'], params['inner_eps'],params,x_k,gamma_k,isbfgs,t_s)
        func_val_record.extend(fvals.tolist())
        time_record.extend(times.tolist())
        x_k += gamma_k * (v_k - x_k)
        fval_prev=fval
        fval = np.average(np.log(1+np.exp(-params['b']*(params['A_o']@x_k))))+inv_m*params['lambda']*np.linalg.norm(x_k)**2
        func_val_record.append(fval)
        time_record.append(time()-t_s)
        Ax = params['A'].dot(x_k)
        data_accesses += m
        pbar.set_description('Function value: %.8f / Grad norm: %.8f'%(fval,grad_norm))
        # print("function value:",np.average(np.log(1+np.exp(-params['b']*(params['A_o']@x_k))))+inv_m*params['lambda']*np.linalg.norm(x_k)**2)
    print(f"Num newton:{num_newton}.")
    pbar.close()
    t_e=time()
    return x_k, t_e-t_s, np.array(func_val_record),np.array(time_record)
