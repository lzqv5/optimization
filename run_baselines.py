from tqdm import tqdm, trange
from time import time
from scipy.interpolate import interp1d
from libsvm.svmutil import svm_read_problem # https://blog.csdn.net/u013630349/article/details/47323883

import json, argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#* 数据读取
def read_data(path, rm_zeros=True):
    b, A = svm_read_problem(path)
    rows = len(b)   # 矩阵行数, i.e. sample 数
    cols = max([max(row.keys()) if len(row)>0 else 0 for row in A])  # 矩阵列数, i.e. feature 数
    b = np.array(b)
    A_np = np.zeros((rows,cols))
    for r in range(rows):
        for c in A[r].keys():
            # MatLab 是 1-index, python 则是 0-index
            A_np[r,c-1] = A[r][c]
    if rm_zeros:
        # 清除全 0 features
        effective_row_ids = []
        for idx, row in enumerate(A_np):
            if np.sum(np.abs(row)) > 1e-3:
                effective_row_ids.append(idx)
        return b[effective_row_ids], A_np[effective_row_ids]
    return b, A_np

#* 实验结果保存
def save_as_json_files(objs, filename):
    with open(filename, 'w') as f:
        json.dump(objs, f)

#* Armijo rule
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

#* 外部迭代
def barrier_method(t_init, f, f_grad, f_hessian, phi, phi_grad, phi_hessian, A, b, x0, D, num_constraints, mu,
                        method='newton', epsilon=1e-6, maxIter=20):
    xt = x0
    t = t_init
    duality_gaps = []
    func_val_record = []
    t_s = time()
    for i in trange(maxIter):
        xt,num_newton_step, fvals = solve_central(objective=f,
                                f=lambda x:t*f(x)+phi(x,D), 
                                f_grad=lambda x:t*f_grad(x)+phi_grad(x,D), 
                                f_hessian=lambda x:t*f_hessian(x)+phi_hessian(x,D),
                                x0=xt, D=D, method=method, epsilon=epsilon*1e3)
        duality_gaps.extend([num_constraints/t]*num_newton_step)
        func_val_record.extend(fvals)
        if num_constraints/t < epsilon:
            break
        t *= mu
    t_e = time()
    return xt, t_e-t_s, np.array(duality_gaps), np.array(func_val_record)

#* 中心问题求解
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
    for idx in range(max_iter):
        iter_cnt += 1
        fvals.append(objective(xk))
        grad = f_grad(xk)
        hessian = f_hessian(xk)
        dk = -np.linalg.inv(hessian)@grad
        decrement = (-grad@dk)**0.5
        if decrement**2/2 <= epsilon:
            # print('** End The Loop - Iter Cnt.:',iter_cnt, 'Decrement:',decrement, 'fval:',f(xk))
            return xk, iter_cnt, fvals
        tk = armijo_search(f, f_grad, xk, t_hat=1, alpha=0.1, beta=0.5, D=D, isNewton=True, dk=dk)
        # print('Iter Cnt.:',iter_cnt, 'Decrement:',decrement, 'fval:',f(xk), 'tk:',tk)
        xk += tk*dk
    return xk, iter_cnt, fvals

#* 拟牛顿
def bfgs(objective, f, f_grad, f_hessian, x0, D, alpha=0.1, beta=0.5, epsilon=1e-6, max_iter=500):
    xk = x0
    hessian = f_hessian(x0)
    mat_k = np.linalg.inv(hessian) 
    # mat_k = np.eye(n) 
    iter_cnt = 0
    fvals = []
    for idx in range(max_iter):
        iter_cnt += 1
        grad_k = f_grad(xk)
        dk = -mat_k@grad_k 
        tk = wolfe_condition(f, f_grad, xk, dk, D, c1=1e-4, c2=0.9)
        if tk<0:
            return xk, iter_cnt-1, fvals
        fvals.append(objective(xk))
        sk = tk*dk
        xk_next = xk + sk
        grad_next = f_grad(xk_next)
        # if np.linalg.norm(grad_next, ord=2) <= epsilon:
        if np.linalg.norm(grad_next, ord=2) <= epsilon or np.linalg.norm(xk_next)>=D/2-1e-2:
            return xk_next, iter_cnt, fvals
        # print(f'Iteration {iter_cnt} - grad_norm:',np.linalg.norm(grad_next),"tk:",tk, "x_norm:",np.linalg.norm(xk_next))
        # mat_k = np.linalg.inv(f_hessian(xk_next))
        mat_k = update_approximation_bfgs(mat=mat_k, sk=sk, yk=grad_next-grad_k)
        xk = xk_next
    return xk_next, iter_cnt, fvals

#* 更新 Bk or Hk
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

#* Wolfe Condition
def wolfe_condition(f, f_grad, xk, pk, D, c1=1e-4, c2=0.9, multiplier=1.2, t0=0, tmax=2):
    while (np.linalg.norm(xk+tmax*pk)>=D/2):
        tmax /= 2
        if tmax<1e-6:
            return -1
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

#* Projection 
def project(x,D):
    x_norm = np.linalg.norm(x)
    if x_norm <= D/2:
        return x
    coef = D/2/x_norm
    return coef*x

#* 投影(次)梯度法
def projected_gradient_descent(f, f_grad, x0, D, t_hat=1, epsilon=1e-6, max_iters=10000):
    func_val_record = [f(x0)]
    # grad_norm_record = [np.linalg.norm(f_grad(x0))]
    xk = x0
    xk_norm = np.linalg.norm(xk)
    t_s = time()
    # for idx in range(max_iters):
    for idx in trange(max_iters):
        tk = armijo_search(f, f_grad, xk, t_hat=t_hat, alpha=0.1, beta=0.5, D=D)
        xk_next = project(xk-tk*f_grad(xk), D)
        fval_xk_next = f(xk_next)
        grad_xk_next = f_grad(xk_next)
        func_val_record.append(fval_xk_next)
        grad_norm_next = np.linalg.norm(grad_xk_next,ord=2)
        # grad_norm_record.append(grad_norm_next)
        norm_diff = np.linalg.norm(xk_next-xk)
        if norm_diff<=epsilon:
            break
        # print(f'Iteration {idx} - Grad. Norm.:',grad_norm_next, 'Norm. Diff.:',norm_diff,'tk:',tk, 'x_norm:',np.linalg.norm(xk_next))
        xk = xk_next
    t_e = time()
    # return xk_next, t_e-t_s, np.array(func_val_record), np.array(grad_norm_record)
    return xk_next, t_e-t_s, np.array(func_val_record)


parser = argparse.ArgumentParser(description='Train Baselines')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--diameter', type=float, required=True)
parser.add_argument('--lamda', type=float, required=True)
# parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--rm_zeros', type=int, required=True)
args = parser.parse_args()

# e.g. python run_baselines.py --data_path w8a --save_path D_20_test.json --diameter 20 --lamda 100 --rm_zeros 0

if __name__ == "__main__":
    data_path = args.data_path
    output_path = args.save_path
    D = args.diameter
    lamda = args.lamda
    # batch_size = args.batch_size
    rm_zeros = True if args.rm_zeros else False
    seed = 1000

    b, A = read_data(data_path)
    m,n = A.shape

    #* objective
    def f(x):
        bAx = b*(A@x)
        exp_mbAx = np.exp(-bAx)
        log1p_exp = np.log(1+exp_mbAx)
        overflow_idxs = np.where(exp_mbAx==float('inf'))
        log1p_exp[overflow_idxs] = -bAx[overflow_idxs]
        return log1p_exp.mean() + 1/(lamda*m)* x.T@x

    def f_grad(x):
        return np.ones(m)@(np.expand_dims((-b)/(1+np.exp(b*(A@x))), axis=1)*A)/m + 2/(lamda*m)*x

    def f_hessian(x):
        Ax = A@x
        exp_bAx = np.exp(b*Ax)
        return (A.T @ (np.expand_dims(b*b*exp_bAx/(1+exp_bAx)**2, axis=1)*A) )/m + 2/(lamda*m)*np.eye(x.size)
    
    #* logarithm barrier
    def phi(x,D):
        return -np.log(D/2-np.linalg.norm(x,ord=2))
        # return -np.log(D**2/4-x@x)

    def phi_grad(x,D):
        x_norm = np.linalg.norm(x,ord=2)
        return x/(x_norm*(D/2-x_norm))
        # return 8/(D**2-4*x@x)*x

    def phi_hessian(x,D):
        x_norm = np.linalg.norm(x,ord=2)
        # xTx = x@x
        xxT = np.matmul(x[:,None],x[None,:])    # x * xT
        return np.eye(x.size)/(x_norm*(D/2-x_norm)) + (2*x_norm-D/2)/(x_norm**3 * (D/2-x_norm)**2)*xxT
        # return 4/((D**2-xTx)**2)*xxT + 8/(D**2-4*x@x)*np.eye(x.size)
    
    #* 高精度 - 求解问题最优解
    np.random.seed(seed)
    t_init = 1
    x0 = np.zeros(n)+0.005
    x_opt, t, _, _ = barrier_method(t_init=t_init, f=f, f_grad=f_grad, f_hessian=f_hessian, phi=phi, phi_grad=phi_grad, phi_hessian=phi_hessian, 
                    A=A, b=b, x0=x0, D=D, num_constraints=1, method='newton', mu=10, epsilon=1e-10, maxIter=20)
    print(f'求解问题最优解 - 最小值: {f(x_opt):>2f}\t耗时: {t:>2f}s')

    #* Dampe Newton
    np.random.seed(seed)
    t_init = 1
    x0 = np.zeros(n)+0.005
    x_opt_ipm_damped, t_ipm_damped, duality_gaps_damped, fvals_damped = barrier_method(t_init=t_init, f=f, f_grad=f_grad, f_hessian=f_hessian, phi=phi, phi_grad=phi_grad, phi_hessian=phi_hessian, 
                    A=A, b=b, x0=x0, D=D, num_constraints=1, method='newton', mu=10, epsilon=1e-6, maxIter=20)
    print(f'阻尼牛顿 - 最小值: {f(x_opt_ipm_damped):>2f}\t耗时: {t_ipm_damped:>2f}s')

    #* Quasi Newton - BFSG
    np.random.seed(seed)
    t_init = 1
    x0 = np.zeros(n)+0.005
    x_opt_ipm_bfgs, t_ipm_bfgs, duality_gaps_bfgs, fvals_bfgs = barrier_method(t_init=t_init, f=f, f_grad=f_grad, f_hessian=f_hessian, phi=phi, phi_grad=phi_grad, phi_hessian=phi_hessian, 
                    A=A, b=b, x0=x0, D=D, num_constraints=1, method='bfgs', mu=10, epsilon=1e-6, maxIter=20)
    print(f'拟牛顿 BFGS - 最小值: {f(x_opt_ipm_bfgs):>2f}\t耗时: {t_ipm_bfgs:>2f}s')

    #* Projectd gd
    np.random.seed(seed)
    init_x = np.zeros(n)+0.005
    # x_opt_pgd, t_pgd, fvals_pgd, grad_norm_pgd = projected_gradient_descent(f=f, f_grad=f_grad, x0=init_x, D=D, t_hat=5, epsilon=1e-6, max_iters=4010)
    x_opt_pgd, t_pgd, fvals_pgd = projected_gradient_descent(f=f, f_grad=f_grad, x0=init_x, D=D, t_hat=5, epsilon=1e-6, max_iters=4010)
    print(f'投影次梯度 - 最小值: {f(x_opt_pgd):>2f}\t耗时: {t_pgd:>2f}s')

    # 保存计算迭代的计算结果
    results = {
        'newton': (fvals_damped.tolist(), t_ipm_damped, x_opt_ipm_damped.tolist()),
        'bfgs': (fvals_bfgs.tolist(), t_ipm_bfgs, x_opt_ipm_bfgs.tolist()),
        'gd': (fvals_pgd.tolist(), t_pgd, x_opt_pgd.tolist()),
        'fval_opt': f(x_opt)
    }    
    
    save_as_json_files(results, filename=output_path)
