import argparse
from tqdm import tqdm, trange
from time import time
from scipy.interpolate import interp1d
from libsvm.svmutil import svm_read_problem # https://blog.csdn.net/u013630349/article/details/47323883

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import contrnewtonstoch as cns
from runall_non_stoch import barrier_method
def read_data(path,readzero):
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
        if readzero or np.sum(row) > 1e-3:
            effective_row_ids.append(idx)
    return b[effective_row_ids], A_np[effective_row_ids]
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
            if f(xk-tk*grad) <= f(xk)-alpha*tk*grad.T@grad:
                break
        tk *= beta
    return tk
def project(x,D):
    x_norm = np.linalg.norm(x)
    if x_norm <= D/2:
        return x
    coef = D/2/x_norm
    return coef*x

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
#* 投影梯度法
def projected_sgd(f, f_grad, x0, D, t_hat=1, epsilon=1e-6, batch_size=8192, epochs=50, variance_reduction=False):
    func_val_record = []
    time_record=[]
    xk = x0
    xk_norm = np.linalg.norm(xk)
    t_s = time()
    inv_m=1/A.shape[0]
    full_g_k=0
    for epoch in range(epochs):
        if variance_reduction and (epoch==0 or abs(2**np.floor(np.log2(epoch)) - epoch)<1e-6):
            # print(k)
            Ax = -np.multiply(b,A.T).T @ xk
            full_g_k = inv_m * (-np.multiply(b,A.T).dot(np.array([Sigmoid(ax) for ax in Ax])))
            z_k = xk.copy()
        for batch_A, batch_b in data_generator(A, b, batch_size):    
            tk = armijo_search(f=lambda x:f(x,batch_A,batch_b), f_grad=lambda x:f_grad(x,batch_A,batch_b), 
                                xk=xk, t_hat=t_hat, alpha=0.1, beta=0.5, D=D)
            g_k=f_grad(xk,batch_A,batch_b)+full_g_k-(f_grad(z_k,batch_A,batch_b) if variance_reduction else 0)
            xk_next = project(xk-tk*g_k, D)
            norm_diff = np.linalg.norm(xk_next-xk)
            if norm_diff<=epsilon:
                break
            xk = xk_next
        fval = f(xk,A,b)
        grad_norm = np.linalg.norm(f_grad(xk,A,b))
        func_val_record.append(fval)
        time_record.append(time()-t_s)
        print(f'Epoch {epoch} - Grad. Norm.:',grad_norm, 'F val.:',fval, 'Norm. Diff.:',norm_diff, 'x_norm:', np.linalg.norm(xk))
    t_e = time()
    return xk_next, np.asarray(func_val_record), t_e-t_s, np.asarray(time_record)
def write_tsv(fvals, times, fopt, filep):
    with open(filep,'w') as f:
        f.write('Epoch\tf-f*\tTime\n')
        for i in range(fvals.shape[0]):
            f.write('%d\t%.8f\t%.4f\n'%(i,fvals[i]-fopt,times[i]))
def write_tsv2(epochs,fvals, times, fopt, filep):
    with open(filep,'w') as f:
        f.write('Epoch\tf-f*\tTime\n')
        for i in range(fvals.shape[0]):
            f.write('%d\t%.8f\t%.4f\n'%(epochs[i],fvals[i]-fopt,times[i]))
parser = argparse.ArgumentParser(description='Train All stoch')
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--diameter', type=float, required=True)
parser.add_argument('--lamda', type=float, required=True)
# parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--rm_zeros', type=int, required=True)
parser.add_argument('--convert', type=int, default=1)
parser.add_argument('--maxiter', type=int, default=400)
args = parser.parse_args()

if __name__ == "__main__":
    data_path = args.data_path
    output_path = args.save_path
    max_iter=args.maxiter
    D = args.diameter
    lamda = args.lamda
    # batch_size = args.batch_size
    rm_zeros = True if args.rm_zeros else False
    seed = 1000

    b, A = read_data(data_path,not rm_zeros)
    if(args.convert == 1):
        b = 2*b-3
    m,n = A.shape
    # b=np.expand_dims(b, axis=1)
    params=dict()
    params['A']=-np.multiply(b,A.T).T
    params['A_o']=A
    params['b']=b
    # print(params['A'][0,:])
    print(np.sum(A==params['A']))
    # params['x_0']=np.zeros(n)
    c_0 = 3.0
    params['R']=D/2
    params['inner_eps']=1e-7
    params['outer_eps']=1e-4
    params['n_iters']=max_iter # 400次Doikov迭代相当于50个convtype epoch
    # params['n_iters_newton']=800
    params['lambda']=1/lamda
    params['t_init']=1
    history=None
    decrease_gamma=True


    def f(x,A,b):
        m,n = A.shape
        bAx = b*(A@x)
        exp_mbAx = np.exp(-bAx)
        log1p_exp = np.log(1+exp_mbAx)
        overflow_idxs = np.where(exp_mbAx==float('inf'))
        log1p_exp[overflow_idxs] = -bAx[overflow_idxs]
        return log1p_exp.mean() + 1/(lamda*m)* x.T@x

    def f_grad(x,A,b):
        m,n = A.shape
        return np.ones(m)@(np.expand_dims((-b)/(1+np.exp(b*(A@x))), axis=1)*A)/m + 2/(lamda*m)*x

    def f_hessian(x,A,b):
        m,n = A.shape
        Ax = A@x
        exp_bAx = np.exp(b*Ax)
        return (A.T @ (np.expand_dims(b*b*exp_bAx/(1+exp_bAx)**2, axis=1)*A) )/m + 2/(lamda*m)*np.eye(x.size)
    
    #* logarithm barrier
    def phi(x,D):
        real = D/2-np.linalg.norm(x,ord=2)
        return -np.log(real) if real>0 else float("inf")
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

    def data_generator(A, b, batch_size=8192):
        m,n = A.shape
        new_idxs = np.arange(m)
        np.random.shuffle(new_idxs)
        A = A[new_idxs]
        b = b[new_idxs]
        numBatches = m//batch_size+1
        for i in range(numBatches):
            idx_begin = i*batch_size
            idx_end = (i+1)*batch_size
            yield (A[idx_begin:idx_end], b[idx_begin:idx_end])

   #* 高精度 - 求解问题最优解
    np.random.seed(seed)
    t_init = 1
    x0 = np.zeros(n)+0.001
    x_opt, t, _, _,_ = barrier_method(t_init=t_init, f=lambda x:f(x,A,b), f_grad=lambda x:f_grad(x,A,b), f_hessian=lambda x:f_hessian(x,A,b), phi=phi, phi_grad=phi_grad, phi_hessian=phi_hessian, 
                    A=A, b=b, x0=x0, D=D, num_constraints=1, method='newton', mu=10, epsilon=1e-10, maxIter=20)
    print(f'求解问题最优解 - 最小值: {f(x_opt,A,b):>2f}\t耗时: {t:>2f}s')
    fopt = f(x_opt,A,b)
    # print(f(x0,A,b))
    # 投影法
    np.random.seed(1000)
    init_x = np.zeros(n)+0.001
    x_opt_spgd, fvals_spgd, t_spgd,times_spgd = projected_sgd(f=f, f_grad=f_grad, x0=init_x, D=D, t_hat=5, epsilon=1e-4, epochs=50)
    print(f'最小值: {f(x_opt_spgd,A,b):>2f}\t耗时: {t_spgd:>2f}s')
    # 投影法（方差缩减）
    np.random.seed(1000)
    init_x = np.zeros(n)+0.001
    x_opt_spgd_vr, fvals_spgd_vr, t_spgd_vr,times_spgd_vr = projected_sgd(f=f, f_grad=f_grad, x0=init_x, D=D, t_hat=5, epsilon=1e-4, epochs=50, variance_reduction=True)
    print(f'最小值: {f(x_opt_spgd_vr,A,b):>2f}\t耗时: {t_spgd_vr:>2f}s')

    # 收缩域牛顿（无方差缩减）
    np.random.seed(1000)
    init_x = np.zeros(n)+0.001
    params['x_0']=init_x
    x_opt_vtr,fvals_ctr, times_ctr, epochs_ctr = cns.contracting_newton(params, c_0, decrease_gamma, False)
    print(f'最小值: {f(x_opt_vtr,A,b):>2f}\t耗时: {times_ctr[-1]:>2f}s')
    # 收缩域牛顿（方差缩减）
    np.random.seed(1000)
    init_x = np.zeros(n)+0.001
    params['x_0']=init_x
    x_opt_vtr_vr,fvals_ctr_vr, times_ctr_vr, epochs_ctr_vr = cns.contracting_newton(params, c_0, decrease_gamma, True)
    print(f'最小值: {f(x_opt_vtr_vr,A,b):>2f}\t耗时: {times_ctr_vr[-1]:>2f}s')
 

    write_tsv(fvals_spgd,times_spgd,fopt,args.save_path+".spgd.csv")
    write_tsv(fvals_spgd_vr,times_spgd_vr,fopt,args.save_path+".spgd_vr.csv")
    write_tsv2(epochs_ctr_vr,fvals_ctr_vr,times_ctr_vr,fopt,args.save_path+".ctr_vr.csv")
    write_tsv2(epochs_ctr,fvals_ctr,times_ctr,fopt,args.save_path+".ctr.csv")