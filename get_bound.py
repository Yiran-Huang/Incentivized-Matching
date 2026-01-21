import numpy as np
from scipy.optimize import linprog
import torch
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import gc
import pickle



def c_simple(N, d, M, target):
    c_all = torch.eye(N*d+N*M).float()
    return c_all

def update_matrix(N, d, M, X_all, agent_number, recommend_arm,pull_times,
                     reward, pull_or_not,sigma,error_rate=0.05, Use_Incentive_Compatibility=True,
                  get_c = c_simple,target=None):
    error_rate_each = torch.tensor([error_rate/(d*N+M*N)])
    recommend_times = X_all.shape[0]
    model_matrixp1 = torch.zeros(recommend_times, d * N)
    model_matrixp1[
        torch.arange(recommend_times).unsqueeze(1), (torch.arange(d) + recommend_arm.unsqueeze(1) * d)] = X_all
    model_matrixp2 = torch.zeros(recommend_times, M * N)
    model_matrixp2[torch.arange(recommend_times), agent_number * N + recommend_arm] = 1.
    model_matrix = torch.cat((model_matrixp1, model_matrixp2), 1)
    Fx = model_matrix[pull_or_not]
    Fy = reward

    # When agent j pull arm i, only supermartingales related to beta_i part and g_{ji} are updatad
    # lambda_all is of shape (pull_or_not.sum())*(d+1)
    # The first d rows of lambda all are for some beta_i, and the last column is for some g_{ji}

    times_pull_i = pull_times.sum(dim=0)
    pulled_agent = agent_number[pull_or_not]
    pulled_arm = recommend_arm[pull_or_not]

    c_all = get_c(N=N, d=d, M=M, target=target)
    tempxxbar = torch.einsum('abc,acd->abd', Fx.unsqueeze(2), Fx.unsqueeze(1))
    tempxybar = Fx * Fy.unsqueeze(1)

    loc_mar_update = (pulled_arm * d).unsqueeze(1).expand(len(pulled_arm), d) + torch.arange(d)
    loc_mar_update = torch.cat((loc_mar_update, (pulled_agent * N + pulled_arm + d * N).unsqueeze(1)), 1)
    cneed = c_all[:, loc_mar_update].permute(1, 2, 0)
    cxx = torch.einsum('abc,acd->abd', cneed, tempxxbar)
    cxxc = (cneed * cxx).sum(dim=2)
    cxy = (tempxybar.unsqueeze(1) * cneed).sum(dim=2)
    sigma_use = sigma[pulled_arm].unsqueeze(1).expand(len(pulled_arm), d + 1)

    temp_pull_times = torch.tensor([torch.sum(pulled_arm[:i] == pulled_arm[i]).item() for i in range(len(pulled_arm))])
    temp_pull_times += (times_pull_i[pulled_arm]+1)
    lambda_part11 = 1 / (sigma_use[:, :d])
    lambda_part12 = (2 * torch.log(2 / error_rate_each) / temp_pull_times / torch.log(temp_pull_times + 1)) ** 0.5
    lambda_part1 = lambda_part11 * lambda_part12.unsqueeze(1)
    center_part1 = cxy[:, :d] * lambda_part1
    band_part1 = 0.5 * lambda_part1 ** 2 * sigma_use[:, :d] ** 2 * cxxc[:, :d]
    constraint_part1 = cxx[:, :d, :] * lambda_part1.unsqueeze(2)

    temp_each_pull_times = pull_times[pulled_agent, pulled_arm]+1
    lambda_part21 = 1 / (sigma_use[:, -1:])
    lambda_part22 = (2 * torch.log(2 / error_rate_each) / temp_each_pull_times / torch.log(
        temp_each_pull_times + 1)) ** 0.5
    lambda_part2 = lambda_part21 * lambda_part22.unsqueeze(1)
    center_part2 = cxy[:, -1:] * lambda_part2
    band_part2 = 0.5 * lambda_part2 ** 2 * sigma_use[:, -1:] ** 2 * cxxc[:, -1:]
    constraint_part2 = cxx[:, -1:, :] * lambda_part2.unsqueeze(2)

    center_part = torch.cat((center_part1, center_part2), 1)
    band_part = torch.cat((band_part1, band_part2), 1)
    constraint_part = torch.cat((constraint_part1, constraint_part2), 1)
    # Recover center_part and band_part to T*(M*N+d*N) and then to (M*N+d*N) by summation
    # Recover constraint_part to T*(M*N+d*N)*(M*N+d*N)and then to (M*N+d*N)*(M*N+d*N) by summation
    loc_au = torch.arange(len(pulled_arm)).unsqueeze(1).expand(len(pulled_arm), d + 1)
    center_recover = torch.zeros(len(pulled_arm), M * N + d * N)
    center_recover[loc_au, loc_mar_update] = center_part
    center_recover = center_recover.sum(dim=0)
    band_recover = torch.zeros(len(pulled_arm), M * N + d * N)
    band_recover[loc_au, loc_mar_update] = band_part
    band_recover = band_recover.sum(dim=0)
    constraint_recover = torch.zeros(len(pulled_arm), M * N + d * N, M * N + d * N)
    constraint_recover[loc_au, loc_mar_update] = constraint_part
    constraint_recover = constraint_recover.sum(dim=0)

    if len(pulled_arm) < 0.5:
        center_recover = torch.zeros(M*N+d*N)
        band_recover = torch.zeros(M*N+d*N)
        constraint_recover = torch.zeros(M*N+d*N,M*N+d*N)



    if Use_Incentive_Compatibility:
        IC_matrix = model_matrix.clone()
        IC_matrix = torch.cat((IC_matrix, -X_all, -torch.ones(recommend_times, 1)), 1)
        return constraint_recover, center_recover,band_recover, IC_matrix
    return constraint_recover, center_recover,band_recover,0


def combine_pull_n(matched_times,newIC_matrix,success_or_not,oc_baseline_coming_agent,ntimes=10.5,dropother=False):
    mt = matched_times.sum(dim=1)
    preserve_newIC_matrix=newIC_matrix[mt<ntimes]
    preserve_success_or_not = success_or_not[mt<ntimes]
    preserve_oc_baseline_coming_agent = oc_baseline_coming_agent[mt<ntimes]
    if dropother:
        newIC_matrix = preserve_newIC_matrix
        success_or_not = preserve_success_or_not
        oc_baseline_coming_agent = preserve_oc_baseline_coming_agent
    else:
        newIC_matrix = newIC_matrix[mt >= ntimes]
        success_or_not = success_or_not[mt >= ntimes]
        oc_baseline_coming_agent = oc_baseline_coming_agent[mt >= ntimes]

        part1 = (newIC_matrix[success_or_not]).mean(dim=0).unsqueeze(0)
        oc_baseline_part1 = (oc_baseline_coming_agent[success_or_not]).mean(dim=0).unsqueeze(0)
        if success_or_not.sum() < 0.5:
            part1 = torch.empty((0, newIC_matrix.shape[1]))
            oc_baseline_part1 = torch.empty((0))
        part2 = (newIC_matrix[~success_or_not]).mean(dim=0).unsqueeze(0)
        oc_baseline_part2 = (oc_baseline_coming_agent[~success_or_not]).mean(dim=0).unsqueeze(0)
        if (~success_or_not).sum() < 0.5:
            part2 = torch.empty((0, newIC_matrix.shape[1]))
            oc_baseline_part2 = torch.empty((0))
        pull_or_not_part1 = torch.tensor([True]) if part1.shape[0] > 0 else torch.tensor([], dtype=bool)
        pull_or_not_part2 = torch.tensor([False]) if part2.shape[0] > 0 else torch.tensor([], dtype=bool)

        newIC_matrix = torch.cat((part1, part2), 0)
        success_or_not = torch.cat((pull_or_not_part1, pull_or_not_part2))
        oc_baseline_coming_agent = torch.cat((oc_baseline_part1, oc_baseline_part2))

        newIC_matrix = torch.cat((newIC_matrix,preserve_newIC_matrix),0)
        success_or_not = torch.cat((success_or_not, preserve_success_or_not))
        oc_baseline_coming_agent = torch.cat((oc_baseline_coming_agent, preserve_oc_baseline_coming_agent))

    return newIC_matrix,success_or_not,oc_baseline_coming_agent

def get_predict_bound(N, d, M, xx_bar, xy_bar,band_part1,

                      coming_agent, agent_x,error_rate=0.05,need_lower=None,

                      IC_matrix=None, pull_or_not=None, oc_baseline=None,relax=True,

                      use_sparse=True, parallel_num=4, largevalue=1e5):
    # the coefficient is like eta^T = beta_1^T, beta_2^T...., g_{1,1}, g_{1,2},..., g_{1,N}, g_{2,1}...., g_{M,N}
    # when new agent joins, we can easily concat zeros to the model matrix at the end of the column
    # when agent j (with covariate x) pulled arm j, it model vector should be:
    # (0^T,..,x^T,0^T,0,..,0,1,0,...0)^T where x^T correspond to location of beta_i^T and
    # 1 correspond to location of g_{j,i}

    error_rate_each = torch.tensor([error_rate / (N*d+N*M)])
    band = band_part1 + torch.log(2/error_rate_each)
    A_ub = torch.cat((xx_bar, -xx_bar), 0)
    b_ub = torch.cat((xy_bar+band, -xy_bar+band))
    bounds = [(None, None)] * (d * N + N * M)

    #If we use slack variable s, we set lambda = 10 \max_i ||A_i||_2 * ||x||_i
    #but A_i will be very large when the matching number is large, which makes lambda extremely large
    #It may affect the stability of numerical result
    #Hence we normalize A_ub
    AL2 = (A_ub ** 2).sum(dim=1) ** 0.5+1e-2
    A_ub /= AL2.unsqueeze(1)
    b_ub /= AL2

    if IC_matrix is not None:
        # the coefficient now is eta^T, gamma^T , gamma0
        # the row of IC_matrix is like (0^T,..,x^T,0^T,0,..,0,1,0,...0,-x^T,-1)
        # If the number of recommended_x is too large, we can combine some of them
        # pull_or_not is a vector of True and False
        A_ub = torch.cat((A_ub, torch.zeros(2 * (d * N + N * M), (d + 1))), 1)
        pull_A = IC_matrix.clone()
        pull_b = oc_baseline.clone()
        pull_A[pull_or_not] *= -1
        pull_b[pull_or_not] *= -1

        A_ub = torch.cat((A_ub, pull_A), 0)
        b_ub = torch.cat((b_ub, pull_b))
        bounds = bounds + [(None, None)] * (d + 1)

        if relax:
            A_max = (((A_ub**2).sum(dim=1))**0.5).max()
            A_ub = torch.cat((A_ub,torch.zeros(A_ub.shape[0],1)),1)
            A_ub[:(2*d * N + 2*N * M),-1]=-1
            bounds = bounds+ [(0, None)]

    # c_list are the bounds we hope to finds
    # it arranges in this way
    # agent_come[0]'s,agent_come[1]'s,... reward on arm1
    # then agent_come[0]'s,agent_come[1]'s,... reward on arm2....

    # x^T beta part
    c_part1 = torch.block_diag(*([agent_x] * N))
    # G part
    rows = len(coming_agent) * N
    cols = N * M
    row_indices = torch.arange(rows)
    i_indices = row_indices // len(coming_agent)
    j_indices = row_indices % len(coming_agent)
    col_indices = coming_agent[j_indices] * N + i_indices
    c_part2 = torch.zeros(rows, cols, dtype=torch.float32)
    c_part2[torch.arange(rows), col_indices] = 1.0

    c_list = torch.cat((c_part1, c_part2), 1)

    if IC_matrix is not None:
        # gamma part
        c_part3 = torch.cat((torch.zeros(len(coming_agent) * N,d), -torch.zeros(len(coming_agent) * N, 1)), 1)
        c_list = torch.cat((c_list, c_part3), 1)


    if need_lower is None:
        # need_lower (length len(coming_agent)) indicate which agent need lower bound
        upper_or_lower = torch.zeros(len(coming_agent) * N, dtype=bool)
    else:
        upper_or_lower = need_lower.repeat(N)
    upper_or_lower = upper_or_lower.numpy()
    c_list[~upper_or_lower] *= -1
    if relax and IC_matrix is not None:
        c_norm = ((c_list ** 2).sum(dim=1)) ** 0.5
        lambda_relax = 10 * c_norm * A_max
        c_list = torch.cat((c_list,lambda_relax.unsqueeze(1)),1)

    c_list = c_list.numpy()
    A_ub = A_ub.numpy()

    if use_sparse:
        A_ub = csr_matrix(A_ub)

    def tempf(c):
        b_ub_now = b_ub.clone()
        b_ub_now = b_ub_now.numpy()
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub_now, bounds=bounds, method="highs",
                      options={'time_limit': 20,'presolve':False,'disp': False})
        return res.success, res.status, res.fun

    if parallel_num==1:
        results = [tempf(c) for c in c_list]
    else:
        results = Parallel(n_jobs=parallel_num)(
            delayed(tempf)(c)
            for c in c_list)
    del A_ub,b_ub,tempf,bounds
    gc.collect()

    results_arr = np.array(results, dtype=object)
    optimize_success = results_arr[:, 0]
    optimize_status = results_arr[:, 1]
    optimize_value = results_arr[:, 2]
    optimize_value[optimize_value == None] = -largevalue
    optimize_value = optimize_value.astype(float)
    optimize_value[np.isnan(optimize_value)] = -largevalue
    optimize_value[~upper_or_lower] *= -1
    optimize_value = optimize_value.reshape((len(coming_agent), N), order='F')
    optimize_value = torch.tensor(optimize_value, dtype=torch.float32)
    return optimize_value