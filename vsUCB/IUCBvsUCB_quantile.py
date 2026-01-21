from DA_algorithm import *
from get_bound import *
from Matching_Method import *
from Matching_Method_non_covariates import *
import numpy as np
import os
import pickle
simu_num = 0

np.random.seed(1120220054 + simu_num)
torch.manual_seed(1120220054 + simu_num)

looptimes = 1000
parallel_num = 1
total_agent_num = 1
d = 0
N = 6
M = 1
T = 200

if not os.path.exists("result"):
    os.makedirs("result")

def get_compare(T, d, N, M, true_beta, true_gamma, true_const, x_all, G_all, prob_all, locini, locother, newagent_time,
                Use_Incentive_Compatibility,Use_Individual_Rationality, truncate, get_c,
                error_use=0.05,methodname="",
                part_IC_matrix=None, program_name="",useora=False):
    X = x_all[locini].clone()
    true_G = G_all[locini].clone()

    trust_incre = torch.tensor(1.)
    trust_decre = torch.tensor(5.)
    trust_ini_value = 10.1
    trust_ini = torch.ones(M) * trust_ini_value

    def trans_X(X,auxiliary_info, agent_comes, pullarm):
        if agent_comes is not None:
            X[agent_comes] = torch.rand(len(agent_comes), X.shape[1]) * 2
        return X,None

    def get_v(oc_baseline, N):
        num_agent = len(oc_baseline)
        return oc_baseline.unsqueeze(0).expand(N, num_agent) + torch.randn(N, num_agent) * 0.1

    def get_oc_baseline(x,y):
        return (torch.rand(x.shape[0]) * 0.3 + 0.01)*10
        #return torch.rand(x.shape[0])*0.05+0.2

    prob_comes = prob_all[locini].clone()
    sigma = torch.tensor([50.])
    update_info = update_matrix
    device = "cpu"
    ini_data = None

    newagent_X = x_all[locother].clone()
    newagent_trustini = torch.ones(len(newagent_time)) * trust_ini_value
    newagent_G = G_all[locother].clone()
    newagent_prob = prob_all[locother].clone()

    quota1 = None
    quota2 = torch.tensor([5,5,5,5,5,5], dtype=torch.int64)
    quota2 = torch.ones(6, dtype=torch.int64) * 2
    propose = 1
    largevalue = 1e5
    print_process = False
    def get_error(x):
        return torch.randn(len(x))*50
    use_sparse = True

    if useora:
        out1 = matching_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
                        trust_incre=trust_incre, trust_decre=trust_decre, trust_ini=trust_ini, trans_X=trans_X,
                        get_v=get_v, get_oc_baseline=get_oc_baseline, prob_comes=prob_comes, T=T, method_use=Oracle,
                        sigma=sigma, error_use=error_use,Use_Incentive_Compatibility=Use_Incentive_Compatibility,
                        get_c=get_c, update_info=update_info, truncate=truncate,
                        device=device, ini_data=ini_data,
                        newagent_time=newagent_time, newagent_X=newagent_X, newagent_trustini=newagent_trustini,
                        newagent_G=newagent_G, newagent_prob=newagent_prob,
                        quota1=quota1, quota2=quota2, propose=propose, largevalue=largevalue,
                        program_name=program_name,print_process=print_process, get_error=get_error,
                        use_sparse=use_sparse, parallel_num=parallel_num,
                        part_IC_matrix=part_IC_matrix, target = None)
        return out1
    out1 = IUCB_noncovariate_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
                        trust_incre=trust_incre, trust_decre=trust_decre, trust_ini=trust_ini, trans_X=trans_X,
                        get_v=get_v, get_oc_baseline=get_oc_baseline, prob_comes=prob_comes, T=T,
                        Use_Individual_Rationality=Use_Individual_Rationality,
                        sigma=sigma, error_use=error_use,Use_Incentive_Compatibility=Use_Incentive_Compatibility,
                        get_c=get_c, update_info=update_info, truncate=truncate,
                        device=device, ini_data=ini_data,
                        newagent_time=newagent_time, newagent_X=newagent_X, newagent_trustini=newagent_trustini,
                        newagent_G=newagent_G, newagent_prob=newagent_prob,
                        quota1=quota1, quota2=quota2, propose=propose, largevalue=largevalue,
                        program_name=program_name,print_process=print_process, get_error=get_error,
                        use_sparse=use_sparse, parallel_num=parallel_num,
                        part_IC_matrix=part_IC_matrix, target = None)
    out2 = UCB_noncovariate_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
                        trust_incre=trust_incre, trust_decre=trust_decre, trust_ini=trust_ini, trans_X=trans_X,
                        get_v=get_v, get_oc_baseline=get_oc_baseline, prob_comes=prob_comes, T=T,
                        sigma=sigma, error_use=error_use,Use_Incentive_Compatibility=Use_Incentive_Compatibility,
                        Use_Individual_Rationality=Use_Individual_Rationality,
                        get_c=get_c, update_info=update_info, truncate=truncate,
                        device=device, ini_data=ini_data,
                        newagent_time=newagent_time, newagent_X=newagent_X, newagent_trustini=newagent_trustini,
                        newagent_G=newagent_G, newagent_prob=newagent_prob,
                        quota1=quota1, quota2=quota2, propose=propose, largevalue=largevalue,
                        program_name=program_name,print_process=print_process, get_error=get_error,
                        use_sparse=use_sparse, parallel_num=parallel_num,
                        part_IC_matrix=part_IC_matrix, target = None)

    return out1,out2


outall = []
tall1 = 0.
tall2 = 0.
tall3 = 0.
tall4 = 0.
tall5 = 0.
tall6 = 0.
for loop in range(looptimes):
    print(loop + 1, "/", looptimes,flush=True)

    error_use_now = 0.001

    true_beta = torch.ones(d, N, dtype=torch.float32)
    true_gamma = torch.ones(d, dtype=torch.float32)
    true_const = torch.tensor([0.0])

    x_all = torch.rand(total_agent_num, d) * 2
    G_all = (((torch.randperm(N)/2).floor().reshape(1,N))*(torch.ones(M).unsqueeze(1)) / (N/2 - 1)) ** 0.5 * 0.1 + 0.2
    G_all = torch.tensor([0.01,0.01,0.31,0.31,0.61,0.61])[torch.randperm(N)].reshape(M, N)*10
    prob_all = torch.ones(total_agent_num) * 1

    locall = torch.randperm(x_all.shape[0])
    locini = locall[:M]
    locother = locall[M:]
    newagent_time = torch.randint(0, T, (len(locother),), dtype=torch.int64)
    t1 = time.time()
    out_IUCB,out_UCB_withc_withr=get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                locini=locini,locother=locother, newagent_time=newagent_time,
                Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,
                truncate=False,get_c=c_simple, error_use=error_use_now, methodname="",
                part_IC_matrix=None, program_name="With both IC and IR")
    t2 = time.time()
    tall1 += (t2 - t1)
    t3 = time.time()
    out_IUCB_c,out_UCB_withr=get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                locini=locini,locother=locother, newagent_time=newagent_time,
                Use_Incentive_Compatibility=False,Use_Individual_Rationality=True,
                truncate=False,get_c=c_simple, error_use=error_use_now, methodname="",
                part_IC_matrix=None, program_name="With IR")
    t4 = time.time()
    tall2 += (t4 - t3)
    t5 = time.time()
    out_IUCB_r,out_UCB_withc=get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                locini=locini,locother=locother, newagent_time=newagent_time,
                Use_Incentive_Compatibility=True,Use_Individual_Rationality=False,
                truncate=False,get_c=c_simple, error_use=error_use_now, methodname="",
                part_IC_matrix=None, program_name="With IC")
    t6 = time.time()
    tall3 += (t6 - t5)
    t7 = time.time()
    out_IUCB_c_r,out_UCB=get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                locini=locini,locother=locother, newagent_time=newagent_time,
                Use_Incentive_Compatibility=False,Use_Individual_Rationality=False,
                truncate=False,get_c=c_simple, error_use=error_use_now, methodname="",
                part_IC_matrix=None, program_name="Without IC and IR")
    t8 = time.time()
    tall4 += (t8 - t7)
    t7 = time.time()
    out_Ora_full=get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                locini=locini,locother=locother, newagent_time=newagent_time,
                Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,
                truncate=True,get_c=c_simple, error_use=error_use_now, methodname="",
                part_IC_matrix=None, program_name="Oracle Full",useora=True)
    t8 = time.time()
    tall5 += (t8 - t7)
    t7 = time.time()
    out_Ora_part=get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                locini=locini,locother=locother, newagent_time=newagent_time,
                Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,
                truncate=False,get_c=c_simple, error_use=error_use_now, methodname="",
                part_IC_matrix=None, program_name="Oracle Full",useora=True)
    t8 = time.time()
    tall6 += (t8 - t7)
    outthis = [out_IUCB,out_UCB_withc_withr,out_IUCB_c,out_UCB_withr,
               out_IUCB_r,out_UCB_withc,out_IUCB_c_r,out_UCB,
               out_Ora_full,out_Ora_part]
    outall.append(outthis)

    with open('result/IUCBvsUCB_quantile'+str(simu_num)+'.pkl', 'wb') as f:
        pickle.dump(outall, f)
