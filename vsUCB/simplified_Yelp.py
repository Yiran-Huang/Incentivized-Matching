from DA_algorithm import *
from get_bound import *
from Matching_Method import *
from Matching_Method_non_covariates import *
import numpy as np
import os
import pickle
import random
from functools import partial

simu_num = 0

np.random.seed(1120220054 + simu_num)
torch.manual_seed(1120220054 + simu_num)

looptimes = 40
parallel_num = 1
total_agent_num = 120
d = 0
N = 10
M = 70
T = 120

data_use_now = "santab"
with open("../processed_data/Yelp_"+data_use_now +'.pkl', 'rb') as f:
    datause = pickle.load(f)

if not os.path.exists("result"):
    os.makedirs("result")

[stars_all, loclist, starlist, beta_all] = datause
beta_all = beta_all.t()
true_gamma = beta_all.mean(dim=1)
true_gamma = torch.ones(0, dtype=torch.float32)
# torch.tensor(np.array(starlist_uptonow),dtype=torch.float32).quantile(0.4)
true_const = torch.tensor([3.])
sigma = torch.tensor([1.])

def get_compare(T, d, N, M, true_beta, true_gamma, true_const, x_all, G_all, prob_all, locini, locother, newagent_time,
                Use_Incentive_Compatibility,Use_Individual_Rationality, truncate, get_c,
                error_use=0.05,methodname="",
                part_IC_matrix=None, program_name="",useora=False,useIUCB=False,useUCB=False):
    X = x_all[locini].clone()
    true_G = G_all[locini].clone()
    auxiliary_info=None

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
        return oc_baseline.unsqueeze(0).expand(N, num_agent) + torch.randn(N, num_agent) * 1

    def get_oc_baseline(x, pubnum):
        return torch.zeros(x.shape[0], dtype=torch.float32)

    prob_comes = prob_all[locini].clone()
    update_info = update_matrix
    device = "cpu"
    ini_data = None

    newagent_X = x_all[locother].clone()
    newagent_trustini = torch.ones(len(newagent_time)) * trust_ini_value
    newagent_G = G_all[locother].clone()
    newagent_prob = prob_all[locother].clone()
    newagent_auxiliary_info = None

    quota1 = None
    quota2 = torch.ones(N, dtype=torch.int64) * 3
    propose = 1
    largevalue = 1e5
    print_process = False

    def get_error(pulled_arm):
        return torch.randn(len(pulled_arm))

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
    if useIUCB:
        out1 = IUCB_noncovariate_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
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
        return out1
    if useUCB:
        out1 = UCB_noncovariate_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
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

        return out1


outall = []
tall1 = 0.
tall2 = 0.
tall3 = 0.
tall4 = 0.
tall5 = 0.
tall6 = 0.
for loop in range(looptimes):

    loc_store = np.random.choice(beta_all.shape[1], N, replace=False)
    true_beta = torch.ones(d, N, dtype=torch.float32)
    tempresult = []
    for i in loc_store:
        candidates = torch.tensor(stars_all[i])
        idx = torch.randint(0, len(candidates), (total_agent_num,))
        sampled = candidates[idx]
        tempresult.append(sampled)
    G_all = torch.tensor(torch.stack(tempresult), dtype=torch.float32).t()
    del tempresult

    x_all = torch.rand(total_agent_num, d) * 2

    prob_all = torch.rand(total_agent_num) * 0.2 + 0.2

    locall = torch.randperm(x_all.shape[0])
    locini = locall[:M]
    locother = locall[M:]
    newagent_time = torch.randint(0, T, (len(locother),), dtype=torch.int64)

    t1 = time.time()
    print(loop + 1, "/", looptimes,  flush=True)

    out_IUCB = get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                locini=locini, locother=locother, newagent_time=newagent_time,
                                Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,truncate=False,
                                get_c=c_simple,error_use=0.05,methodname="",
                                part_IC_matrix=None, program_name="IUCB",useIUCB=True)
    out_UCB = get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                locini=locini, locother=locother, newagent_time=newagent_time,
                                Use_Incentive_Compatibility=False,Use_Individual_Rationality=False,truncate=False,
                                get_c=c_simple,error_use=0.05,methodname="",
                                part_IC_matrix=None, program_name="UCB",useUCB=True)
    out_Ora_full = get_compare(T=T, d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                locini=locini, locother=locother, newagent_time=newagent_time,
                                Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,truncate=True,
                                get_c=c_simple,error_use=0.05,methodname="",
                                part_IC_matrix=None, program_name="Oracle Full",useora=True)

    outthis = [out_IUCB, out_UCB, out_Ora_full]
    outall.append(outthis)


    with open('result/simplified_yelp' + str(simu_num) + '.pkl', 'wb') as f:
        pickle.dump(outall, f)

