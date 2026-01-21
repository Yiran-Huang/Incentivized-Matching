from DA_algorithm import *
from get_bound import *
from Matching_Method import *
import numpy as np
import os
import pickle
from functools import partial

simu_num = int(os.path.basename(__file__)[4])

np.random.seed(1120220054 + simu_num)
torch.manual_seed(1120220054 + simu_num)

looptimes = 8
parallel_num = 6
total_agent_num = 120
d = 12
N = 10
M = 70
T = 120

if not os.path.exists("result"):
    os.makedirs("result")

data_use_now = "santab"
with open("../processed_data/Yelp_"+data_use_now +'.pkl', 'rb') as f:
    datause = pickle.load(f)

[stars_all, loclist, starlist, beta_all] = datause
beta_all = beta_all.t()
true_gamma = beta_all.mean(dim=1)
true_const = torch.tensor([3.])
sigma = torch.tensor([1.])


def get_compare(T, d, N, M, true_beta, true_gamma, true_const, x_all, G_all, prob_all, locini, locother, newagent_time,
                method_use, Use_Incentive_Compatibility,Use_Individual_Rationality, truncate, get_c,auxiliary_info_all,
                error_use=0.05,part_IC_matrix=None, program_name=""):
    X = x_all[locini].clone()
    true_G = G_all[locini].clone()
    if auxiliary_info_all is not None:
        auxiliary_info = auxiliary_info_all[locini].clone()
    else:
        auxiliary_info=None

    trust_incre = torch.tensor(1.)
    trust_decre = torch.tensor(5.)
    trust_ini_value = 10.1
    trust_ini = torch.ones(M) * trust_ini_value

    def trans_X(X, auxiliary_info, agent_comes, pullarm):
        if agent_comes is not None:
            temp1 = torch.randint(0, X.shape[1], (len(agent_comes),))
            temp2 = torch.zeros(len(agent_comes), X.shape[1], dtype=torch.float32)
            temp3 = (torch.randint(0, 3, (len(agent_comes),)) + 1) * 2 / 3
            temp2[torch.arange(len(agent_comes)), temp1] = 1.
            temp2 *= temp3.unsqueeze(1)
            X[agent_comes] = temp2
        return X, auxiliary_info

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
    if auxiliary_info_all is not None:
        newagent_auxiliary_info = auxiliary_info_all[locother].clone()
    else:
        newagent_auxiliary_info = None

    quota1 = None
    quota2 = torch.ones(N, dtype=torch.int64) * 3
    propose = 1
    largevalue = 1e5
    print_process = True

    def get_error(pulled_arm):
        return torch.randn(len(pulled_arm))

    use_sparse = True

    out = matching_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
                        trust_incre=trust_incre, trust_decre=trust_decre, trust_ini=trust_ini, trans_X=trans_X,
                        get_v=get_v, get_oc_baseline=get_oc_baseline, prob_comes=prob_comes, T=T, method_use=method_use,
                        sigma=sigma, auxiliary_info=auxiliary_info, error_use=error_use,
                        Use_Incentive_Compatibility=Use_Incentive_Compatibility,
                        Use_Individual_Rationality=Use_Individual_Rationality,
                        get_c=get_c, update_info=update_info, truncate=truncate,
                        device=device, ini_data=ini_data,
                        newagent_time=newagent_time, newagent_X=newagent_X, newagent_trustini=newagent_trustini,
                        newagent_G=newagent_G, newagent_prob=newagent_prob,
                        newagent_auxiliary_info=newagent_auxiliary_info,
                        quota1=quota1, quota2=quota2, propose=propose, largevalue=largevalue,
                        program_name=program_name,print_process=print_process, get_error=get_error,
                        use_sparse=use_sparse, parallel_num=parallel_num,
                        part_IC_matrix=part_IC_matrix, target = None)
    return out


outall = []
tall1 = 0.
tall2 = 0.
tall3 = 0.
tall4 = 0.
tall5 = 0.
tall6 = 0.
for loop in range(looptimes):
    print(loop + 1, "/", looptimes, flush=True)

    loc_store = np.random.choice(beta_all.shape[1], N, replace=False)
    true_beta = beta_all[:, loc_store].clone()
    tempresult = []
    for i in loc_store:
        candidates = torch.tensor(stars_all[i])
        idx = torch.randint(0, len(candidates), (total_agent_num,))
        sampled = candidates[idx]
        tempresult.append(sampled)
    G_all = torch.tensor(torch.stack(tempresult), dtype=torch.float32).t()
    del tempresult

    temp1 = torch.randint(0, d, (total_agent_num,))
    temp2 = torch.zeros(total_agent_num, d, dtype=torch.float32)
    temp3 = (torch.randint(0, 3, (total_agent_num,)) + 1) * 2 / 3
    temp2[torch.arange(total_agent_num), temp1] = 1.
    temp2 *= temp3.unsqueeze(1)
    x_all = temp2.clone()
    del temp1, temp2, temp3

    prob_all = torch.rand(total_agent_num) * 0.2 + 0.2

    locall = torch.randperm(x_all.shape[0])
    locini = locall[:M]
    locother = locall[M:]
    newagent_time = torch.randint(0, T, (len(locother),), dtype=torch.int64)
    print("avg time: ", tall1 / (loop + 1e-5), tall2 / (loop + 1e-5),
          tall3 / (loop + 1e-5), tall4 / (loop + 1e-5),
          tall5 / (loop + 1e-5), tall6 / (loop + 1e-5), flush=True)
    t1 = time.time()
    out_IUCB = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=True,
                                   Use_Individual_Rationality = True,truncate=False,
                                   get_c=c_simple,auxiliary_info_all=None,error_use=0.05,
                                   part_IC_matrix=partial(combine_pull_n,ntimes=10.5,dropother=False),
                                   program_name="IUCB")
    t2 = time.time()
    tall1 += (t2 - t1)
    t3 = time.time()
    out_IUCB_c = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=False,
                                   Use_Individual_Rationality=True,truncate=False,
                                   get_c=c_simple,auxiliary_info_all=None,error_use=0.05,
                                   part_IC_matrix=None, program_name="IUCB_c")
    t4 = time.time()
    tall2 += (t4 - t3)
    t5 = time.time()
    out_IUCB_r = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=True,
                                   Use_Individual_Rationality=False,truncate=False,
                                   get_c=c_simple,auxiliary_info_all=None,error_use=0.05,
                                   part_IC_matrix=partial(combine_pull_n,ntimes=10.5,dropother=False),
                                   program_name="IUCB_r")
    t6 = time.time()
    tall3 += (t6 - t5)
    t7 = time.time()
    out_IUCB_c_r = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=False,
                                   Use_Individual_Rationality=False,truncate=False,
                                   get_c=c_simple,auxiliary_info_all=None,error_use=0.05,
                                   part_IC_matrix=None, program_name="IUCB_c_r")
    t8 = time.time()
    tall4 += (t8 - t7)
    t7 = time.time()
    out_Ora_full = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=Oracle, Use_Incentive_Compatibility=False,
                                   Use_Individual_Rationality=False,truncate=True,
                                   get_c=c_simple,auxiliary_info_all=None,error_use=0.05,
                                   part_IC_matrix=None, program_name="Full Oracle")
    t8 = time.time()
    tall5 += (t8 - t7)
    t7 = time.time()
    out_Ora_part = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=Oracle, Use_Incentive_Compatibility=False,
                                   Use_Individual_Rationality=False,truncate=False,
                                   get_c=c_simple,auxiliary_info_all=None,error_use=0.05,
                                   part_IC_matrix=None, program_name="Partial Oracle")
    t8 = time.time()
    tall6 += (t8 - t7)

    outthis = [out_IUCB,out_IUCB_c,out_IUCB_r,out_IUCB_c_r,out_Ora_full,out_Ora_part]
    outall.append(outthis)

    with open('result/result'+str(simu_num)+'.pkl', 'wb') as f:
        pickle.dump(outall, f)
