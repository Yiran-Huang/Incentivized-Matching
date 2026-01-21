from DA_algorithm import *
from get_bound import *
from Matching_Method import *
import numpy as np
import os
import pickle
simu_num = int(os.path.basename(__file__)[4])

np.random.seed(1120220054 + simu_num)
torch.manual_seed(1120220054 + simu_num)

looptimes = 5
parallel_num = 4
total_agent_num = 150
d = 6
N = 6
M = 100
T = 150

if not os.path.exists("result"):
    os.makedirs("result")

def get_compare(T, d, N, M, true_beta, true_gamma, true_const, x_all, G_all, prob_all, locini, locother, newagent_time,
                method_use, Use_Incentive_Compatibility,Use_Individual_Rationality, truncate, get_c,
                error_use=0.05,part_IC_matrix=None, program_name=""):
    X = x_all[locini].clone()
    true_G = G_all[locini].clone()

    trust_incre = torch.tensor(1.)
    trust_decre = torch.tensor(2.)
    trust_ini_value = 4.1
    trust_ini = torch.ones(M) * trust_ini_value

    def trans_X(X,auxiliary_info, agent_comes, pullarm):
        if agent_comes is not None:
            X[agent_comes] = torch.rand(len(agent_comes), X.shape[1]) * 2
        return X,None

    def get_v(oc_baseline, N):
        num_agent = len(oc_baseline)
        return oc_baseline.unsqueeze(0).expand(N, num_agent) + torch.randn(N, num_agent) * 0.1

    def get_oc_baseline(x,y):
        return torch.randn(x.shape[0])

    prob_comes = prob_all[locini].clone()
    sigma = torch.tensor([1.])
    update_info = update_matrix
    device = "cpu"
    ini_data = None

    newagent_X = x_all[locother].clone()
    newagent_trustini = torch.ones(len(newagent_time)) * trust_ini_value
    newagent_G = G_all[locother].clone()
    newagent_prob = prob_all[locother].clone()

    quota1 = None
    quota2 = torch.tensor([5,5,5,5,5,5], dtype=torch.int64)
    propose = 1
    largevalue = 1e5
    print_process = True
    def get_error(x):
        return torch.randn(len(x))
    use_sparse = True

    out = matching_procedure(X=X, true_beta=true_beta, true_G=true_G, true_gamma=true_gamma, true_const=true_const,
                        trust_incre=trust_incre, trust_decre=trust_decre, trust_ini=trust_ini, trans_X=trans_X,
                        get_v=get_v, get_oc_baseline=get_oc_baseline, prob_comes=prob_comes, T=T, method_use=method_use,
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
    return out


outall = []
tall1 = 0.
tall2 = 0.
tall3 = 0.
tall4 = 0.
tall5 = 0.
tall6 = 0.
for loop in range(looptimes):
    print(loop + 1, "/", looptimes,flush=True)

    true_beta = torch.ones(d, N, dtype=torch.float32)
    true_gamma = torch.ones(d, dtype=torch.float32)
    true_const = torch.tensor([-0.5])

    x_all = torch.rand(total_agent_num, d) * 2
    G_all = torch.randn(total_agent_num, N)
    prob_all = torch.rand(total_agent_num) * (0.15 - 0.05) + 0.05 + 0.1

    locall = torch.randperm(x_all.shape[0])
    locini = locall[:M]
    locother = locall[M:]
    newagent_time = torch.randint(0, T, (len(locother),), dtype=torch.int64)
    print("avg time: ", tall1 / (loop + 1e-5), tall2 / (loop + 1e-5),
          tall3 / (loop + 1e-5),tall4 / (loop + 1e-5),
         tall5 / (loop + 1e-5),tall6 / (loop + 1e-5),flush=True)
    t1 = time.time()
    out_IUCB = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=True,
                                   Use_Individual_Rationality = True,truncate=False,
                                   get_c=c_simple,error_use=0.05,
                                   part_IC_matrix=None, program_name="IUCB")
    t2 = time.time()
    tall1 += (t2 - t1)
    t3 = time.time()
    out_IUCB_c = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=False,
                                   Use_Individual_Rationality=True,truncate=False,
                                   get_c=c_simple,error_use=0.05,
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
                                   get_c=c_simple,error_use=0.05,
                                   part_IC_matrix=None, program_name="IUCB_r")
    t6 = time.time()
    tall3 += (t6 - t5)
    t7 = time.time()
    out_IUCB_c_r = get_compare(T=T,d=d, N=N, M=M, true_beta=true_beta, true_gamma=true_gamma,
                                   true_const=true_const, x_all=x_all, G_all=G_all, prob_all=prob_all,
                                   locini=locini,
                                   locother=locother, newagent_time=newagent_time,
                                   method_use=IUCB, Use_Incentive_Compatibility=False,
                                   Use_Individual_Rationality=False,truncate=False,
                                   get_c=c_simple,error_use=0.05,
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
                                   get_c=c_simple,error_use=0.05,
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
                                   get_c=c_simple,error_use=0.05,
                                   part_IC_matrix=None, program_name="Partial Oracle")
    t8 = time.time()
    tall6 += (t8 - t7)



    outthis = [out_IUCB,out_IUCB_c,out_IUCB_r,out_IUCB_c_r,out_Ora_full,out_Ora_part]
    outall.append(outthis)

    with open('result/result'+str(simu_num)+'.pkl', 'wb') as f:
        pickle.dump(outall, f)
