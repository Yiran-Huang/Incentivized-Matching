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

looptimes = 10
parallel_num = 6
total_agent_num = 150
d = 5
N = 6
M = 80
T = 120

if not os.path.exists("result"):
    os.makedirs("result")

year_range = "0110"
with open('../processed_data/Journal'+year_range+'.pkl', 'rb') as f:
    datause = pickle.load(f)

[true_beta,true_G,epsall,pub_num_all,
cit_num_all,journal_score,group_indices]=datause
true_G += 1e-5
true_gamma = true_beta[:,-2]
true_const = torch.tensor([0.])
sigma_all = torch.empty((0))
for i in range(len(epsall)):
    eps = epsall[i]
    q25 = eps.quantile(0.25)
    q75 = eps.quantile(0.75)
    loc = torch.where((eps>=(q25-1.5*(q75-q25))) & (eps<=(q75+1.5*(q75-q25))))[0]
    eps=eps[loc]
    eps-=eps.mean()
    epsall[i]=eps
    sigmathis = (eps.max()-eps.min())/2
    sigma_all = torch.cat((sigma_all,sigmathis.unsqueeze(0)))

group_journal = torch.zeros(len(cit_num_all),dtype=torch.int64)
for i in range(len(group_indices)):
    group_journal[group_indices[i]]=i

pub_num_all = torch.zeros(len(group_indices)).index_add(0, group_journal, torch.tensor(pub_num_all,dtype=torch.float32))
cit_num_all = torch.zeros(len(group_indices)).index_add(0, group_journal, torch.tensor(cit_num_all,dtype=torch.float32))
journal_score = cit_num_all/pub_num_all


def get_compare(T, d, N, M, true_beta, true_gamma, true_const, x_all, G_all, prob_all, locini, locother, newagent_time,
                method_use, Use_Incentive_Compatibility,Use_Individual_Rationality, truncate, get_c,auxiliary_info_all,
                error_use=0.05,part_IC_matrix=None, program_name=""):
    X = x_all[locini].clone()
    true_G = G_all[locini].clone()
    if auxiliary_info_all is not None:
        auxiliary_info = auxiliary_info_all[locini].clone()

    trust_incre = torch.tensor(1.)
    trust_decre = torch.tensor(5.)
    trust_ini_value = 10.1
    trust_ini = torch.ones(M) * trust_ini_value

    def trans_X(X,auxiliary_info, agent_comes, pullarm):
        if agent_comes is not None:
            xtemp = (X[agent_comes]).clone()
            xtemp = torch.cat((xtemp,(1-xtemp.sum(dim=1)).unsqueeze(1)),1)
            pubnum_temp = (auxiliary_info[agent_comes]).clone()
            xtemp[pubnum_temp<0.5]=0
            xtemp = xtemp * pubnum_temp.unsqueeze(1)
            xtemp[torch.arange(xtemp.shape[0]),pullarm] +=1
            xtemp /= (xtemp.sum(dim=1).unsqueeze(1))
            pubnum_temp+=1
            xtemp = xtemp[:,:-1]
            X[agent_comes] = xtemp
            auxiliary_info[agent_comes] = pubnum_temp
        return X,auxiliary_info

    def get_v(oc_baseline, N):
        num_agent = len(oc_baseline)
        return oc_baseline.unsqueeze(0).expand(N, num_agent) + torch.randn(N, num_agent) * 1

    def get_oc_baseline(x,pubnum):
        xtemp = x.clone()
        xtemp = torch.cat((xtemp, (1 - xtemp.sum(dim=1)).unsqueeze(1)), 1)
        xtemp[pubnum < 0.5] = 0
        xtemp = xtemp * pubnum.unsqueeze(1)
        oc_baseline_max = 0.5 * torch.log((xtemp  @ journal_score.unsqueeze(1)).view(-1) + 1)
        return (2*torch.rand(x.shape[0])-1)*oc_baseline_max

    prob_comes = prob_all[locini].clone()
    sigma = sigma_all.clone()
    update_info = update_matrix
    device = "cpu"
    ini_data = None

    newagent_X = x_all[locother].clone()
    newagent_trustini = torch.ones(len(newagent_time)) * trust_ini_value
    newagent_G = G_all[locother].clone()
    newagent_prob = prob_all[locother].clone()
    newagent_auxiliary_info = auxiliary_info_all[locother].clone()

    quota1 = None
    quota2 = torch.tensor(np.array((pub_num_all/15/365*30/4).numpy().round(0),dtype=np.int64),dtype=torch.int64)
    propose = 1
    largevalue = 1e8
    print_process = True
    def get_error(pulled_arm):
        result = torch.tensor([
            np.random.choice(epsall[i]) for i in pulled_arm.tolist()
        ], dtype=torch.float32)
        return result
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

    temp = torch.zeros((total_agent_num,N),dtype=torch.float32)
    auxiliary_info_all = temp.sum(dim=1)
    temp /= (auxiliary_info_all.unsqueeze(1))
    temp[auxiliary_info_all<0.5]=0
    x_all = temp[:,:-1].clone()
    del temp

    G_all = true_G[np.random.choice(range(true_G.shape[0]), total_agent_num, replace=True)].clone()

    prob_all = torch.rand(total_agent_num) * 0.1+0.25

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
                                   get_c=c_simple,auxiliary_info_all=auxiliary_info_all,error_use=0.05,
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
                                   get_c=c_simple,auxiliary_info_all=auxiliary_info_all,error_use=0.05,
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
                                   get_c=c_simple,auxiliary_info_all=auxiliary_info_all,error_use=0.05,
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
                                   get_c=c_simple,auxiliary_info_all=auxiliary_info_all,error_use=0.05,
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
                                   get_c=c_simple,auxiliary_info_all=auxiliary_info_all,error_use=0.05,
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
                                   get_c=c_simple,auxiliary_info_all=auxiliary_info_all,error_use=0.05,
                                   part_IC_matrix=None, program_name="Partial Oracle")
    t8 = time.time()
    tall6 += (t8 - t7)


    outthis = [out_IUCB,out_IUCB_c,out_IUCB_r,out_IUCB_c_r,out_Ora_full,out_Ora_part]
    outall.append(outthis)

    with open('result/result'+str(simu_num)+'.pkl', 'wb') as f:
        pickle.dump(outall, f)

