from DA_algorithm import *
from get_bound import *
def get_true_info(X, true_beta, true_G, true_gamma, true_const):
    # true_G here is the G of coming agent
    meanreward = X @ true_beta + true_G
    trueoutside = (X * true_gamma).sum(dim=1) + true_const
    return meanreward, trueoutside
def UCB_noncovariate_procedure(X, true_beta, true_G, true_gamma, true_const,
                  trust_incre, trust_decre, trust_ini, trans_X,
                  get_v, get_oc_baseline, prob_comes, T, sigma,auxiliary_info=None,
                  error_use=0.05, Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,
                  get_c = c_simple,update_info=update_matrix, truncate = False,
                  device = "cpu", ini_data = None,
                  newagent_time = None, newagent_X = None, newagent_trustini = None,
                  newagent_G = None, newagent_prob = None,newagent_auxiliary_info=None,
                  quota1 = None, quota2 = None, propose = 1, largevalue = 1e5,
                  program_name = "",
                  print_process = False, get_error = torch.randn,
                  use_sparse = True, parallel_num = 4, part_IC_matrix=None,target=None,
                  get_true=get_true_info):

    # For non-covariate case

    # Initialize
    M = X.shape[0]
    d = 0
    N = true_G.shape[1]
    if len(sigma)==1:
        sigma = sigma.repeat(N)
    trust_now = trust_ini.clone()
    # build true and predict model
    rewardget = torch.empty(0)
    pull = torch.empty(0, dtype=bool)
    mean_reward = torch.zeros((M,N), dtype=torch.float32)
    band_reward = torch.randn((M,N), dtype=torch.float32)+largevalue
    tilde_b = torch.randn((M,N), dtype=torch.float32) - largevalue
    hat_b = torch.randn((M,N), dtype=torch.float32) + largevalue
    tilde_a = (mean_reward-band_reward-hat_b).max()
    hat_a = (mean_reward+band_reward-tilde_b).min()

    oc_baselineall = torch.empty(0)
    matched_times = torch.zeros((M, N), dtype=torch.int64)
    pulled_times = torch.zeros((M, N), dtype=torch.int64)

    trust_threshold = trust_decre.clone()

    re_trust = trust_ini.mean().unsqueeze(0)
    re_successpull = torch.tensor([0])
    re_rewardsum = torch.tensor([0])
    re_agent_survive = (trust_now > 0).sum().reshape(1)

    for timenow in range(T):
        if print_process:
            print(
                f'\r|{"█" * int(50 * (timenow + 1) / T)}{"-" * (50 - int(50 * (timenow + 1) / T))}| {(timenow + 1) / T * 100:.2f}%{program_name}',
                end='',flush=True)
        # Find which agents come
        agent_available = torch.where(trust_now > 0)[0]
        random_values = torch.rand(agent_available.size(0))
        agent_comes = agent_available[random_values < prob_comes[agent_available]]
        if len(agent_comes) > 0:
            if auxiliary_info is not None:
                oc_baseline_coming_agent = get_oc_baseline(X[agent_comes],auxiliary_info[agent_comes])
            else:
                oc_baseline_coming_agent = get_oc_baseline(X[agent_comes],None)
            true_v = get_v(oc_baseline_coming_agent, N)

            true_reward, true_backup = get_true(X=X[agent_comes], true_beta=true_beta,
                                                     true_G=true_G[agent_comes],
                                                     true_gamma=true_gamma, true_const=true_const)
            dif_reward_back = true_reward - true_backup.unsqueeze(1)

            upper_1 = mean_reward + band_reward
            upper_2 = hat_a + hat_b
            lower_1 = mean_reward - band_reward
            lower_2 = tilde_a + tilde_b
            if Use_Incentive_Compatibility:
                temp = upper_1 - upper_2
                upper_use = upper_1 - (temp > 0) * temp
                temp = lower_1 - lower_2
                lower_use = (temp > 0) * temp + lower_2
            else:
                upper_use = upper_1
                lower_use = lower_1
            mu_hat = upper_use.clone()
            if Use_Individual_Rationality:
                loclow = torch.where(trust_now <= trust_threshold)[0]
                mu_hat[loclow] = lower_use[loclow]
            mu_hat=mu_hat[agent_comes]
            agent_came_match_matrix = DA_algorithm(Value1=mu_hat, Value2=true_v,
                                                   quota1=quota1, quota2=quota2, propose=propose)

            # Update the true agent comes if an arm is actually recommend to the agent
            # Because if the agent number is too large, or some agent are picky, we may not recommend any arm.
            # NOTE!!!!!: if the quato for agent is greater than 1, this part should be revised!!!!
            (loc1, loc2) = torch.where(agent_came_match_matrix)
            agent_comes = agent_comes[loc1]
            agent_came_match_matrix = agent_came_match_matrix[loc1]
            true_reward = true_reward[loc1]
            true_backup = true_backup[loc1]
            oc_baseline_coming_agent = oc_baseline_coming_agent[loc1]

            (loc1, loc2) = torch.where(agent_came_match_matrix)
            success_or_not = (true_reward[loc1, loc2] > (true_backup[loc1] + oc_baseline_coming_agent[loc1]))
            agent_success = agent_comes[success_or_not]
            agent_failed = agent_comes[~success_or_not]
            # update trust
            trust_now[agent_success] += trust_incre
            trust_now[agent_failed] -= trust_decre
            # update data we get
            matched_times_old = matched_times.clone()
            matched_times[agent_comes] += (agent_came_match_matrix).int()


            nonnoise_reward = true_reward[loc1[success_or_not], loc2[success_or_not]]
            noised_reward = nonnoise_reward + get_error(loc2[success_or_not])
            rewardget = torch.cat((rewardget, noised_reward))

            temp = mean_reward[agent_comes[loc1[success_or_not]], loc2[success_or_not]]*\
                   pulled_times[agent_comes[loc1[success_or_not]], loc2[success_or_not]]
            mean_reward[loc1[success_or_not], loc2[success_or_not]] = (temp+noised_reward)/\
                        (pulled_times[agent_comes[loc1[success_or_not]], loc2[success_or_not]]+1)
            pulled_times[agent_comes[loc1[success_or_not]], loc2[success_or_not]] += 1
            band_reward = (6 * torch.log(matched_times+1)/pulled_times)**0.5*sigma.unsqueeze(0)
            band_reward[pulled_times==0] = torch.randn((pulled_times==0).sum(), dtype=torch.float32)+largevalue

            success_b = oc_baseline_coming_agent[loc1[success_or_not]]
            corres_b = tilde_b[agent_comes[loc1[success_or_not]], loc2[success_or_not]]
            temp = success_b-corres_b
            new_tilde_b = (temp>0)*temp+corres_b
            tilde_b[agent_comes[loc1[success_or_not]], loc2[success_or_not]] = new_tilde_b

            unsuccess_b = oc_baseline_coming_agent[loc1[~success_or_not]]
            corres_b = hat_b[agent_comes[loc1[~success_or_not]], loc2[~success_or_not]]
            temp = unsuccess_b-corres_b
            new_hat_b = unsuccess_b-(temp>0)*temp
            hat_b[agent_comes[loc1[~success_or_not]], loc2[~success_or_not]] = new_hat_b

            tilde_a = (mean_reward - band_reward - hat_b).max()
            hat_a = (mean_reward + band_reward - tilde_b).min()


            re_trust = torch.cat((re_trust, trust_now.mean().unsqueeze(0)))
            re_successpull = torch.cat((re_successpull, (success_or_not).sum().unsqueeze(0)))
            re_rewardsum = torch.cat((re_rewardsum, nonnoise_reward.sum().unsqueeze(0)))
            re_agent_survive = torch.cat((re_agent_survive, (trust_now > 0).sum().reshape(1)))

            oc_baselineall = torch.cat((oc_baselineall, oc_baseline_coming_agent))
            pull = torch.cat((pull,success_or_not))

        else:
            re_trust = torch.cat((re_trust, trust_now.mean().unsqueeze(0)))
            re_successpull = torch.cat((re_successpull, torch.tensor(0).unsqueeze(0)))
            re_rewardsum = torch.cat((re_rewardsum, torch.tensor(0).unsqueeze(0)))
            re_agent_survive = torch.cat((re_agent_survive, (trust_now > 0).sum().reshape(1)))

            X,auxiliary_info = trans_X(X,auxiliary_info, None, None)
            # print(program_name+"finish")
        new_agent_loc = torch.where(newagent_time == timenow)[0]
        if len(new_agent_loc) > 0:
            temp = torch.zeros(len(new_agent_loc), N, dtype=torch.int64)
            matched_times = torch.cat((matched_times, temp), 0)
            pulled_times = torch.cat((pulled_times, temp), 0)
            X = torch.cat((X, newagent_X[new_agent_loc]), 0)
            if auxiliary_info is not None:
                auxiliary_info = torch.cat((auxiliary_info,newagent_auxiliary_info))
            trust_now = torch.cat((trust_now, newagent_trustini[new_agent_loc]))
            prob_comes = torch.cat((prob_comes, newagent_prob[new_agent_loc]))

            true_G = torch.cat((true_G, newagent_G[new_agent_loc]), 0)

            mean_reward = torch.cat((mean_reward,torch.zeros((len(new_agent_loc), N), dtype=torch.float32)),0)
            band_reward = torch.cat((band_reward,torch.randn((len(new_agent_loc), N), dtype=torch.float32) + largevalue),0)
            tilde_b = torch.cat((tilde_b,torch.randn((len(new_agent_loc), N), dtype=torch.float32) - largevalue),0)
            hat_b = torch.cat((hat_b, torch.randn((len(new_agent_loc), N), dtype=torch.float32) + largevalue), 0)
            tilde_a = (mean_reward - band_reward - hat_b).max()
            hat_a = (mean_reward + band_reward - tilde_b).min()
            M += len(new_agent_loc)

    if print_process:
        print('\n')
    out = [re_trust, re_successpull, re_rewardsum, re_agent_survive]
    return out


def IUCB_noncovariate_procedure(X, true_beta, true_G, true_gamma, true_const,
                  trust_incre, trust_decre, trust_ini, trans_X,
                  get_v, get_oc_baseline, prob_comes, T, sigma,auxiliary_info=None,
                  error_use=0.05, Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,
                  get_c = c_simple,update_info=update_matrix, truncate = False,
                  device = "cpu", ini_data = None,
                  newagent_time = None, newagent_X = None, newagent_trustini = None,
                  newagent_G = None, newagent_prob = None,newagent_auxiliary_info=None,
                  quota1 = None, quota2 = None, propose = 1, largevalue = 1e5,
                  program_name = "",
                  print_process = False, get_error = torch.randn,
                  use_sparse = True, parallel_num = 4,part_IC_matrix=None,target=None,
                  get_true=get_true_info):

    # For non-covariate case

    # Initialize
    M = X.shape[0]
    d = 0
    N = true_G.shape[1]
    if len(sigma)==1:
        sigma = sigma.repeat(N)
    trust_now = trust_ini.clone()
    # build true and predict model
    rewardget = torch.empty(0)
    pull = torch.empty(0, dtype=bool)
    sumlambda = torch.zeros((M,N), dtype=torch.float32)
    sumlambda2 = torch.zeros((M, N), dtype=torch.float32)
    mean_reward = torch.zeros((M,N), dtype=torch.float32)
    band_reward = torch.randn((M,N), dtype=torch.float32)+largevalue
    tilde_b = torch.randn((M,N), dtype=torch.float32) - largevalue
    hat_b = torch.randn((M,N), dtype=torch.float32) + largevalue
    tilde_a = (mean_reward-band_reward-hat_b).max()
    hat_a = (mean_reward+band_reward-tilde_b).min()

    oc_baselineall = torch.empty(0)
    matched_times = torch.zeros((M, N), dtype=torch.int64)
    pulled_times = torch.zeros((M, N), dtype=torch.int64)

    trust_threshold = trust_decre.clone()

    re_trust = trust_ini.mean().unsqueeze(0)
    re_successpull = torch.tensor([0])
    re_rewardsum = torch.tensor([0])
    re_agent_survive = (trust_now > 0).sum().reshape(1)

    for timenow in range(T):
        fix_para = torch.log(2 * (M * N) / torch.tensor([error_use]))
        if print_process:
            print(
                f'\r|{"█" * int(50 * (timenow + 1) / T)}{"-" * (50 - int(50 * (timenow + 1) / T))}| {(timenow + 1) / T * 100:.2f}%{program_name}',
                end='',flush=True)
        # Find which agents come
        agent_available = torch.where(trust_now > 0)[0]
        random_values = torch.rand(agent_available.size(0))
        agent_comes = agent_available[random_values < prob_comes[agent_available]]
        if len(agent_comes) > 0:
            if auxiliary_info is not None:
                oc_baseline_coming_agent = get_oc_baseline(X[agent_comes],auxiliary_info[agent_comes])
            else:
                oc_baseline_coming_agent = get_oc_baseline(X[agent_comes],None)
            true_v = get_v(oc_baseline_coming_agent, N)

            true_reward, true_backup = get_true(X=X[agent_comes], true_beta=true_beta,
                                                     true_G=true_G[agent_comes],
                                                     true_gamma=true_gamma, true_const=true_const)
            dif_reward_back = true_reward - true_backup.unsqueeze(1)

            upper_1 = mean_reward + band_reward
            upper_2 = hat_a + hat_b
            lower_1 = mean_reward - band_reward
            lower_2 = tilde_a + tilde_b
            if Use_Incentive_Compatibility:
                temp = upper_1 - upper_2
                upper_use = upper_1 - (temp > 0) * temp
                temp = lower_1 - lower_2
                lower_use = (temp > 0) * temp + lower_2
            else:
                upper_use = upper_1
                lower_use = lower_1
            mu_hat = upper_use.clone()
            if Use_Individual_Rationality:
                loclow = torch.where(trust_now <= trust_threshold)[0]
                mu_hat[loclow] = lower_use[loclow]
            mu_hat=mu_hat[agent_comes]
            agent_came_match_matrix = DA_algorithm(Value1=mu_hat, Value2=true_v,
                                                   quota1=quota1, quota2=quota2, propose=propose)

            # Update the true agent comes if an arm is actually recommend to the agent
            # Because if the agent number is too large, or some agent are picky, we may not recommend any arm.
            # NOTE!!!!!: if the quato for agent is greater than 1, this part should be revised!!!!
            (loc1, loc2) = torch.where(agent_came_match_matrix)
            agent_comes = agent_comes[loc1]
            agent_came_match_matrix = agent_came_match_matrix[loc1]
            true_reward = true_reward[loc1]
            true_backup = true_backup[loc1]
            oc_baseline_coming_agent = oc_baseline_coming_agent[loc1]

            (loc1, loc2) = torch.where(agent_came_match_matrix)
            success_or_not = (true_reward[loc1, loc2] > (true_backup[loc1] + oc_baseline_coming_agent[loc1]))
            agent_success = agent_comes[success_or_not]
            agent_failed = agent_comes[~success_or_not]
            # update trust
            trust_now[agent_success] += trust_incre
            trust_now[agent_failed] -= trust_decre
            # update data we get
            matched_times_old = matched_times.clone()
            matched_times[agent_comes] += (agent_came_match_matrix).int()


            nonnoise_reward = true_reward[loc1[success_or_not], loc2[success_or_not]]
            noised_reward = nonnoise_reward + get_error(loc2[success_or_not])
            rewardget = torch.cat((rewardget, noised_reward))

            pulled_times[agent_comes[loc1[success_or_not]], loc2[success_or_not]] += 1
            temppulltime = pulled_times[agent_comes[loc1[success_or_not]], loc2[success_or_not]]
            temp = mean_reward[agent_comes[loc1[success_or_not]], loc2[success_or_not]]*\
                   sumlambda[agent_comes[loc1[success_or_not]], loc2[success_or_not]]

            newlambda=(2*fix_para/temppulltime/torch.log(temppulltime+1))**0.5/sigma[loc2[success_or_not]]

            mean_reward[loc1[success_or_not], loc2[success_or_not]] = (temp+noised_reward*newlambda)/\
                            (sumlambda[agent_comes[loc1[success_or_not]], loc2[success_or_not]]+newlambda)
            sumlambda[agent_comes[loc1[success_or_not]], loc2[success_or_not]] += newlambda
            sumlambda2[agent_comes[loc1[success_or_not]], loc2[success_or_not]]+= newlambda**2

            band_reward = (sigma.unsqueeze(0)**2/2*sumlambda2+fix_para)/sumlambda
            band_reward[pulled_times==0] = torch.randn((pulled_times==0).sum(), dtype=torch.float32)+largevalue

            success_b = oc_baseline_coming_agent[loc1[success_or_not]]
            corres_b = tilde_b[agent_comes[loc1[success_or_not]], loc2[success_or_not]]
            temp = success_b-corres_b
            new_tilde_b = (temp>0)*temp+corres_b
            tilde_b[agent_comes[loc1[success_or_not]], loc2[success_or_not]] = new_tilde_b

            unsuccess_b = oc_baseline_coming_agent[loc1[~success_or_not]]
            corres_b = hat_b[agent_comes[loc1[~success_or_not]], loc2[~success_or_not]]
            temp = unsuccess_b-corres_b
            new_hat_b = unsuccess_b-(temp>0)*temp
            hat_b[agent_comes[loc1[~success_or_not]], loc2[~success_or_not]] = new_hat_b

            tilde_a = (mean_reward - band_reward - hat_b).max()
            hat_a = (mean_reward + band_reward - tilde_b).min()


            re_trust = torch.cat((re_trust, trust_now.mean().unsqueeze(0)))
            re_successpull = torch.cat((re_successpull, (success_or_not).sum().unsqueeze(0)))
            re_rewardsum = torch.cat((re_rewardsum, nonnoise_reward.sum().unsqueeze(0)))
            re_agent_survive = torch.cat((re_agent_survive, (trust_now > 0).sum().reshape(1)))

            oc_baselineall = torch.cat((oc_baselineall, oc_baseline_coming_agent))
            pull = torch.cat((pull,success_or_not))

        else:
            re_trust = torch.cat((re_trust, trust_now.mean().unsqueeze(0)))
            re_successpull = torch.cat((re_successpull, torch.tensor(0).unsqueeze(0)))
            re_rewardsum = torch.cat((re_rewardsum, torch.tensor(0).unsqueeze(0)))
            re_agent_survive = torch.cat((re_agent_survive, (trust_now > 0).sum().reshape(1)))

            X,auxiliary_info = trans_X(X,auxiliary_info, None, None)
            # print(program_name+"finish")
        new_agent_loc = torch.where(newagent_time == timenow)[0]
        if len(new_agent_loc) > 0:
            temp = torch.zeros(len(new_agent_loc), N, dtype=torch.int64)
            matched_times = torch.cat((matched_times, temp), 0)
            pulled_times = torch.cat((pulled_times, temp), 0)
            X = torch.cat((X, newagent_X[new_agent_loc]), 0)
            if auxiliary_info is not None:
                auxiliary_info = torch.cat((auxiliary_info,newagent_auxiliary_info))
            trust_now = torch.cat((trust_now, newagent_trustini[new_agent_loc]))
            prob_comes = torch.cat((prob_comes, newagent_prob[new_agent_loc]))

            true_G = torch.cat((true_G, newagent_G[new_agent_loc]), 0)

            mean_reward = torch.cat((mean_reward,torch.zeros((len(new_agent_loc), N), dtype=torch.float32)),0)
            band_reward = torch.cat((band_reward,torch.randn((len(new_agent_loc), N), dtype=torch.float32) + largevalue),0)
            sumlambda = torch.cat((sumlambda,torch.zeros((len(new_agent_loc), N), dtype=torch.float32)),0)
            sumlambda2 = torch.cat((sumlambda2, torch.zeros((len(new_agent_loc), N), dtype=torch.float32)), 0)

            tilde_b = torch.cat((tilde_b,torch.randn((len(new_agent_loc), N), dtype=torch.float32) - largevalue),0)
            hat_b = torch.cat((hat_b, torch.randn((len(new_agent_loc), N), dtype=torch.float32) + largevalue), 0)
            tilde_a = (mean_reward - band_reward - hat_b).max()
            hat_a = (mean_reward + band_reward - tilde_b).min()
            M += len(new_agent_loc)

    if print_process:
        print('\n')
    out = [re_trust, re_successpull, re_rewardsum, re_agent_survive]
    return out