from DA_algorithm import *
from get_bound import *
def IUCB(trust, trust_threshold, true_dif_reward_back, N, d, M, xx_bar, xy_bar, band_part1,
        coming_agent, agent_x, oc_baseline_coming_agent,error_rate=0.05,
        IC_matrix=None, pull_or_not=None, oc_baseline=None,Use_Individual_Rationality=True,
        use_sparse=True, parallel_num=4, largevalue=1e5,
        truncate=False):
    if Use_Individual_Rationality:
        need_lower = torch.zeros(len(coming_agent), dtype=bool)
        low_trust_low = torch.where(trust <= trust_threshold)[0]
        need_lower[low_trust_low] = True
    else:
        need_lower = None

    usex = get_predict_bound(N=N, d=d, M=M, xx_bar=xx_bar,xy_bar=xy_bar,band_part1=band_part1,
                             coming_agent=coming_agent, agent_x=agent_x,
                             error_rate = error_rate,need_lower=need_lower,
                             IC_matrix=IC_matrix, pull_or_not=pull_or_not, oc_baseline=oc_baseline,
                             use_sparse=use_sparse, parallel_num=parallel_num, largevalue=largevalue)
    usex -= oc_baseline_coming_agent.unsqueeze(1)
    usex[usex > (largevalue / 2)] = largevalue+torch.randn((usex > (largevalue / 2)).sum())
    usex[usex < (-largevalue / 2)] = -largevalue + torch.randn((usex < (-largevalue / 2)).sum())
    return usex

def Oracle(trust, trust_threshold, true_dif_reward_back, N, d, M, xx_bar, xy_bar, band_part1,
        coming_agent, agent_x, oc_baseline_coming_agent,error_rate=0.05,
        IC_matrix=None, pull_or_not=None, oc_baseline=None,Use_Individual_Rationality=True,
        use_sparse=True, parallel_num=4, largevalue=1e5,
        truncate=False):
    usex = true_dif_reward_back.clone()
    usex -= oc_baseline_coming_agent.unsqueeze(1)
    if truncate:
        usex[usex < 0] = torch.tensor(-float('inf'))
    return usex


def get_true_info(X, true_beta, true_G, true_gamma, true_const):
    # true_G here is the G of coming agent
    meanreward = X @ true_beta + true_G
    trueoutside = (X * true_gamma).sum(dim=1) + true_const
    return meanreward, trueoutside

def matching_procedure(X, true_beta, true_G, true_gamma, true_const,
                  trust_incre, trust_decre, trust_ini, trans_X,
                  get_v, get_oc_baseline, prob_comes, T, method_use, sigma,auxiliary_info=None,
                  error_use=0.05, Use_Incentive_Compatibility=True,Use_Individual_Rationality=True,
                  get_c = c_simple,update_info=update_matrix, truncate = False,
                  device = "cpu", ini_data = None,
                  newagent_time = None, newagent_X = None, newagent_trustini = None,
                  newagent_G = None, newagent_prob = None,newagent_auxiliary_info=None,
                  quota1 = None, quota2 = None, propose = 1, largevalue = 1e5,
                  program_name = "",
                  print_process = False, get_error = torch.randn,
                  use_sparse = True, parallel_num = 4, epsilon_explo = 0.1,part_IC_matrix=None,target=None,
                  get_true=get_true_info):

    # X M*d, covariates of current agents
    # true_beta d*N, column i correspond to arm i
    # true_G M*N, (j,i) correspond to agent j's preference on arm i.
    # true_gamma is of shape d
    # true_const is of shape 1
    # trans_X, how covariates evolves each time after pulling
    # get_v, get the arms' preferences
    # get_oc_baseline, get the oc_baseline of agent
    # method use, UCB ICUB and Oracle
    # newagent...., information of newly coming agents

    # Initialize
    M = X.shape[0]
    d = X.shape[1]
    N = true_G.shape[1]
    if len(sigma)==1:
        sigma = sigma.repeat(N)
    trust_now = trust_ini.clone()
    # build true and predict model
    rewardget = torch.empty(0)
    pull = torch.empty(0, dtype=bool)
    xx_bar = torch.zeros((d * N + M * N, d * N + M * N), dtype=torch.float32)
    xy_bar = torch.zeros(d * N + M * N, dtype=torch.float32)
    band_part1 = torch.zeros(d * N + M * N, dtype=torch.float32)
    if Use_Incentive_Compatibility:
        IC_matrix = torch.empty((0, d * N + M * N + d + 1), dtype=torch.float32)
    else:
        IC_matrix = None
    oc_baselineall = torch.empty(0)
    matched_times = torch.zeros((M, N), dtype=torch.int64)
    pulled_times = torch.zeros((M, N), dtype=torch.int64)

    trust_threshold = trust_decre

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

            muhat = method_use(trust=trust_now[agent_comes], trust_threshold=trust_threshold,
                               true_dif_reward_back=dif_reward_back, N=N, d=d, M=M,
                               xx_bar=xx_bar, xy_bar=xy_bar, band_part1=band_part1,
                               coming_agent=agent_comes, agent_x=X[agent_comes],
                               oc_baseline_coming_agent=oc_baseline_coming_agent,error_rate=error_use,
                               IC_matrix=IC_matrix, pull_or_not=pull, oc_baseline=oc_baselineall,
                               Use_Individual_Rationality=Use_Individual_Rationality,
                              use_sparse=use_sparse,parallel_num=parallel_num, largevalue=largevalue,
                               truncate=truncate)
            agent_came_match_matrix = DA_algorithm(Value1=muhat, Value2=true_v,
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


            newxxbar, newxybar,newband, newIC_matrix = update_info(N=N, d=d, M=M,X_all=X[agent_comes[loc1]],
                                                           agent_number=agent_comes[loc1], recommend_arm=loc2,
                                                           pull_times=pulled_times,reward=noised_reward,
                                                           pull_or_not=success_or_not,sigma=sigma,error_rate=error_use,
                                                           Use_Incentive_Compatibility=(IC_matrix != None),
                                                            get_c=get_c,target=target)

            xx_bar +=newxxbar
            xy_bar +=newxybar
            band_part1 += newband
            pulled_times[agent_comes[loc1[success_or_not]], loc2[success_or_not]] += 1
            if pulled_times.sum()<0.5:
                xx_bar = torch.zeros_like(xx_bar)
                xy_bar = torch.zeros_like(xy_bar)
                band_part1 = torch.zeros_like(band_part1)
            X , auxiliary_info= trans_X(X, auxiliary_info,agent_comes[success_or_not], loc2[success_or_not])


            re_trust = torch.cat((re_trust, trust_now.mean().unsqueeze(0)))
            re_successpull = torch.cat((re_successpull, (success_or_not).sum().unsqueeze(0)))
            re_rewardsum = torch.cat((re_rewardsum, nonnoise_reward.sum().unsqueeze(0)))
            re_agent_survive = torch.cat((re_agent_survive, (trust_now > 0).sum().reshape(1)))

            if IC_matrix is not None:
                if part_IC_matrix is not None:
                    newIC_matrix,success_or_not,oc_baseline_coming_agent = part_IC_matrix (
                        matched_times_old[agent_comes],newIC_matrix,success_or_not,oc_baseline_coming_agent)
                IC_matrix = torch.cat((IC_matrix, newIC_matrix), 0)

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

            xx_bar = torch.cat((xx_bar, torch.zeros(d * N + M * N, len(new_agent_loc) * N)), 1)
            xx_bar = torch.cat((xx_bar, torch.zeros(len(new_agent_loc) * N, d * N + M * N + len(new_agent_loc) * N)), 0)
            xy_bar = torch.cat((xy_bar, torch.zeros(len(new_agent_loc) * N)))
            band_part1 = torch.cat((band_part1, torch.zeros(len(new_agent_loc) * N)))
            if IC_matrix is not None:
                IC_matrix = torch.cat((IC_matrix[:, :-(d + 1)],
                                         torch.zeros(IC_matrix.shape[0], len(new_agent_loc) * N),
                                         IC_matrix[:, -(d + 1):]), dim=1)

            M += len(new_agent_loc)

    if print_process:
        print('\n')
    out = [re_trust, re_successpull, re_rewardsum, re_agent_survive]
    return out


