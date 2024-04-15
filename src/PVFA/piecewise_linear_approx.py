"""
File: piecewise_linear_approx.py
Author: Julius Luy
Date: November 10th 2023
Description: This module implements the piecewise-linear value function approximation (Algorithm 2 of the paper).

Parameters
----------
problem: object
    Class that implements the MDP's environment.
method : object
    Class that contains all hyperparameter settings.
base_configuration : object
    Contains all configuration parameters.
costs_to_go_calculation : string
    Indicates whether operational costs should always be calculated or if cached values can be used.

Returns
----------
v : Dict
    Contains the learned slope vectors for all encountered post decision states
total_costs_hist_pd_av: List
    Contains the total cost history over the number of training iterations.
"""

import time
import numpy as np
import os,sys
import pandas as pd
import copy
from common.state_transfer_fct import state_transfer_fct
from common._brent import brent
import matplotlib.pyplot as plt

from common.state_transfer_fct import state_transfer_fct
from PVFA.setup_fluid_model import setup_fluid_model
from pathlib import Path
current_working_dir = Path(os.getcwd())
main_project_dir = str(current_working_dir.parent) # .parent
sys.path.append(main_project_dir+'/cpp/cmake-build-release/lib')

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

class PLVFA:

    def __init__(self,problem,method,base_configuration,costs_to_go_calculation):
        # Initialize parameters
        self.problem = problem
        self.method = method
        self.base_configuration = base_configuration
        self.costs_to_go_calculation = costs_to_go_calculation
        if base_configuration.mode == 'train' and base_configuration.load_existing_slopes == 'yes':
            self.v = np.load(base_configuration.results_path / 'v.npy', allow_pickle=True).tolist()

        elif base_configuration.mode == 'train':
            self.v = {}
        else:
            self.v = np.load(base_configuration.results_path / 'v.npy',allow_pickle=True).tolist()
            self.n = method.max_epochs/2
        self.max_epochs = method.max_epochs
        self.running_average_interval = 20  # Smoothing of reward curve
        self.n_check_policy = base_configuration.n_check_policy  # Check policy every x steps
        self.n_convergence_check = base_configuration.n_convergence_check  # Check convergence form 5k iterations onwards
        self.n_memory_limit = 2
        self.act = self.problem.action_space
        self.FD_stepsize =  np.ones(len(self.act)-1)*method.fd_discretization
        self.alpha = self.method.alpha
        self.gamma = method.gamma
        self.fd_discretization = method.fd_discretization
        if base_configuration.include_firing_decisions == 'yes':
            self.include_firing_decisions = 1
        else:
            self.include_firing_decisions = 0
        self.shape_slope_vector = np.zeros(self.FD_stepsize.shape)

        self.N_testing = method.N_testing
        if self.max_epochs < self.N_testing:
            self.N_testing = int(0.1*self.max_epochs)

        # plvfa or greedy; episode; result specific; time step
        self.drivers = np.zeros((2,self.N_testing,3,self.problem.tmax+1)) # fd,gw,od
        self.costs = np.zeros((2,self.N_testing,9, self.problem.tmax + 1)) # total,operational, fdfix,fdvar,emptyrout,gw,od,pen,severance
        self.cds_non_matched = np.zeros((2,self.N_testing,2, self.problem.tmax + 1)) #gw,od
        self.request_delivered = np.zeros((2,self.N_testing,4, self.problem.tmax + 1)) # fd,gw,od,not
        self.hiring_decisions = np.zeros((2,self.N_testing,self.problem.tmax+1))

        # Initialize tracking buffers
        self.total_costs_hist = []
        self.costs_to_go_buffer = {}
        self.sCD_buffer = {}

        # Initialize fluid model
        self.fluid_model = setup_fluid_model(problem)
        test = 0

        # self.v_hist = []

    # Main loop (cf. Algorithm 2)
    def piecewise_linear_approx(self):
        t1 = time.time()
        for n in range(0, self.max_epochs):
                            self.n = n
                            t_tmp = time.time()
                            total_utility_n = self.forward_backward_pass()
                            print('episode: '+str(n)+' / time required for one episode: '+str(time.time()-t_tmp))
                            self.total_costs_hist.append(total_utility_n)
                            if np.mod(n,self.n_check_policy) == 0 and n>0:
                                delta_t = time.time()-t1
                                print('no. of iterations = '+str(n)+', time needed for '+str(self.n_check_policy)+' iterations: '+str(delta_t)+'s')
                                # plot intermediate results
                                fig = plt.figure()
                                plt.plot(self.total_costs_hist)
                                plt.ylabel('Total costs in one episode')
                                plt.xlabel('Number of episodes')
                                fig.set_size_inches(8, 6)
                                plt.savefig(self.base_configuration.results_path / 'total_cost_hist.pdf',bbox_inches='tight', dpi=500)
                                plt.close()

                                # Ensuring memory limits
                                if sys.getsizeof(self.costs_to_go_buffer)>10**8:
                                    self.costs_to_go_buffer.clear()
                                if sys.getsizeof(self.sCD_buffer)>10**7:
                                    self.sCD_buffer.clear()
                                if sys.getsizeof(self.v)>10**7:
                                    while len({key: val for (key, val) in self.method.v_counter.items() if val < self.n_memory_limit}) == 0:
                                        self.n_memory_limit += 1
                                        if self.n_memory_limit>50:
                                            break
                                    if self.n_memory_limit > 50:
                                        print('memory limit of v reached')
                                        break
                                    for (key, val) in self.method.v_counter.items():
                                        if val < self.n_memory_limit:
                                            self.v = removekey(self.v, key)
                                            self.method.v_counter = removekey(self.method.v_counter, key)

                                if len(self.total_costs_hist) > 4*self.running_average_interval and len(self.total_costs_hist)>self.n_convergence_check:
                                    total_costs_hist_pd_av = pd.DataFrame(self.total_costs_hist).rolling(self.running_average_interval).mean()
                                    total_costs_hist_pd_av[0][:self.running_average_interval-1] = self.total_costs_hist[:self.running_average_interval-1]
                                self.save_results(self.total_costs_hist)
                                t1 = time.time()

        if n<self.n_check_policy or n<self.n_convergence_check or n < 4*self.running_average_interval:
            total_costs_hist_pd_av = self.total_costs_hist
        if n<self.max_epochs-self.N_testing:
            for l in range(n,n+self.N_testing):
                self.n = l
                t1 = time.time()
                total_utility_n = self.forward_backward_pass()
                self.total_costs_hist.append(total_utility_n)
                t2 = time.time() - t1
                print('Duration of one PL-VFA iteration: ' + str(t2))
                print('Episode: ' + str(l) + ' of ' + str(self.max_epochs) + ' episodes')
        self.save_results(self.total_costs_hist)
        return self.v,total_costs_hist_pd_av

    def save_results(self,total_costs_hist_pd_av):
        np.save(self.base_configuration.results_path / 'drivers.npy',self.drivers)
        np.save(self.base_configuration.results_path / 'costs.npy', self.costs)
        np.save(self.base_configuration.results_path / 'cds_non_matched.npy', self.cds_non_matched)
        np.save(self.base_configuration.results_path / 'request_delivered.npy', self.request_delivered)
        np.save(self.base_configuration.results_path / 'v.npy', self.v)
        np.save(self.base_configuration.results_path / 'total_costs_hist.npy', np.array(total_costs_hist_pd_av))
        np.save(self.base_configuration.results_path / 'hiring_decisions.npy', self.hiring_decisions)

    def store_results(self,output,post_decision_state,costs_to_go_max_act,max_act,t):
        k = int(self.n>self.N_testing)

        if k == 1:
            n = self.max_epochs - self.N_testing - self.n
        else:
            n = self.n

        self.drivers[k,n,0,t] = post_decision_state[0]
        self.drivers[k,n, 1, t] = post_decision_state[1]
        self.drivers[k,n, 2, t] = post_decision_state[2]

        self.costs[k,n,0,t] = costs_to_go_max_act
        self.costs[k,n, 1, t] = output.operational_utility
        self.costs[k,n, 2, t] = -post_decision_state[0]*self.problem.c_fix
        self.costs[k,n, 3, t] = output.c_fd
        self.costs[k,n, 4, t] = output.costs_empty_routing
        self.costs[k,n, 5, t] = output.c_cd1
        self.costs[k,n, 6, t] = output.c_cd2
        self.costs[k,n, 7, t] = output.c_pen
        self.costs[k,n, 8, t] = max_act*(max_act<0)*self.severance_payment

        self.hiring_decisions[k,n,t] = max_act

        self.cds_non_matched[k,n,0,t] = output.s_g
        self.cds_non_matched[k,n, 1, t] = output.s_o

        self.request_delivered[k,n,0,t] = output.requests_delivered_by_FDs
        self.request_delivered[k,n, 1, t] = output.requests_delivered_by_GWs
        self.request_delivered[k,n, 2, t] = output.requests_delivered_by_ODs
        self.request_delivered[k,n, 3, t] = output.requests_not_delivered

    # This function implements the loop through the environment
    def forward_backward_pass(self):
        # Initialize parameters
        state_t = np.zeros(3)
        state_t[0] = self.base_configuration.init_state[0]
        state_t[1] = self.base_configuration.init_state[1]
        state_t[2] = self.base_configuration.init_state[2]
        state_aggregated = np.zeros(3)
        post_decision_state_t = np.zeros(3)
        cumulated_total_utility = np.zeros(self.problem.tmax+1)

        # Forward Pass in which trajectories are generated and slopes updated
        for t in range(0,self.problem.tmax+1):

                ### 1.Obtain post decision state of t
                candidate_fds_post_decision = np.minimum(state_t[0]+self.act,np.ones(self.act.shape[0])*self.problem.maxFD).astype(int)
                state_aggregated[0] = int(self.method.fd_vec_aggr[int(state_t[0]),t])
                state_aggregated[1] = int(self.method.gw_vec[int(state_t[1]),t])
                state_aggregated[2] = int(self.method.od_vec[int(state_t[2]),t])

                if (state_aggregated[0],state_aggregated[1],state_aggregated[2],t) not in self.v:
                    self.v[state_aggregated[0],state_aggregated[1],state_aggregated[2],t] = self.shape_slope_vector.copy()

                cd_state_unaggregated = [state_t[1],state_t[2]]

                f,max_act,costs_to_go_max_act,output = self.action_selection(candidate_fds_post_decision,state_aggregated,cd_state_unaggregated,t)

                cumulated_total_utility[t] = costs_to_go_max_act

                # Post decision state
                post_decision_state_t[0] =  min(state_t[0]+max_act,self.problem.maxFD)
                post_decision_state_t[1] = state_t[1]
                post_decision_state_t[2] = state_t[2]

                if self.n >self.max_epochs-self.N_testing or self.n<self.N_testing:
                    self.store_results(output,post_decision_state_t,costs_to_go_max_act,max_act,t)


                ### 2.Value of next pre-decision state
                # Sample observation of post decision state's value
                next_state = state_transfer_fct(post_decision_state_t,self.problem,t,self.fluid_model,self.sCD_buffer)
                cd_state_unaggregated[0] = next_state[1]
                cd_state_unaggregated[1] = next_state[2]

                candidate_fds_post_decision = np.minimum(next_state[0]+self.act,np.ones(self.act.shape[0])*self.problem.maxFD).astype(int)
                state_aggregated[0] = int(self.method.fd_vec_aggr[int(next_state[0]),t])
                state_aggregated[1] = int(self.method.gw_vec[int(next_state[1]),t])
                state_aggregated[2]  = int(self.method.od_vec[int(next_state[2]),t])

                t_next = min(t+1,self.problem.tmax)

                if (state_aggregated[0],state_aggregated[1],state_aggregated[2],t_next) not in self.v:
                    self.v[state_aggregated[0],state_aggregated[1],state_aggregated[2],t_next] = self.shape_slope_vector.copy()

                value_next_state,_, __,___ = self.action_selection(candidate_fds_post_decision,state_aggregated,cd_state_unaggregated,t_next)

                ### 3. Sample observation of post decision state's value - 1 (only if post decision state not 0)
                if post_decision_state_t[0]>0:

                    next_state_minus = np.array([post_decision_state_t[0]-self.fd_discretization,post_decision_state_t[1], post_decision_state_t[2]])
                    tmp_state_minus = state_transfer_fct(next_state_minus, self.problem, t, self.fluid_model, self.sCD_buffer)
                    next_state_minus[0] = tmp_state_minus[0]
                    next_state_minus[1] = tmp_state_minus[1]
                    next_state_minus[2] = tmp_state_minus[2]

                    cd_state_unaggregated[0] = next_state_minus[1]
                    cd_state_unaggregated[1] = next_state_minus[2]

                    t_next = min(t + 1, self.problem.tmax)
                    candidate_fds_post_decision = np.minimum(next_state_minus[0]+self.act,np.ones(self.act.shape[0])*self.problem.maxFD).astype(int)
                    state_aggregated[0] = int(self.method.fd_vec_aggr[int(next_state_minus[0]), t])
                    state_aggregated[1] = int(self.method.gw_vec[int(next_state_minus[1]), t])
                    state_aggregated[2] = int(self.method.od_vec[int(next_state_minus[2]), t])

                    if (state_aggregated[0], state_aggregated[1], state_aggregated[2], t_next) not in self.v:
                        self.v[state_aggregated[0], state_aggregated[1], state_aggregated[2], t_next] = self.shape_slope_vector.copy()

                    value_next_state_minus,_,__,___ = self.action_selection(candidate_fds_post_decision,state_aggregated,cd_state_unaggregated,t_next)

                    # Sample obs of slopes around post-decision state and post-decision state + 1
                    v_obs_nfd = value_next_state - value_next_state_minus

                ### 4.Sample observation of post decision state's value + 1 (only if post decision state not maxFD)
                if post_decision_state_t[0] < self.problem.maxFD:

                    next_state_plus = np.array([min(post_decision_state_t[0] + self.fd_discretization, self.problem.maxFD),post_decision_state_t[1],post_decision_state_t[2]])

                    tmp_state_plus = state_transfer_fct(next_state_plus, self.problem, t, self.fluid_model, self.sCD_buffer)
                    next_state_plus[0] = tmp_state_plus[0]
                    next_state_plus[1] = tmp_state_plus[1]
                    next_state_plus[2] = tmp_state_plus[2]

                    cd_state_unaggregated[0] = next_state_plus[1]
                    cd_state_unaggregated[1] = next_state_plus[2]

                    t_next = min(t+1,self.problem.tmax)
                    candidate_fds_post_decision = np.minimum(next_state_plus[0]+self.act,np.ones(self.act.shape[0])*self.problem.maxFD).astype(int)
                    state_aggregated[0] = int(self.method.fd_vec_aggr[int(next_state_plus[0]), t])
                    state_aggregated[1] = int(self.method.gw_vec[int(next_state_plus[1]), t])
                    state_aggregated[2] = int(self.method.od_vec[int(next_state_plus[2]), t])

                    if (state_aggregated[0], state_aggregated[1], state_aggregated[2], t_next) not in self.v:
                        self.v[state_aggregated[0], state_aggregated[1], state_aggregated[2], t_next] = self.shape_slope_vector.copy()

                    value_next_state_plus,_,__,___ = self.action_selection(candidate_fds_post_decision,state_aggregated,cd_state_unaggregated,t_next)

                    v_obs_nfd_plus = value_next_state_plus - value_next_state

                # Do some initial greedy policy testing before starting to update value function
                if self.n > self.N_testing:
                    slope_id_of_post_decision_fd = int(self.method.fd_vec_disc[int(post_decision_state_t[0]),t]-1)
                    state_aggregated[0] = int(self.method.fd_vec_aggr[int(state_t[0]),t])
                    state_aggregated[1] = int(self.method.gw_vec[int(post_decision_state_t[1]),t])
                    state_aggregated[2] = int(self.method.od_vec[int(post_decision_state_t[2]),t])

                    if (state_aggregated[0],state_aggregated[1],state_aggregated[2],t) in self.method.v_counter:
                        self.method.v_counter[state_aggregated[0],state_aggregated[1],state_aggregated[2],t] += 1
                    else:
                        self.method.v_counter[state_aggregated[0],state_aggregated[1],state_aggregated[2],t] = 1

                    # v-update through projection to ensure concavity
                    v_unprojected = self.v[state_aggregated[0],state_aggregated[1],state_aggregated[2],t].copy()
                    limit = 2

                    # Slope left to post decision FD
                    if post_decision_state_t[0]>0:
                        v_unprojected[slope_id_of_post_decision_fd] = (1-self.alpha)*self.v[state_aggregated[0],state_aggregated[1],state_aggregated[2],t][slope_id_of_post_decision_fd]+self.alpha*v_obs_nfd
                    else:
                        limit = 0

                    # Slope right to post decision FD
                    if post_decision_state_t[0] < self.problem.maxFD:
                        slope_id_of_post_decision_fd = slope_id_of_post_decision_fd+1
                        v_unprojected[slope_id_of_post_decision_fd] = (1 - self.alpha) * self.v[state_aggregated[0],state_aggregated[1],state_aggregated[2],t][slope_id_of_post_decision_fd] + self.alpha * v_obs_nfd_plus
                    else:
                        limit = 1
                    current_FD_post_decision_state = post_decision_state_t[0]
                    current_FD_post_decision_state_plus = post_decision_state_t[0]+self.fd_discretization
                    v_tmp=self.projection(v_unprojected,current_FD_post_decision_state,current_FD_post_decision_state_plus,limit,t)
                    self.v[state_aggregated[0],state_aggregated[1],state_aggregated[2], t] = v_tmp.copy()
                state_t = next_state.copy()
        return np.sum(cumulated_total_utility)

    # This function selects the action maximizing the post decision state value function
    def action_selection(self,fds_post_decision,cd_state_aggregated,cd_state_unaggregated,t):
        fd_pre_decision = fds_post_decision[0]
        lower_limit_brent_search = (self.include_firing_decisions==0)*fd_pre_decision
        upper_limit_brent_search = fds_post_decision[fds_post_decision.shape[0] - 1]
        value_lower_limit = self.value_function(lower_limit_brent_search, fd_pre_decision, fds_post_decision, cd_state_aggregated,cd_state_unaggregated, t)
        if self.include_firing_decisions==0:
            value_lower_limit_plus_one = self.value_function(min(lower_limit_brent_search+1,self.problem.maxFD), fd_pre_decision, fds_post_decision,cd_state_aggregated, cd_state_unaggregated, t)
        else:
            value_lower_limit_plus_one = value_lower_limit

        if value_lower_limit_plus_one<value_lower_limit and self.include_firing_decisions==0:
            max_act = 0
            f = value_lower_limit
            costs_to_go_max_act, output = self.calc_costs_to_go([int(fd_pre_decision + max_act), cd_state_unaggregated[0], cd_state_unaggregated[1]], t, fd_pre_decision)
        else:
            middle_point = int(np.round((upper_limit_brent_search + lower_limit_brent_search) / 2))
            value_middle_point = self.value_function(middle_point, fd_pre_decision, fds_post_decision, cd_state_aggregated,cd_state_unaggregated, t)
            brent_search_output = brent(self.value_function,fd_pre_decision,fds_post_decision,
                                        cd_state_aggregated,cd_state_unaggregated,t, a=lower_limit_brent_search,
                                        b=upper_limit_brent_search,x0=middle_point, f0=-value_middle_point,multiple=self.fd_discretization)
            max_act = np.round(brent_search_output[0]) - fd_pre_decision
            f = brent_search_output[1]
            costs_to_go_max_act, output = self.calc_costs_to_go([int(fd_pre_decision + max_act), cd_state_unaggregated[0], cd_state_unaggregated[1]], t,fd_pre_decision)
        return f,max_act, costs_to_go_max_act,output

    # This function calculates the post decision state value function
    def V_post_decision_state_fct(self,v,fd_post_decision,t):
        id = int(self.method.fd_vec_disc[int(fd_post_decision),t])
        return sum(self.FD_stepsize[:id] * v[:id])

    # This function calculates the state value function
    def value_function(self,fd_post_decision,fd_pre_decision,fds_post_decision,cd_state_aggregated,cd_state_unaggregated,t):
        fd_u = int(self.fd_discretization*np.ceil(fd_post_decision/self.fd_discretization))
        post_decision_state = [fd_post_decision, cd_state_unaggregated[0], cd_state_unaggregated[1]]

        # For Brent search, when fds are continuous
        if (fd_u != fd_post_decision):
            fd_l = int(self.fd_discretization*np.floor(fd_post_decision/self.fd_discretization))
            if fd_post_decision > fds_post_decision[fds_post_decision.shape[0] - 1]:
                fds_help_1 = fds_post_decision[fds_post_decision.shape[0] - 1]-self.fd_discretization
                fds_help_2 = fds_post_decision[fds_post_decision.shape[0] - 1] - 2*self.fd_discretization
                v_post = (self.V_post_decision_state_fct(self.v[cd_state_aggregated[0],cd_state_aggregated[1],cd_state_aggregated[2],t], fds_help_1,t) -
                          self.V_post_decision_state_fct(self.v[cd_state_aggregated[0],cd_state_aggregated[1],cd_state_aggregated[2],t], fds_help_2,t)) * \
                         (fd_post_decision - fds_help_1)/(fds_help_1-fds_help_2) \
                         + self.V_post_decision_state_fct(self.v[cd_state_aggregated[0],cd_state_aggregated[1],cd_state_aggregated[2],t], fds_help_1,t)
            else:
                V_u = self.V_post_decision_state_fct(self.v[cd_state_aggregated[0],cd_state_aggregated[1],cd_state_aggregated[2],t], fd_u,t)
                V_l = self.V_post_decision_state_fct(self.v[cd_state_aggregated[0],cd_state_aggregated[1],cd_state_aggregated[2],t], fd_l,t)
                v_post = (V_u - V_l) * (fd_post_decision - fd_l)/(fd_u-fd_l) + V_l
            costs_to_go = self.calc_costs_to_go_continuous(fd_u, fd_l,post_decision_state,t,fd_pre_decision)
        # For normal cases
        else:
            v_post = self.V_post_decision_state_fct(self.v[cd_state_aggregated[0],cd_state_aggregated[1],cd_state_aggregated[2],t], fd_post_decision,t)
            costs_to_go,_ = self.calc_costs_to_go(post_decision_state, t, fd_pre_decision)
        return (costs_to_go + self.gamma * (v_post))

    # This function calculates the operational costs
    def calc_costs_to_go(self,post_decision_state, t, fd_pre_decision):
        i = int(post_decision_state[0])
        j = int(post_decision_state[1])
        k = int(post_decision_state[2])
        self.severance_payment = self.problem.severance_pen* (i - fd_pre_decision < 0)*(fd_pre_decision-i)*self.include_firing_decisions

        if ((i,j,k,t) in self.costs_to_go_buffer) and (self.n < self.max_epochs-self.N_testing) and (self.n>self.N_testing) and not self.costs_to_go_calculation:
            costs_to_go = self.costs_to_go_buffer[i,j,k,t] - self.severance_payment
            output = []
        else:
            if i == 0:
                output = self.fluid_model[t].solve_model(1, j, k)
                costs_to_go_1 = output.total_utility
                sCD1_1 = output.s_g
                sCD2_1 = output.s_o

                output_2 = self.fluid_model[t].solve_model(2, j, k)
                costs_to_go_2 = output_2.total_utility
                sCD1_2 = output_2.s_g
                sCD2_2 = output_2.s_o

                costs_to_go_0 = costs_to_go_1 - (costs_to_go_2 - costs_to_go_1)
                costs_to_go_tmp = costs_to_go_0
                costs_to_go = costs_to_go_0 - self.severance_payment
                sCD1 = sCD1_1 - (sCD1_2 - sCD1_1)
                sCD2 = sCD2_1 - (sCD2_2 - sCD2_1)
            else:
                output = self.fluid_model[t].solve_model(i, j, k)
                costs_to_go_tmp = output.total_utility
                costs_to_go = output.total_utility - self.severance_payment
                sCD1 = output.s_g
                sCD2 = output.s_o
            self.costs_to_go_buffer[i,j,k,t] = costs_to_go_tmp
            self.sCD_buffer[i,j,k,t] = [sCD1, sCD2]
        return costs_to_go,output

    # This function calculates costs if using continuous values for FDs
    def calc_costs_to_go_continuous(self, fd_u_post_decision, fd_l_post_decision,post_decision_state, t,fd_pre_decision):
        fd_post_decision = post_decision_state[0]
        cd1 = post_decision_state[1]
        cd2 = post_decision_state[2]
        costs_to_go_u,_ = self.calc_costs_to_go([fd_u_post_decision, cd1, cd2],t, fd_pre_decision)
        costs_to_go_l,_ = self.calc_costs_to_go([fd_l_post_decision, cd1, cd2], t, fd_pre_decision)
        costs_to_go = (costs_to_go_u - costs_to_go_l) * (fd_post_decision - fd_l_post_decision)/(fd_u_post_decision-fd_l_post_decision) + costs_to_go_l
        return costs_to_go

    # This function ensures convexity of sloep approximation
    def projection(self,v_unprojected, post_decision_state, post_decision_state_plus, limit,t):
        if limit == 2:
            id0 = int(self.method.fd_vec_disc[int(post_decision_state), t] - 1)
            id1 = int(self.method.fd_vec_disc[int(post_decision_state_plus), t] - 1)
            if v_unprojected[id0] < v_unprojected[id1]:
                tmp = (v_unprojected[id0] + v_unprojected[id1]) / 2
                v_unprojected[id0] = tmp
                v_unprojected[id1] = tmp
            for i in range(0, v_unprojected.shape[0]):
                if i < id0 and v_unprojected[i] <= v_unprojected[id0]:
                    v_unprojected[i] = v_unprojected[id0]
                elif i > id1 and v_unprojected[i] >= v_unprojected[id1]:
                    v_unprojected[i] = v_unprojected[id1]
        elif limit == 0:
            id1 = int(self.method.fd_vec_disc[int(post_decision_state_plus), t] - 1)
            for i in range(0, v_unprojected.shape[0]):
                if i > id1 and v_unprojected[i] >= v_unprojected[id1]:
                    v_unprojected[i] = v_unprojected[id1]
        elif limit == 1:
            id0 = int(self.method.fd_vec_disc[int(post_decision_state), t] - 1)
            for i in range(0, v_unprojected.shape[0]):
                if i < id0 and v_unprojected[i] <= v_unprojected[id0]:
                    v_unprojected[i] = v_unprojected[id0]
        return v_unprojected