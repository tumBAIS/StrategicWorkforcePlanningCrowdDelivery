"""
File: model_creation.py
Author: Julius Luy
Date: November 10th 2023
Description: This class implements and stores all MDP environment settigns and parameters.

Parameters
----------
base_configuration : object
    Contains all configuration parameters.
"""

import numpy as np
import time
import csv
from csv import writer
import os,sys
from scipy.optimize import curve_fit

from common.cost_calculation import cost_calculation
from common.revenue_calculation import revenue_calculation
from common.find_paths import find_paths

from pathlib import Path
current_working_dir = Path(os.getcwd())
main_project_dir = str(current_working_dir.parent)
sys.path.append(main_project_dir+'/cpp/cmake-build-release/bin')


# Definition of problem_parameters
class model_creation:
    def __init__(self,base_configuration):
        input_file = base_configuration.input_file
        x = {}
        with open(input_file, 'r',encoding="utf-8") as fd:
            reader = csv.reader(fd)
            for row in reader:
                x[row[0]] = row[1]
        fd.close()
        self.instance_name = base_configuration.input_file
        self.analysis_path = base_configuration.analysisPath

        # Fix cost per hr of staffing a dedicated driver
        self.c_fix = float(x['c_fix'])

        # Variable costs
        self.c_d = float(x['c_d'])
        self.c_g = float(x['c_g'])
        self.c_o = float(x['c_o'])

        # Penalty costs
        self.c_Rpen = float(x['c_rpen']) # Penalty/timestep for each request not served, i.e., revenue compensation + penalty fee

        # Time horizon
        self.tmax = int(float(x['tmax']))

        # Maximum number of drivers of states
        self.maxCD1 = int(x['maxcd1'])
        self.maxCD2 = int(x['maxcd2'])
        self.maxFD = int(x['maxfd'])

        # Action space
        self.action_space = np.arange(0, self.maxFD + 1)
        self.firing_decisions = x['include_firing_decisions']

        # Number of locations
        self.loc = int(x['loc']) # No. of locations

        total_demand = float(x['total_demand'])

        self.random_od_patterns = str(x['random_od_patterns'])

        # Mobility patterns
        if base_configuration.load_data != 'yes':
            self.route_pat,self.req_pat,self.mob_pat,self.demand_per_station,self.demand_arrival_patterns = self.generate_mobility_network(self.loc,total_demand,self,base_configuration) # demand_dist

            if base_configuration.save_data == 'yes':
                route_pat_file = base_configuration.results_path / 'route_pat.npy'
                f = open(route_pat_file, 'wb')
                np.save(f, self.route_pat)

                req_pat_file = base_configuration.results_path / 'req_pat.npy'
                f = open(req_pat_file, 'wb')
                np.save(f, self.req_pat)

                mob_pat_file = base_configuration.results_path / 'mob_pat.npy'
                f = open(mob_pat_file, 'wb')
                np.save(f, self.mob_pat)

                demand_per_station_file = base_configuration.results_path / 'demand_per_station.npy'
                f = open(demand_per_station_file, 'wb')
                np.save(f,self.demand_per_station)

                demand_arrival_patterns_file = base_configuration.results_path/'demand_arrival_patterns.npy'
                f = open(demand_arrival_patterns_file, 'wb')
                np.save(f,self.demand_arrival_patterns)
        else:
            self.req_pat = np.load(base_configuration.analysisPath/'req_pat.npy')
            self.demand_arrival_patterns = np.load(base_configuration.analysisPath/ 'demand_arrival_patterns.npy')
            self.route_pat,_,self.mob_pat,self.demand_per_station,____ = self.generate_mobility_network(self.demand_arrival_patterns.shape[0], total_demand, self,base_configuration)
            test = 0

        # Definition of demand-development over time
        self.demand = np.zeros((self.tmax + 1, self.loc))
        self.demand_case = x['demand_case']  # 'peak' # peak;constant;linear
        for i in range(0, self.loc):
            if self.demand_case == 'peak':
                tvec = np.arange(0, self.tmax + 1)
                peak_magnitude = np.fromstring(x['peak_magnitude'],dtype = float,sep=',')
                peak_sharpness = np.fromstring(x['peak_sharpness'],dtype=float,sep=',')
                t_peaks = np.fromstring(x['t_peaks'],dtype=float,sep=',')
                t_peaks = t_peaks.astype(int)
                for j in range(0,len(t_peaks)):
                    tmp = self.demand_per_station[i]*peak_magnitude[j]
                    self.demand[:, i] += tmp * np.exp(-peak_sharpness[j] * (tvec - t_peaks[j]) ** 2)
                self.demand[:, i]+= self.demand_per_station[i]
            elif self.demand_case == 'cagr_growth':
                tvec = np.arange(0, self.tmax + 1)
                self.demand[0, i] = self.demand_per_station[i]/((1+float(x['growth_rate']))**self.tmax)
                self.demand[1:, i] = self.demand[0, i]*(1+float(x['growth_rate']))**tvec[1:]
            elif self.demand_case == 'ramp_up':
                tvec = np.arange(0, self.tmax + 1)
                self.ramp_up_time = int(x['ramp_up_time'])
                self.demand[0, i] = self.demand_per_station[i]*1/self.ramp_up_time
                self.demand[1:self.ramp_up_time, i] = self.demand_per_station[i] * tvec[1:self.ramp_up_time] / self.ramp_up_time
                self.demand[self.ramp_up_time:, i] = self.demand_per_station[i]
            elif self.demand_case == 'growth_stepwise':
                increments = float(x['increments'])
                demand_increments = self.demand_per_station[i]/increments
                t_increments = self.tmax/increments
                current_demand = demand_increments
                self.demand[0, i] = current_demand
                for t in range(1,self.tmax+1,int(t_increments)):
                    self.demand[t:t+int(t_increments),i] = current_demand
                    current_demand += demand_increments
            else:
                self.demand[:, i] = self.demand_per_station[i]
        np.save(base_configuration.results_path / 'demand.npy',self.demand)

        # Resignation patterns
        self.resig_fd = float(x['resig_fd'])
        self.resig_gw = x['resig_gw']
        if self.resig_gw != 'dynamic':
            self.resig_gw = float(self.resig_gw)
        self.resig_od = x['resig_od']
        if self.resig_od != 'dynamic':
            self.resig_od = float(self.resig_od)

        # Growth patterns
        self.add_gw = float(x['add_gw'])
        self.add_od = float(x['add_od'])

        # upper lower bounds for resignation and joining rates
        self.p_add_cd1_low = float(x['p_add_cd1_low'])
        self.p_add_cd2_low = float(x['p_add_cd2_low'])
        self.p_resig_cd1_low = float(x['p_resig_cd1_low'])
        self.p_resig_cd2_low = float(x['p_resig_cd2_low'])

        self.p_add_cd1_high = float(x['p_add_cd1_high'])
        self.p_add_cd2_high = float(x['p_add_cd2_high'])
        self.p_resig_cd1_high = float(x['p_resig_cd1_high'])
        self.p_resig_cd2_high = float(x['p_resig_cd2_high'])


        self.v_avg = float(x['v_avg'])

        # Distance vectors
        if x['load_data'] != 'yes':
            coords = np.zeros((self.loc,2))
            k=0
            if self.loc==2:
                l = self.loc
            else:
                l = np.sqrt(self.loc).astype(int)
            delta = 1
            if self.loc==2: # for 2 locations extra case
                coords[0] = np.array([0,0])
                coords[1] = np.array([0,1])
            else:
                for i in range(l):
                    for j in range(l):
                        coords[k] = np.array([i,j])
                        k = k+1
            self.r = np.zeros((self.loc,self.loc))
            for i in range(0,self.loc):
                for j in range(0,self.loc):
                    self.r[i,j] = 2*np.sqrt((coords[i][0]-coords[j][0])**2+(coords[i][1]-coords[j][1])**2)+delta
            r_scaling = float(x['r_scaling'])
            self.r = r_scaling*self.r
        else:
            self.r = np.load(base_configuration.analysisPath /'od_matrix.npy')


        self.mu = (self.r/self.v_avg)**(-1)
        np.save(base_configuration.results_path.joinpath('mu.npy'), self.mu)

        self.c, self.c_pen_req,self.c_empty_routing  = cost_calculation(self)  # currently negative costs
        np.save(base_configuration.results_path.joinpath('c1.npy'),self.c)
        np.save(base_configuration.results_path / 'c_empty_routing.npy', self.c_empty_routing)
        self.severance_pen = float(x['severance_pen'])

        self.gw_capacity = float(x['gw_capacity'])
        self.od_capacity = float(x['od_capacity'])

        # Order is important
        with open(base_configuration.results_path.joinpath('constants.txt'), 'w') as f:
            f.write(str(self.maxFD))
            f.write('\n')
            f.write(str(self.maxCD1))
            f.write('\n')
            f.write(str(self.maxCD2))
            f.write('\n')
            f.write(str(self.tmax))
            f.write('\n')
            f.write(str(self.c_pen_req))
            f.write('\n')
            f.write(str(self.c_fix))
            f.write('\n')
            f.write(str(self.gw_capacity))
            f.write('\n')
            f.write(str(self.od_capacity))
        f.close()


        # This part only relevant if exact solution approach wished
        if base_configuration.solution_approach == 'exact' and base_configuration.mode == 'train':
            if base_configuration.reward_calculation == 'yes':

                time.sleep(2)
                index = find_paths(sys.path,'cpp/cmake-build-release/bin')
                os.system(sys.path[index]+'/cd_cpp'+' '+base_configuration.results_path.as_posix())
                time.sleep(2)

                file = base_configuration.results_path.joinpath('rew.npy')
                self.rew = np.load(file)

                # Calculation for 0 FDs to avoid discontinuities
                self.rew[0, :, :, :] = self.rew[1, :, :, :] - (self.rew[2, :, :, :] - self.rew[1, :, :, :])

                file = base_configuration.results_path.joinpath('sCD1.npy')
                self.sCD1 = np.load(file)
                file = base_configuration.results_path.joinpath('sCD2.npy')
                self.sCD2 = np.load(file)
                file = base_configuration.results_path.joinpath('c_fd.npy')
                self.c_fd = np.load(file)
                file = base_configuration.results_path.joinpath('c_cd1.npy')
                self.c_cd1 = np.load(file)
                file = base_configuration.results_path.joinpath('c_cd2.npy')
                self.c_cd2 = np.load(file)
                file = base_configuration.results_path.joinpath('c_pen.npy')
                self.c_pen = np.load(file)


                file = base_configuration.results_path.joinpath('costs_empty_routing.npy')
                self.costs_empty_routing = np.load(file)

                file = base_configuration.results_path.joinpath('c_empty_routing.npy')
                self.c_empty_routing = np.load(file)

                file = base_configuration.results_path.joinpath('c1.npy')
                self.c = np.load(file)


                self.p_add_cd1_low = float(x['p_add_cd1_low'])
                self.p_add_cd2_low = float(x['p_add_cd2_low'])
                self.p_resig_cd1_low = float(x['p_resig_cd1_low'])
                self.p_resig_cd2_low = float(x['p_resig_cd2_low'])

                self.p_add_cd1_high = float(x['p_add_cd1_high'])
                self.p_add_cd2_high = float(x['p_add_cd2_high'])
                self.p_resig_cd1_high = float(x['p_resig_cd1_high'])
                self.p_resig_cd2_high = float(x['p_resig_cd2_high'])

                if self.resig_gw != 1:
                    self.p_resig_cd1 = self.resig_gw*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                    self.p_resig_cd2 = self.resig_od*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                else:
                    self.p_resig_cd1 = self.p_resig_cd1_low*(1-self.sCD1[:, :, :, :])+self.p_resig_cd1_high*self.sCD1[:, :, :, :]
                    self.p_resig_cd2 = self.p_resig_cd2_low*(1-self.sCD2[:,:,:,:]) + self.p_resig_cd2_high*self.sCD2[:,:,:,:]
                if self.add_gw != 1:
                    self.p_add_cd1 = self.add_gw*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                    self.p_add_cd2 = self.add_od*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                else:
                    self.p_add_cd1 = self.p_add_cd1_high*(1-self.sCD1[:, :, :, :])+self.p_add_cd1_low*self.sCD1[:, :, :, :]
                    self.p_add_cd2 = self.p_add_cd2_high*(1-self.sCD2[:, :, :, :])+self.p_add_cd2_low*self.sCD2[:, :, :, :]
            else:
                file = base_configuration.analysisPath+'/rew.npy'
                self.rew = np.load(file)

                # EXPERIMENTAL - needs to be adjusted
                self.rew[0, :, :, :] = self.rew[1, :, :, :] - (self.rew[2, :, :, :] - self.rew[1, :, :, :])

                file = base_configuration.results_path.joinpath('sCD1.npy')
                self.sCD1 = np.load(file)
                file = base_configuration.results_path.joinpath('sCD2.npy')
                self.sCD2 = np.load(file)
                file = base_configuration.results_path.joinpath('c_fd.npy')
                self.c_fd = np.load(file)
                file = base_configuration.results_path.joinpath('c_cd1.npy')
                self.c_cd1 = np.load(file)
                file = base_configuration.results_path.joinpath('c_cd2.npy')
                self.c_cd2 = np.load(file)
                file = base_configuration.results_path.joinpath('c_pen.npy')
                self.c_pen = np.load(file)

                file = base_configuration.results_path.joinpath('c_empty_routing.npy')
                self.c_empty_routing = np.load(file)

                file = base_configuration.results_path.joinpath('costs_empty_routing.npy')
                self.costs_empty_routing = np.load(file)

                file = base_configuration.results_path.joinpath('c1.npy')
                self.c = np.load(file)


                self.p_add_cd1_low = float(x['p_add_cd1_low'])
                self.p_add_cd2_low = float(x['p_add_cd2_low'])
                self.p_resig_cd1_low = float(x['p_resig_cd1_low'])
                self.p_resig_cd2_low = float(x['p_resig_cd2_low'])

                self.p_add_cd1_high = float(x['p_add_cd1_high'])
                self.p_add_cd2_high = float(x['p_add_cd2_high'])
                self.p_resig_cd1_high = float(x['p_resig_cd1_high'])
                self.p_resig_cd2_high = float(x['p_resig_cd2_high'])

                if self.resig_gw != 1:
                    self.p_resig_cd1 = self.resig_gw*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                    self.p_resig_cd2 = self.resig_od*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                else:
                    self.p_resig_cd1 = self.p_resig_cd1_low*(1-self.sCD1[:, :, :, :])+self.p_resig_cd1_high*self.sCD1[:, :, :, :]
                    self.p_resig_cd2 = self.p_resig_cd2_low*(1-self.sCD2[:,:,:,:]) + self.p_resig_cd2_high*self.sCD2[:,:,:,:]
                if self.add_gw != 1:
                    self.p_add_cd1 = self.add_gw*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                    self.p_add_cd2 = self.add_od*np.ones((self.maxFD+1,self.maxCD1+1,self.maxCD2+1,self.tmax+1))
                else:
                    self.p_add_cd1 = self.p_add_cd1_high*(1-self.sCD1[:, :, :, :])+self.p_add_cd1_low*self.sCD1[:, :, :, :]
                    self.p_add_cd2 = self.p_add_cd2_high*(1-self.sCD2[:, :, :, :])+self.p_add_cd2_low*self.sCD2[:, :, :, :]
        if base_configuration.load_existing_costs == 'yes':
            file = base_configuration.results_path.joinpath('rew.npy')
            self.rew = np.load(file)

    def generate_mobility_network(self,loc, total_demand, problem, base_configuration):

        if base_configuration.load_data == 'yes':
            demand_arrival_patterns = np.zeros(loc)
            req_pat = np.zeros((loc, loc))
            req_pat[:, :] = problem.req_pat[:, :]
            demand_arrival_patterns[:] = problem.demand_arrival_patterns[:]
        else:
            req_pat = np.random.random((loc, loc))
            for i in range(0, loc):
                req_pat[i] = (1 / np.sum(req_pat[i])) * req_pat[i]
            demand_arrival_patterns = np.random.random(loc)
            demand_arrival_patterns = demand_arrival_patterns / (sum(demand_arrival_patterns))

        demand = np.zeros(loc)
        mob_pat = np.zeros(loc)

        if problem.random_od_patterns != 0:
            for i in range(0, loc):
                mob_pat[i] = np.random.random(1)[0]
            mob_pat[:] = mob_pat[:] / sum(mob_pat)
            route_pat = np.random.random((loc, loc))
            for i in range(0, loc):
                route_pat[i] = (1 / np.sum(route_pat[i])) * route_pat[i]
        else:
            mob_pat[:] = demand_arrival_patterns[:]
            route_pat = req_pat.copy()
        for i in range(0, loc):
            demand[i] = demand_arrival_patterns[i] * total_demand
        return route_pat, req_pat, mob_pat, demand, demand_arrival_patterns