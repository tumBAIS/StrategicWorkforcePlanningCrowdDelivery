#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <chrono>
#include "npy.hpp"
#include "spdlog/spdlog.h"
#include "utils_cd.hpp"
#include "operational_model_costs.hpp"
#include "gurobi_c++.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;


int main(int argc, char** argv) {
    string analysis_path = argv[1];
    auto route_pat = load_numpy_array<NPYArray2D>(analysis_path+"/route_pat.npy");
    auto req_pat = load_numpy_array<NPYArray2D>(analysis_path+"/req_pat.npy");
    auto mob_pat = load_numpy_array<NPYArray1D>(analysis_path+"/mob_pat.npy");

    auto gw_pat = load_numpy_array<NPYArray1D>(analysis_path+"/demand_arrival_patterns.npy");

    auto demand = load_numpy_array<NPYArray2D>(analysis_path+"/demand.npy");
    auto c1 = load_numpy_array<NPYArray3D>(analysis_path+"/c1.npy");
    auto c_empty_routing = load_numpy_array<NPYArray2D>(analysis_path+"/c_empty_routing.npy");
    auto mu = load_numpy_array<NPYArray2D>(analysis_path+"/mu.npy");

    float float_constants[4];
    int int_constants[4];

    std::fstream myfile(analysis_path+"/constants.txt",std::ios_base::in);
    float a;
    int i = 0;
    while (myfile >> a)
    {
        if (i<4){
            int_constants[i] = a;
        }
        else if (i>3){
            float_constants[i-4] = a;
        }
        i++;
    }

    auto c_pen_req = (float)float_constants[0];
    auto maxFD = (int)int_constants[0];
    auto maxCD1 = (int)int_constants[1];
    auto maxCD2 = (int)int_constants[2];
    auto tmax =  (int)int_constants[3];
    auto c_fix = (float)float_constants[1];
    auto gw_capacity = (float)float_constants[2];
    auto od_capacity = (float)float_constants[3];
    spdlog::info(fmt::format("starting"));
    auto t_start = std::chrono::steady_clock::now();

    auto env = new GRBEnv();
    env->set(GRB_IntParam_OutputFlag, 0);
    auto rew = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto sCD1 = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto sCD2 = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto c_fd = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto c_cd1 = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto c_cd2 = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto c_pen = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    auto costs_empty_routing = create_npy_array_4d({maxFD + 1, maxCD1 + 1, maxCD2 + 1, tmax+1});
    for (auto t = 0; t < tmax+1; ++t) {
        spdlog::info(fmt::format("calculate reward for t={}", t));
        auto demand_t = demand.copy_row(t);
        auto ops_model = OperationalModel(env, c1, c_empty_routing, c_pen_req,c_fix, route_pat, req_pat, mu, demand_t, mob_pat,gw_capacity,od_capacity,gw_pat);
        auto t_calculated = false;
        if (t > 0) {
            auto prev_demand = demand.copy_row(t-1);
            if (std::equal(demand_t.data.begin(), demand_t.data.end(), prev_demand.data.begin())) {
                for (auto j = 0; j <= maxCD1; ++j) {
                    for (auto k = 0; k <= maxCD2; ++k) {
                        for (auto i = 0; i <= maxFD; ++i) {
                            rew.at(i, j, k, t) = rew.at(i, j, k, t - 1);
                            sCD1.at(i, j, k, t) = sCD1.at(i, j, k, t - 1);
                            sCD2.at(i, j, k, t) = sCD2.at(i, j, k, t - 1);
                            c_fd.at(i, j, k, t) = c_fd.at(i, j, k, t - 1);
                            c_cd1.at(i, j, k, t) = c_cd1.at(i, j, k, t - 1);
                            c_cd2.at(i, j, k, t) = c_cd2.at(i, j, k, t - 1);
                            c_pen.at(i, j, k, t) = c_pen.at(i, j, k, t - 1);
                            costs_empty_routing.at(i,j,k,t) = costs_empty_routing.at(i,j,k,t-1);
                        }
                    }
                }
                t_calculated = true;
            }
        }
        if (!t_calculated) {
            for (auto j = 0; j <= maxCD1; ++j) {
                for (auto k = 0; k <= maxCD2; ++k) {
                    for (auto i = 0; i <= maxFD; ++i) {
                        auto sol = ops_model.solve_model(std::max(i, 1), j, k);
                        rew.at(i, j, k, t) = sol.total_utility;
                        sCD1.at(i, j, k, t) = sol.s_g;
                        sCD2.at(i, j, k, t) = sol.s_o;
                        c_fd.at(i, j, k, t) = sol.c_fd;
                        c_cd1.at(i, j, k, t) = sol.c_cd1;
                        c_cd2.at(i, j, k, t) = sol.c_cd2;
                        c_pen.at(i, j, k, t) = sol.c_pen;
                        costs_empty_routing.at(i,j,k,t) = sol.costs_empty_routing;

                    }
                }
            }
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    auto t_dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    spdlog::info(fmt::format("finished after {}ms", t_dur));

    save_numpy_array<NPYArray4D>(analysis_path+"/rew.npy", rew);
    save_numpy_array<NPYArray4D>(analysis_path+"/sCD1.npy", sCD1);
    save_numpy_array<NPYArray4D>(analysis_path+"/sCD2.npy", sCD2);
    save_numpy_array<NPYArray4D>(analysis_path+"/c_fd.npy", c_fd);
    save_numpy_array<NPYArray4D>(analysis_path+"/c_cd1.npy", c_cd1);
    save_numpy_array<NPYArray4D>(analysis_path+"/c_cd2.npy", c_cd2);
    save_numpy_array<NPYArray4D>(analysis_path+"/c_pen.npy", c_pen);
    save_numpy_array<NPYArray4D>(analysis_path+"/costs_empty_routing.npy", costs_empty_routing);

    return 0;
}

