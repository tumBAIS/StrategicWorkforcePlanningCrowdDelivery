// This file implements the fluid approximation of operational model

#ifndef CD_CPP_OPERATIONAL_MODEL_HPP
#define CD_CPP_OPERATIONAL_MODEL_HPP

#include <numeric>
#include "utils_costs.hpp"
#include "gurobi_c++.h"
#include "fmt/format.h"

template<typename T>
struct Array2D {
    size_t width;
    std::vector<T> data;
    explicit Array2D<T>() : width(0), data() {};
    explicit Array2D<T>(size_t _width, std::vector<T>&& _data) : width(_width), data(_data) {};
    auto at(size_t i,size_t j) -> T& {
        return this->data[i*width+j];
    }
};


struct SolutionValues {
    double total_utility;
    double operational_utility;
    double s_g;
    double s_o;
    double c_fd;
    double c_cd1;
    double c_cd2;
    double c_pen;
    double requests_delivered_by_FDs;
    double requests_delivered_by_GWs;
    double requests_delivered_by_ODs;
    double requests_not_delivered;
    double costs_empty_routing;
    std::vector<double> request_routes_FDs;
    std::vector<double> request_routes_GWs;
    std::vector<double> request_routes_ODs;
    std::vector<double> request_routes_not_matched;
};

class OperationalModel {

    GRBModel fluid_model;
    NPYArray3D c1;
    NPYArray2D c_empty_routing;
    double c_pen_req;
    double c_fix;
    double gw_capacity;
    double od_capacity;

    // Parameters
    double fd;
    int cd1;
    int cd2;
    const NPYArray2D route_pat;
    const NPYArray2D req_pat;
    const NPYArray2D mu;
    const NPYArray1D demand;
    const NPYArray1D mob_pat;
    const NPYArray1D gw_pat;

    // Variables
    std::vector<GRBVar> p_D; // List[gp.Var]
    std::vector<GRBVar> p_G; // List[gp.Var]
    std::vector<GRBVar> p_O; // List[gp.Var]
    std::vector<GRBVar> p_N; // List[gp.Var]
    std::vector<GRBVar> s_G;
    
    Array2D<GRBVar> E; // List[gp.Var]  # 2d
    Array2D<GRBVar> F; // List[gp.Var]  # 2d
    Array2D<GRBVar> s_O; // List[gp.Var]  # 2d


    // # Constraints
    std::vector<GRBConstr> cons_elhs; // List[gp.Constr]
    std::vector<GRBConstr> cons_erhs; // List[gp.Constr]
    std::vector<GRBConstr> cons_f; // List[gp.Constr]
    std::vector<GRBConstr> cons_a; // List[gp.Constr]  # 2d
    std::vector<GRBConstr> cons_b; // List[gp.Constr]  # 1d, BEFORE 2D
    std::vector<GRBConstr> cons_c; // List[gp.Constr]  # 2d
public:
    // Here the model is initialized with 1 fd and 0 cds
    OperationalModel(GRBEnv env,NPYArray3D _c1, NPYArray2D _c_empty_routing, double _c_pen_req,double _c_fix,
                     const NPYArray2D _route_pat, const NPYArray2D _req_pat,
                     const NPYArray2D _mu, const NPYArray1D _demand, const NPYArray1D _mob_pat, double _gw_capacity, double _od_capacity,const NPYArray1D _gw_pat):
            fluid_model(env),c1(_c1), c_empty_routing(_c_empty_routing), c_pen_req(_c_pen_req),c_fix(_c_fix),
            fd(1.0), cd1(0), cd2(0), route_pat(_route_pat), req_pat(_req_pat), mu(_mu), demand(_demand), mob_pat(_mob_pat), gw_capacity(_gw_capacity), od_capacity(_od_capacity),gw_pat(_gw_pat) {


        fluid_model.getEnv().set(GRB_IntParam_OutputFlag,0);

        auto num_locations = demand.data.size();
        std::vector<GRBVar> _e; // List[gp.Var]  # 2d
        std::vector<GRBVar> _f; // List[gp.Var]  # 2d
        std::vector<GRBVar> _s_O; // List[gp.Var]  # 2d

        const auto& P_O = route_pat;
        const auto& P_R = req_pat;
        const auto& lambda_R = demand;

        // Define decision variables, p_D, p_G, p_O, p_N --> a_i in paper, s_G, s_O are the slack variables
        for (auto i = 0; i < num_locations; ++i) {
            p_D.push_back(fluid_model.addVar(0, 1, 1.0, GRB_CONTINUOUS, fmt::format("p_D{}", i)));
            p_G.push_back(fluid_model.addVar(0, 1, 1.0, GRB_CONTINUOUS, fmt::format("p_G{}", i)));
            p_O.push_back(fluid_model.addVar(0, 1, 1.0, GRB_CONTINUOUS, fmt::format("p_O{}", i)));
            p_N.push_back(fluid_model.addVar(0, 1, 1.0, GRB_CONTINUOUS, fmt::format("p_N{}", i)));
            s_G.push_back(fluid_model.addVar(0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, fmt::format("s_G{}", i)));
            for (auto j = 0; j < num_locations; ++j) {
                _f.push_back(fluid_model.addVar(0, 1, 1.0, GRB_CONTINUOUS, fmt::format("p_G{}{}", i, j)));
                _e.push_back(fluid_model.addVar(0, 1, 1.0, GRB_CONTINUOUS, fmt::format("p_O{}{}", i, j)));
                _s_O.push_back(
                        fluid_model.addVar(0, GRB_INFINITY, 1.0, GRB_CONTINUOUS, fmt::format("s_O{}{}", i, j)));
            }
        }

        F = Array2D(num_locations, std::move(_f));
        E = Array2D(num_locations, std::move(_e));
        s_O = Array2D(num_locations, std::move(_s_O));

        // Constraints
        for (auto i = 0; i < num_locations; ++i) {
            // (3.4b)
            for (auto j = 0; j < num_locations; ++j) {
                cons_a.push_back(fluid_model.addConstr(lambda_R.get(i) * p_D[i] * P_R.get(i, j) ==
                                                       mu.get(i, j) * F.at(i, j), fmt::format("a_{}_{}", i, j)));

                // (3.4d)
                cons_c.push_back(fluid_model.addConstr(
                        lambda_R.get(i) * p_O.at(i) * P_R.get(i, j) + s_O.at(i, j) == 0, fmt::format("c_{}_{}", i, j)));
                // (3.4e)
                if (i != j) {
                    auto quicksum = GRBLinExpr();
                    for (auto k = 0; k < num_locations; ++k) { quicksum += mu.get(k, i) * F.at(k, i); }
                    fluid_model.addConstr(mu.get(i, j) * E.at(i, j) <= quicksum, fmt::format("d_{}_{}", i, j));
                }
            }
            // (3.4c)
            cons_b.push_back(fluid_model.addConstr(
                    lambda_R.get(i) * p_G.at(i)+ s_G.at(i) == 0, fmt::format("b_{}", i)));

            // (3.4f_left)
            if (i == 0) {
                auto quicksum = GRBLinExpr();
                for (auto k = i + 1; k < num_locations; ++k) { quicksum += mu.get(k, i) * E.at(k, i); }
                cons_elhs.push_back(
                        fluid_model.addConstr(lambda_R.get(i) * p_D[i] >= quicksum, fmt::format("elhs_{}", i)));
            } else if (i == num_locations - 1) {
                auto quicksum = GRBLinExpr();
                for (auto k = 0; k < i; ++k) { quicksum += mu.get(k, i) * E.at(k, i); }
                cons_elhs.push_back(
                        fluid_model.addConstr(lambda_R.get(i) * p_D[i] >= quicksum, fmt::format("elhs_{}", i)));
            } else {
                auto quicksum_k = GRBLinExpr();
                for (auto k = 0; k < i; ++k) { quicksum_k += mu.get(k, i) * E.at(k, i); }
                auto quicksum_j = GRBLinExpr();
                for (auto j = i + 1; j < num_locations; ++j) { quicksum_j += mu.get(j, i) * E.at(j, i); }

                cons_elhs.push_back(fluid_model.addConstr(lambda_R.get(i) * p_D[i] >= quicksum_k + quicksum_j,
                                                          fmt::format("elhs_{}", i)));
            }

            // (3.4f_right)
            if (i == 0) {
                auto quicksum_r1 = GRBLinExpr();
                for (auto k = i + 1; k < num_locations; ++k) { quicksum_r1 += mu.get(k, i) * E.at(k, i); }
                auto quicksum_r2 = GRBLinExpr();
                for (auto j = 0; j < num_locations; ++j) { quicksum_r1 += mu.get(j, i) * F.at(j, i); }

                cons_erhs.push_back(fluid_model.addConstr(lambda_R.get(i) * p_D[i] <= quicksum_r1 + quicksum_r2,
                                                          fmt::format("erhs_{}", i)));
            } else if (i == num_locations - 1) {
                auto quicksum_r1 = GRBLinExpr();
                for (auto k = 0; k < i; ++k) { quicksum_r1 += mu.get(k, i) * E.at(k, i); }
                auto quicksum_r2 = GRBLinExpr();
                for (auto j = 0; j < num_locations; ++j) { quicksum_r1 += mu.get(j, i) * F.at(j, i); }

                cons_erhs.push_back(fluid_model.addConstr(lambda_R.get(i) * p_D[i] <= quicksum_r1 + quicksum_r2,
                                                          fmt::format("erhs_{}", i)));

            } else {
                auto quicksum_r1 = GRBLinExpr();
                for (auto k = 0; k < i; ++k) { quicksum_r1 += mu.get(k, i) * E.at(k, i); }
                auto quicksum_r2 = GRBLinExpr();
                for (auto j = i + 1; j < num_locations; ++j) { quicksum_r2 += mu.get(j, i) * E.at(j, i); }
                auto quicksum_r3 = GRBLinExpr();
                for (auto l = 0; l < num_locations; ++l) { quicksum_r3 += mu.get(l, i) * F.at(l, i); }

                cons_erhs.push_back(fluid_model.addConstr(
                        lambda_R.get(i) * p_D[i] <= quicksum_r1 + quicksum_r2 + quicksum_r3,
                        fmt::format("erhs_{}", i)));
            }

            // (3.4g)
            if (i == 0) {
                auto quicksum_l1 = GRBLinExpr();
                for (auto l = i + 1; l < num_locations; ++l) { quicksum_l1 += mu.get(i, l) * E.at(i, l); }

                auto quicksum_r1 = GRBLinExpr();
                for (auto k = i + 1; k < num_locations; ++k) { quicksum_r1 += mu.get(k, i) * E.at(k, i); }
                auto quicksum_r2 = GRBLinExpr();
                for (auto j = 0; j < num_locations; ++j) { quicksum_r2 += mu.get(j, i) * F.at(j, i); }

                cons_f.push_back(fluid_model.addConstr(
                        lambda_R.get(i) * p_D[i] + quicksum_l1 == quicksum_r1 + quicksum_r2,
                        fmt::format("f_{}", i)));
            } else if (i == num_locations - 1) {
                auto quicksum_l1 = GRBLinExpr();
                for (auto l = 0; l < i; ++l) { quicksum_l1 += mu.get(i, l) * E.at(i, l); }

                auto quicksum_r1 = GRBLinExpr();
                for (auto k = 0; k < i; ++k) { quicksum_r1 += mu.get(k, i) * E.at(k, i); }
                auto quicksum_r2 = GRBLinExpr();
                for (auto j = 0; j < num_locations; ++j) { quicksum_r2 += mu.get(j, i) * F.at(j, i); }

                cons_f.push_back(fluid_model.addConstr(
                        lambda_R.get(i) * p_D[i] + quicksum_l1 == quicksum_r1 + quicksum_r2,
                        fmt::format("f_{}", i)));

            } else {
                auto quicksum_l1 = GRBLinExpr();
                for (auto l = 0; l < i; ++l) { quicksum_l1 += mu.get(i, l) * E.at(i, l); }
                auto quicksum_l2 = GRBLinExpr();
                for (auto l = i + 1; l < num_locations; ++l) { quicksum_l2 += mu.get(i, l) * E.at(i, l); }

                auto quicksum_r1 = GRBLinExpr();
                for (auto k = 0; k < i; ++k) { quicksum_r1 += mu.get(k, i) * E.at(k, i); }
                auto quicksum_r2 = GRBLinExpr();
                for (auto j = i + 1; j < num_locations; ++j) { quicksum_r2 += mu.get(j, i) * E.at(j, i); }
                auto quicksum_r3 = GRBLinExpr();
                for (auto l = 0; l < num_locations; ++l) { quicksum_r3 += mu.get(l, i) * F.at(l, i); }

                cons_f.push_back(fluid_model.addConstr(
                        lambda_R.get(i) * p_D[i] + quicksum_l1 + quicksum_l2 == quicksum_r1 + quicksum_r2 + quicksum_r3,
                        fmt::format("f_{}", i)));
            }

            // (3.4h)
            fluid_model.addConstr(p_D[i] + p_G[i] + p_O[i] + p_N[i] == 1);
        }

           // (3.4l)
        auto quicksum_ij = GRBLinExpr();
        for (auto i = 0; i < num_locations; ++i) {
            for (auto j = 0; j < num_locations; ++j) {
                quicksum_ij += E.at(i,j) + F.at(i,j);
            }
        }
        fluid_model.addConstr(quicksum_ij == 1);

        // (3.4a)
        auto objective = GRBLinExpr();
        for (auto i = 0; i < num_locations; ++i) {
            for (auto j = 0; j < num_locations; ++j) {
                objective += E.at(i,j) * c_empty_routing.at(i,j) * 1 +
                        c1.at(i,j,0) * P_R.get(i,j) * lambda_R.get(i) * p_D[i] +
                        c1.at(i,j,1) * P_R.get(i,j) * lambda_R.get(i) * p_G[i] +
                        c1.at(i,j,2) * P_R.get(i,j) * lambda_R.get(i) * p_O[i] +
                        c_pen_req * lambda_R.get(i) * P_R.get(i,j) * p_N[i];
            }
        }

        fluid_model.setObjective(objective, GRB_MAXIMIZE);

        fluid_model.update();
        fluid_model.optimize();
    }

    // This function optimizes the operational level model given a current fleet composition (_fd,_cd1,_cd2)
    auto solve_model(double _fd, int _cd1, int _cd2) {
        fd = std::max(_fd, 0.000001);
        cd1 = _cd1;
        cd2 = _cd2;

        auto num_locations = demand.data.size();
        auto idx = [&](auto i, auto j)->auto{ return i * num_locations + j; };
        

        // Update lambda_g
        auto& P_O = route_pat;
        auto& P_R = req_pat;
        auto& lambda_R = demand;
        auto sum_demand = std::accumulate(demand.data.begin(), demand.data.end(), 0.0);
        auto lambda_G = std::vector<double>();
        for (double i : gw_pat.data) {
            lambda_G.push_back(cd1 * gw_capacity * i);
        }

        // Update lambda_o
        auto lambda_O = std::vector<double>();
        for (double i : mob_pat.data) {
            lambda_O.push_back(cd2 * od_capacity * i);
        }

        // Update constraints 3.4b, 3.4c, 3.4d
        for (auto i = 0; i < num_locations; ++i) {
            fluid_model.chgCoeff(cons_elhs[i], p_D[i], (lambda_R.get(i) / fd));
            fluid_model.chgCoeff(cons_erhs[i], p_D[i], (lambda_R.get(i) / fd));
            fluid_model.chgCoeff(cons_f[i], p_D[i], (lambda_R.get(i) / fd));
            for (auto j = 0; j < num_locations; ++j) {
                fluid_model.chgCoeff(cons_a[idx(i,j)], p_D[i], (lambda_R.get(i)/fd)*P_R.get(i,j));
                cons_c[idx(i, j)].set(GRB_DoubleAttr_RHS, lambda_O[i] * P_O.get(i,j));
            }
            cons_b[i].set(GRB_DoubleAttr_RHS, lambda_G[i]);
        }

        // Update objective function
        for (auto i = 0; i < num_locations; ++i) {
            for (auto j = 0; j < num_locations; ++j) {
                E.at(i,j).set(GRB_DoubleAttr_Obj, c_empty_routing.at(i,j) * fd);
            }
        }

        fluid_model.update();
        fluid_model.optimize();

        auto c_fd = 0.0;
        auto c_cd1 = 0.0;
        auto c_cd2 = 0.0;
        auto c_pen = 0.0;

        auto requests_delivered_by_FDs = 0.0;
        auto requests_delivered_by_GWs = 0.0;
        auto requests_delivered_by_ODs = 0.0;
        auto requests_not_delivered = 0.0;
        auto costs_empty_routing = 0.0;
        auto request_routes_FDs = std::vector<double>();
        auto request_routes_GWs = std::vector<double>();
        auto request_routes_ODs = std::vector<double>();
        auto request_routes_not_matched = std::vector<double>();

        auto sum_s_G = 0.0;
        auto sum_s_O = 0.0;

        // Store relevant result values
        for (auto i = 0; i < num_locations; ++i) {
            for (auto j = 0; j < num_locations; ++j) {
                c_fd += E.at(i, j).get(GRB_DoubleAttr_X) * c_empty_routing.at(i, j) * fd + c1.at(i, j, 0) * P_R.get(i, j) * lambda_R.get(i) * p_D[i].get(GRB_DoubleAttr_X);
                c_cd1 += c1.at(i,j,1) * P_R.get(i, j) * lambda_R.get(i) * p_G[i].get(GRB_DoubleAttr_X);
                c_cd2 += c1.at(i,j,2) * P_R.get(i, j) * lambda_R.get(i) * p_O[i].get(GRB_DoubleAttr_X);
                c_pen += c_pen_req * lambda_R.get(i) * P_R.get(i, j) * p_N[i].get(GRB_DoubleAttr_X);
                requests_delivered_by_FDs += P_R.get(i, j) * lambda_R.get(i) * p_D[i].get(GRB_DoubleAttr_X);
                requests_delivered_by_GWs += P_R.get(i, j) * lambda_R.get(i) * p_G[i].get(GRB_DoubleAttr_X);
                requests_delivered_by_ODs += P_R.get(i, j) * lambda_R.get(i) * p_O[i].get(GRB_DoubleAttr_X);
                requests_not_delivered += P_R.get(i, j) * lambda_R.get(i) * p_N[i].get(GRB_DoubleAttr_X);
                costs_empty_routing += E.at(i, j).get(GRB_DoubleAttr_X) * c_empty_routing.at(i, j) * fd;
                sum_s_O += s_O.at(i,j).get(GRB_DoubleAttr_X);
                request_routes_FDs.push_back(P_R.get(i, j) * lambda_R.get(i) * p_D[i].get(GRB_DoubleAttr_X));
                request_routes_GWs.push_back(P_R.get(i, j) * lambda_R.get(i) * p_G[i].get(GRB_DoubleAttr_X));
                request_routes_ODs.push_back(P_R.get(i, j) * lambda_R.get(i) * p_O[i].get(GRB_DoubleAttr_X));
                request_routes_not_matched.push_back(P_R.get(i, j) * lambda_R.get(i) * p_N[i].get(GRB_DoubleAttr_X));
            }
            sum_s_G += s_G[i].get(GRB_DoubleAttr_X);
        }

        auto operational_utility = fluid_model.getObjective().getValue();

        // Calculate share of CDs not matched
        auto s_g = 0.0;
        if (cd1 != 0) {
            s_g = sum_s_G / cd1;
        }
        auto s_o = 0.0;
        if (cd2 != 0) {
            s_o = sum_s_O / cd2;
        }
        double total_utility = 0.0;
        total_utility = operational_utility-c_fix*fd;
        return SolutionValues {total_utility,operational_utility,s_g,s_o,c_fd,c_cd1,c_cd2,c_pen,requests_delivered_by_FDs,requests_delivered_by_GWs,requests_delivered_by_ODs,requests_not_delivered,costs_empty_routing,request_routes_FDs,request_routes_GWs,request_routes_ODs,request_routes_not_matched};
    }
};

#endif //CD_CPP_OPERATIONAL_MODEL_HPP
