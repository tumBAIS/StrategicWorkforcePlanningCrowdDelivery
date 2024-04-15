#ifndef _MSC_VER
#include <sys/time.h>
#endif
#include <chrono>
#include "npy.hpp"
#include "spdlog/spdlog.h"
#include "utils_costs.hpp"
#include "operational_model_costs.hpp"
//#include "operational_model_cd.hpp"
// #include "operational_model_costs_old.hpp"
#include "gurobi_c++.h"
#include <numeric>
#include "fmt/format.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;
PYBIND11_MODULE(costs_cpp, handle) {
    handle.doc() = "This is the total cost calculation module"; // optional module docstring
    py::class_<GRBEnv>(handle,"GRBEnv").def(py::init<>());
    py::class_<NPYArray1D>(handle,"NPYArray1D").def(py::init<std::vector<double>>()).def("at",&NPYArray1D::at).def("get",&NPYArray1D::get);
    py::class_<NPYArray2D>(handle,"NPYArray2D").def(py::init<std::vector<unsigned long>,std::vector<double>>()).def("at",&NPYArray2D::at).def("get",&NPYArray2D::get);
    py::class_<NPYArray3D>(handle,"NPYArray3D").def(py::init<std::vector<unsigned long>,std::vector<double>>()).def("at",&NPYArray3D::at);
//    py::class_<OperationalModel>(handle,"OperationalModel").def(py::init<GRBEnv,NPYArray3D, NPYArray2D, double, double, NPYArray2D,  NPYArray2D,  NPYArray2D,
//        NPYArray1D, NPYArray1D,double,double,NPYArray1D>()).def("solve_model",&OperationalModel::solve_model);
    py::class_<OperationalModel>(handle,"OperationalModel").def(py::init<GRBEnv*,NPYArray3D&, NPYArray2D&, double, double, const NPYArray2D&, const  NPYArray2D&, const  NPYArray2D&,
        const NPYArray1D&, const NPYArray1D&,double,double,const NPYArray1D&>()).def("solve_model",&OperationalModel::solve_model);
    py::class_<SolutionValues>(handle,"SolutionValues")
    .def_readwrite("total_utility",&SolutionValues::total_utility)
    .def_readwrite("operational_utility",&SolutionValues::operational_utility)
    .def_readwrite("s_g",&SolutionValues::s_g)
    .def_readwrite("s_o",&SolutionValues::s_o)
    .def_readwrite("c_fd",&SolutionValues::c_fd)
    .def_readwrite("c_cd1",&SolutionValues::c_cd1)
    .def_readwrite("c_cd2",&SolutionValues::c_cd2)
    .def_readwrite("c_pen",&SolutionValues::c_pen)
    .def_readwrite("requests_delivered_by_FDs",&SolutionValues::requests_delivered_by_FDs)
    .def_readwrite("requests_delivered_by_GWs",&SolutionValues::requests_delivered_by_GWs)
    .def_readwrite("requests_delivered_by_ODs",&SolutionValues::requests_delivered_by_ODs)
    .def_readwrite("requests_not_delivered",&SolutionValues::requests_not_delivered)
    .def_readwrite("costs_empty_routing",&SolutionValues::costs_empty_routing)
    .def_readwrite("request_routes_FDs",&SolutionValues::request_routes_FDs)
    .def_readwrite("request_routes_GWs",&SolutionValues::request_routes_GWs)
    .def_readwrite("request_routes_ODs",&SolutionValues::request_routes_ODs)
    .def_readwrite("request_routes_not_matched",&SolutionValues::request_routes_not_matched);
}

