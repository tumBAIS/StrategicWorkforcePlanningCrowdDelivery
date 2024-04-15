# Overview
This project implements all algorithms and functions described in the paper "Strategic Workforce Planning in Crowdsourced Delivery with Hybrid Driver Fleets" by Julius Luy, Gerhard Hiermann, and Maximilian Schiffer, see: https://arxiv.org/abs/2311.17935.

## Environment
The code is written in Python and C++. The C++ code is embedded into the Python project using pybind11 and cython. A requirements.txt provides details on the packages needed. We tested the code on Ubuntu 20.04 and a high-performance cluster whose details you find here: https://doku.lrz.de/linux-cluster-10333236.html.

## Project structure
The repository contains the following folders:
* src: Python scripts for the strategic level problem<br>
* cpp: C++ scripts for the operational level problem<br>
* data: Input data location<br>
* results: Output data location<br>
* build: Cython builds<br>

## Installation
 * Create a local Python environment in ./src by subsequently executing the following commands in the root folder
	* `python3 -m venv venv`
	* `source venv/bin/activate`
	* `python -m pip install -r requirements.txt`
	* `deactivate`
* Building the C++ files:
	* If you want to pre-calculate all rewards for all driver combinations (required for BDP), run in ./cpp from the command line (the binary will be located in ./cpp/cmake-build release/bin):
		* `cmake -S ./ -B ./cmake-build-release -DCMAKE_BUILD_TYPE=Release -DCalcAllRewards=TRUE`
		* `cmake --build ./cmake-build-release --target cd_cpp`
	* If you want to build a new operational model C++ based python module (required for PL-VFA), run in ./cpp from the command line (the module will be located in ./cpp/cmake-build-release/lib):
		* `cmake -S ./ -B ./cmake-build-release -DCMAKE_BUILD_TYPE=Release -DCalcAllRewards=FALSE`
		* `cmake --build ./cmake-build-release --target costs_cpp`
	* The resulting files will be located in ./cpp/cmake-build-release/lib and will called "cd_cpp.cpython-38-x86_64-linux-gnu.so" and "costs_cpp.cpython-38-x86_64-linux-gnu.so" respectively
	* Note that ./cpp/cmake-build-release/lib already contains a compiled "costs_cpp.cpython-38-x86_64-linux-gnu.so" corresponding to the operational model used in the paper
* Further Infos:
	* We used gcc to compile the C++ files
	* We use libnpy (https://github.com/llohse/libnpy)
	* We use cython version 0.29.25. A good explanation of how to create the cython builds, can be found here: https://www.youtube.com/watch?v=Ic1oE6SEOBs.

## Usage
* Create a .csv file that contains all the relevant input parameters. You find an example in the data folder. It corresponds to the base case described in the paper.
* You can set the value of the parameters in the column on the right-hand side of the .csv file
* Value range of non-numerical parameters in the .csv file: 
	* **demand_case**: "cagr_growth", "constant", "peak" -- different demand types, their form can be looked up in model_creation.py)
	* **discretization_technique**: "homogeneous", "inhomogeneous" -- whether the CD space should be homogeneously or inhomogeneously discretized, in the paper we have a homogeneous discretization)
	* **solution_approach**: "approx", "exact" -- PL-VFA or BDP
	* **rew**: "yes","no" -- whether rewards should be pre-calculated, only relevant for BDP
	* **mode**: "train", "test" -- whether the code should be called in train mode, i.e., updating the slopes, or in test mode, i.e., not updating the slopes
	* **random_od_patterns**: "yes","no" -- wether the OD patterns should be sampled randomly (according to seed) or wether they should be set equal to the demand_patterns
	* **load_existing_costs**: "yes", "no" -- whether existing costs should be loaded from an existing rew.npy file
	* **load_data**: "yes","no" -- whether one wants to load data or randomly sample. The following data is required if using existing data
		* Demand data: demand_arrival_patterns.npy (normalized $\lambda_i^{R}$) and req_pat.npy ($P_{ij}^R$)
		* Geographical data: od_matrix.npy (origin-destination matrix of the area)
		* GW patterns are automatically set equal to demand patterns
	* **save_data**: "yes", "no" -- whether output data should be saved
	* **load_existing_slopes**: "yes","no" -- whether the PL-VFA algorithm should start the learning process with pre-trained slopes
* The data folder contains the base case, where demand and geographical data is loaded from existing files corresponding to the details in Appendix F and OD patterns are randomly 				sampled based on the same seed as in the paper
* Execute the following in the root folder to run the code
	* `source venv/bin/activate`
	* `python3 main.py <input csv location> <mode>`
	


