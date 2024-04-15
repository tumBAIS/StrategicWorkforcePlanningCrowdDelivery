cmake_minimum_required(VERSION 3.16)
option(CalcAllRewards "CalcAllRewards" TRUE)
option(utility_costs "utility_costs" TRUE)
if (CalcAllRewards)
	project(cd_cpp)
else ()
	if (utility_costs)
		project(costs_cpp)
	else ()
		project(profits_cpp)
	endif()
endif()
#SET(CONAN_EXTRA_REQUIRES benchmark/1.5.0)
include(cmake/Conan.cmake)
run_conan()

include(cmake/FindGUROBI.cmake)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

# enable flto for non-debug builds
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

# Include pybind11
find_package(pybind11 REQUIRED)

if (MSVC)
else ()
    ## Flags
    set(CMAKE_CXX_COMPILER "/usr/bin/g++")
    #set (CMAKE_CXX_COMPILER "/usr/bin/icpc")
    #set (CMAKE_LINKER "/usr/bin/xild")
    #set (CMAKE_AR "/usr/bin/xiar")
endif ()

if (CalcAllRewards)
	file(GLOB_RECURSE SOURCES "src/cd_cpp.cpp")
	#create executable for pre-calculating all rewards 
	add_executable(${PROJECT_NAME} ${SOURCES})
else ()
	if (utility_costs)
		file(GLOB_RECURSE SOURCES "src/costs_cpp.cpp")
	else ()
		file(GLOB_RECURSE SOURCES "src/profits_cpp.cpp")
	endif()
	# create python module based on c++ code that implements the fluid model
	pybind11_add_module(${PROJECT_NAME} ${SOURCES})
endif()

if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:/O2 /DNDEBUG>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:/Od >)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RelWithDebInfo>:/O2>)
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Release>:-O3 -flto -frename-registers -march=skylake -DNDEBUG -s>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:-O0 -ggdb -march=native>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -flto -frename-registers -ggdb -march=native>)

    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-flto")
endif ()

target_include_directories(${PROJECT_NAME} PRIVATE include)
target_include_directories(${PROJECT_NAME} PRIVATE lib/libnpy/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${GUROBI_INCLUDE_DIRS})
# target_include_directories(${PROJECT_NAME} PRIVATE /opt/gurobi912/linux64/include)
target_link_libraries(${PROJECT_NAME} PRIVATE project_options)
target_link_libraries(${PROJECT_NAME} PRIVATE CONAN_PKG::spdlog)
target_link_libraries(${PROJECT_NAME} PRIVATE ${GUROBI_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY} optimized ${GUROBI_CXX_LIBRARY})
#target_link_libraries(${PROJECT_NAME} PRIVATE /opt/gurobi912/linux64/lib/libgurobi91.so /opt/gurobi912/linux64/lib/libgurobi_c++.a)

if (MSVC)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()

# for later
#if(CONAN_CATCH2_ROOT_DEBUG)
#    include(${CONAN_CATCH2_ROOT_DEBUG}/lib/cmake/Catch2/Catch.cmake)
#else()
#    include(${CONAN_CATCH2_ROOT}/lib/cmake/Catch2/Catch.cmake)
#endif()

#add_subdirectory(catch2_tests)
#add_subdirectory(benchmarks)
