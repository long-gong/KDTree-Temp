cmake_minimum_required(VERSION 3.14)
project(bench_KDTree_Adaptors)

set(CMAKE_CXX_STANDARD 17)

set(BOOST_ROOT "C:\\Users\\Long Gong\\Dropbox\\vcpkg\\installed\\x86-windows")
find_package(Boost REQUIRED)
message("include: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})

macro(ADD_EXE _NAME _LIBS)
    add_executable(${_NAME} ${_NAME}.cc)
    target_link_libraries(${_NAME} PRIVATE ${_LIBS})
endmacro()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nanoflann)
find_package(benchmark CONFIG REQUIRED)


set(BENCHMARK_LIBRARIES  benchmark::benchmark benchmark::benchmark_main)


#ADD_EXE(bench_pointer_op "${BENCHMARK_LIBRARIES}")
ADD_EXE(kd_diff_data_types "")