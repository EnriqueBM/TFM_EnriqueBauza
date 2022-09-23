set(CMAKE_CXX_COMPILER "dpcpp")

cmake_minimum_required (VERSION 3.16)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

project (Het_Imp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TBB REQUIRED)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

add_executable(${PROJECT_NAME} main.cpp BitString_Tree_Representation/tree_rep.cpp rand_funcs/rand_funcs.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/BitString_Tree_Representation ${CMAKE_SOURCE_DIR}/rand_funcs)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -g -fp-model=precise ) # -Werror)
target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)

if(NOT CMAKE_BUILD_TYPE STREQUAL Debug)
	target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()
