cmake_minimum_required(VERSION 3.8)
project(cpu)

find_package(Torch REQUIRED)

# Add source to this project's executable
add_executable(cpu "local_product_cpu.cpp")
target_link_libraries(cpu "${TORCH_LIBRARIES}")

set_property(TARGET cpu PROPERTY CXX_STANDARD 14)

