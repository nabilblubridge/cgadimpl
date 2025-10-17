cmake_minimum_required(VERSION 3.20)
project(cgadimpl LANGUAGES CXX CUDA)

# ===========================================================
# --- 1. Common settings
# ===========================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

option(AG_PACKAGING    "Enable install + find_package exports" OFF)
option(AG_GLOB_SOURCES "Glob all .cpp/.cu under src/ (simplifies dev)" ON)
option(AG_BUILD_TESTS  "Build tests in tests/" ON)

# ===========================================================
# --- 2. Source collection
# ===========================================================
if(AG_GLOB_SOURCES)
    file(GLOB_RECURSE CGADIMPL_SRC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    )
else()
    set(CGADIMPL_SRC
        src/core/graph.cpp
        src/core/ops.cpp
        src/core/schema.cpp
        src/core/debug.cpp
        src/core/export_hlo.cpp
        src/core/nodeops.cpp
        src/autodiff/autodiff.cpp
        src/autodiff/autodiff_vjp_ops.cpp
        src/autodiff/autodiff_jvp_ops.cpp
        src/tensor/tensor.cpp
        src/nn/nn.cpp
        src/optimizer/optim.cpp
        src/tools/kernels_loader.cpp
        src/core/cudarray.cu
    )
endif()

if(NOT CGADIMPL_SRC)
    message(FATAL_ERROR "No source files found under ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

# Explicitly mark .cu files as CUDA sources (important when globbing)
foreach(cu_file ${CGADIMPL_SRC})
    if(cu_file MATCHES "\\.cu$")
        set_source_files_properties(${cu_file} PROPERTIES LANGUAGE CUDA)
    endif()
endforeach()

# ===========================================================
# --- 3. Sanity check
# ===========================================================
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/ad/kernels_api.hpp")
    message(FATAL_ERROR "Missing header: include/ad/kernels_api.hpp")
endif()

# ===========================================================
# --- 4. Library definition
# ===========================================================
add_library(cgadimpl SHARED ${CGADIMPL_SRC})
add_library(cgadimpl::cgadimpl ALIAS cgadimpl)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cgadimpl PRIVATE CUDA::cudart)


set_target_properties(cgadimpl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cgadimpl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(cgadimpl PRIVATE /W4 /permissive-)
else()
    target_compile_options(cgadimpl PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(cgadimpl PRIVATE dl)
endif()

# ===========================================================
# --- 5. Tests (optional)
# ===========================================================
if(AG_BUILD_TESTS)
    include(CTest)
    enable_testing()

    function(add_ag_test name src)
        if("${name}" STREQUAL "test")
            set(name "ag_core_test")
        endif()
        add_executable(${name} ${src})
        target_link_libraries(${name} PRIVATE cgadimpl)
        if(UNIX AND NOT APPLE)
            target_link_libraries(${name} PRIVATE dl)
        endif()
        add_test(NAME ${name} COMMAND ${name})
    endfunction()

    add_ag_test(test_ag            tests/test_ag.cpp)
    add_ag_test(test_mlp           tests/test_mlp.cpp)
    add_ag_test(test_complex_mlp   tests/test_complex_mlp.cpp)
    target_compile_definitions(test_complex_mlp PRIVATE AG_EXPOSE_AUTODIFF_RULES)
    add_ag_test(test_graph_compile tests/test_graph_compile.cpp)
    add_ag_test(ag_core_test       tests/test.cpp)
    add_ag_test(test_checkpoint    tests/test_checkpoint.cpp)
endif()

# ===========================================================
# --- 6. Packaging (optional)
# ===========================================================
if(AG_PACKAGING)
    include(GNUInstallDirs)

    install(TARGETS cgadimpl
        EXPORT cgadimplTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
        VERSION 0.1.0
        COMPATIBILITY SameMajorVersion
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in")
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cgadimplConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake" @ONLY
        )
        install(EXPORT cgadimplTargets
            NAMESPACE cgadimpl::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl
        )
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/cgadimplConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cgadimpl
        )
    endif()
endif()

# ===========================================================
# --- 7. Status output
# ===========================================================
message(STATUS "cgadimpl build mode: ${CMAKE_BUILD_TYPE}")
message(STATUS "AG_PACKAGING: ${AG_PACKAGING}")
message(STATUS "AG_GLOB_SOURCES: ${AG_GLOB_SOURCES}")
message(STATUS "AG_BUILD_TESTS: ${AG_BUILD_TESTS}")
