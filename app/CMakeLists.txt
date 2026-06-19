cmake_minimum_required(VERSION 3.22.1)
project(EdgeMind)

# ── ExecuTorch JNI bridge ──────────────────────────────────────────────────
add_library(executorch_jni SHARED
    ${CMAKE_SOURCE_DIR}/../cpp/executorch_jni.cpp
)

target_link_libraries(executorch_jni
    android
    log
)

# Consume the ExecuTorch CMake package installed by
# executorch/scripts/build_android_library.sh (see executorch-config.cmake for
# the exported targets); this resolves the XNNPACK/kernel/quantized-ops static
# link graph for us instead of guessing raw .a link order.
set(_et_cmake_out "${CMAKE_SOURCE_DIR}/../executorch/cmake-out-android-${ANDROID_ABI}")
# Pin transitive package dirs so find_dependency() inside executorch's exported
# configs (tokenizers -> absl/re2) resolves to our cross-compiled tree instead
# of any mismatched abseil/re2 install elsewhere on the system search path.
set(absl_DIR "${_et_cmake_out}/lib/cmake/absl")
set(re2_DIR "${_et_cmake_out}/lib/cmake/re2")
set(tokenizers_DIR "${_et_cmake_out}/lib/cmake/tokenizers")
set(executorch_DIR "${_et_cmake_out}/lib/cmake/ExecuTorch")
find_package(executorch CONFIG)
if(EXECUTORCH_FOUND)
    target_include_directories(executorch_jni PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
    target_link_libraries(executorch_jni
        extension_module_static
        extension_tensor
        extension_data_loader
        extension_llm_runner
        optimized_native_cpu_ops_lib
        quantized_ops_lib
        custom_ops
        xnnpack_backend
    )
    target_compile_definitions(executorch_jni PRIVATE EXECUTORCH_AVAILABLE)
else()
    message(WARNING "ExecuTorch not found under executorch/cmake-out-android-${ANDROID_ABI} -- executorch_jni will build in stub mode")
endif()

# ── FAISS JNI bridge ──────────────────────────────────────────────────────
add_library(faiss_jni SHARED
    ${CMAKE_SOURCE_DIR}/../cpp/faiss_jni.cpp
)

target_include_directories(faiss_jni PRIVATE
    ${CMAKE_SOURCE_DIR}/../cpp/include
)

target_link_libraries(faiss_jni
    android
    log
)

# Link FAISS prebuilt .so if available
find_library(FAISS_LIB faiss HINTS ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI})
if(FAISS_LIB)
    target_link_libraries(faiss_jni ${FAISS_LIB})
endif()
