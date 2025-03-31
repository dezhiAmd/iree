# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(IREE_ENABLE_RUNTIME_TRACING)

message(STATUS "Enabling runtime tracing using the ${IREE_TRACING_PROVIDER} provider")
message(STATUS "Tracing mode set at ${IREE_TRACING_MODE}")

if(${IREE_TRACING_PROVIDER} STREQUAL "console")
cmake -G Ninja -B ../iree-build-trace -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo 
# -DIREE_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ 
# -DIREE_ENABLE_RUNTIME_TRACING=ON -DIREE_BUILD_TRACY=ON 
# -DIREE_ENABLE_LLD=ON -DIREE_BUILD_PYTHON_BINDINGS=ON -DPython3_EXECUTABLE="$(which python3)" 
# -DIREE_TARGET_BACKEND_CUDA=OFF -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_ROCM=ON 
# -DIREE_TRACING_PROVIDER=console
  message(STATUS "Use console")
  iree_cc_library(
    NAME
      provider
    HDRS
      "console.h"
    SRCS
      "console.c"
    DEPS
      iree::base::core_headers
      iree::base::internal::time
    DEFINES
      "IREE_TRACING_PROVIDER_H=\"iree/base/tracing/console.h\""
      "IREE_TRACING_MODE=${IREE_TRACING_MODE}"
    PUBLIC
  )
elseif(${IREE_TRACING_PROVIDER} STREQUAL "tracy")
  iree_cc_library(
    NAME
      provider
    HDRS
      "tracy.h"
    SRCS
      "tracy.cc"
    DEPS
      iree::base::core_headers
      Tracy::TracyClient
    DEFINES
      "IREE_TRACING_PROVIDER_H=\"iree/base/tracing/tracy.h\""
      "IREE_TRACING_MODE=${IREE_TRACING_MODE}"
    PUBLIC
  )
else()
  # Add the dep so that the user-provided provider is linked in.
  # The user will also need to have set a header to include via
  # IREE_TRACING_PROVIDER_H.
  if(NOT ${IREE_TRACING_PROVIDER_H})
    message(FATAL_ERROR "Custom tracing providers require a header file be set in IREE_TRACING_PROVIDER_H")
  endif()
  iree_cc_library(
    NAME
      provider
    DEPS
      ${IREE_TRACING_PROVIDER}
    DEFINES
      "IREE_TRACING_PROVIDER_H=\"${IREE_TRACING_PROVIDER_H}\""
      "IREE_TRACING_MODE=${IREE_TRACING_MODE}"
    PUBLIC
  )
endif()

else()
  # Empty rule for when no tracing provider is defined.
  message(STATUS "Runtime tracing disabled")
  iree_cc_library(
    NAME
      provider
    PUBLIC
  )

endif(IREE_ENABLE_RUNTIME_TRACING)
