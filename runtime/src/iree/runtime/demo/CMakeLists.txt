# NOTE: not using bazel-to-cmake here because of the runtime unified rule.
# We should figure out how to make bazel/cmake consistent with that.

iree_cc_binary(
  NAME
    hello_world_file
  SRCS
    "hello_world_explained.c"
  DEFINES
    "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_COMMAND_LINE_ARG"
  DEPS
    iree::runtime::unified
)

if(NOT IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE OR NOT IREE_TARGET_BACKEND_VMVX)
  return()
endif()

iree_cc_binary(
  NAME
    hello_world_embedded
  SRCS
    "hello_world_explained.c"
  DEFINES
    "IREE_RUNTIME_DEMO_LOAD_FILE_FROM_EMBEDDED_DATA"
  DEPS
    iree::runtime::unified
    ::simple_mul_module_c
)

iree_native_test(
  NAME
    "hello_world_embedded_test"
  SRC
    ::hello_world_embedded
)

iree_cc_binary(
  NAME
    hello_world_terse
  SRCS
    "hello_world_terse.c"
  DEPS
    iree::runtime::unified
    ::simple_mul_module_c
)

iree_cc_binary(
  NAME
    hello_world_two_devices
  SRCS
    "hello_world_two_devices.c"
  DEPS
    iree::runtime::unified
    ::simple_mul_module_c
    ::simple_mul_module_hip_c
)

iree_native_test(
  NAME
    "hello_world_terse_test"
  SRC
    ::hello_world_terse
)

iree_bytecode_module(
  NAME
    simple_mul_module
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_runtime_demo_simple_mul_module"
  FLAGS
    "--iree-hal-target-backends=amd-aie"
    "--iree-amdaie-target-device=npu1_4col"
    "--iree-amdaie-lower-to-aie-pipeline=air"
    "--iree-amdaie-tile-pipeline=pad-pack"
    "--iree-amd-aie-peano-install-dir=c:/fortest/peano/llvm-aie"
    "--iree-amd-aie-install-dir=C:/develop/iree-two-devices/iree-install"
    "--iree-amd-aie-enable-chess=false"
    "--iree-amdaie-enable-packet-flow=false"
    "--iree-hal-dump-executable-files-to=c:/fortest/iree-outputSimple"
    "--iree-amdaie-device-hal=xrt"
    "--iree-hal-memoization=false"
    "--iree-hal-indirect-command-buffers=false"
    "--mlir-elide-resource-strings-if-larger=10"
    "--iree-amd-aie-show-invoked-commands"
  PUBLIC
)

iree_bytecode_module(
  NAME
    simple_mul_module_hip
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_runtime_demo_simple_mul_module_hip"
  FLAGS
    "--iree-hal-target-device=hip"
    "--iree-hip-target=gfx1103"
  PUBLIC
)