// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"
#include "iree/runtime/demo/simple_mul_module_c.h"
#include "iree/runtime/demo/simple_mul_module_hip_c.h"

static void iree_runtime_demo_run_session(iree_runtime_instance_t* instance);
static void iree_runtime_demo_perform_mul_dual(iree_runtime_session_t* cpu_session, iree_runtime_session_t* hip_session);

//===----------------------------------------------------------------------===//
// 1. Entry point / shared iree_runtime_instance_t setup
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {

  // Create and configure the instance shared across all sessions.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;

  IREE_CHECK_OK(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));

  // All sessions should share the same instance.
  iree_runtime_demo_run_session(instance);

  iree_runtime_instance_release(instance);
  return 0;
}

//===----------------------------------------------------------------------===//
// 2. Load modules and initialize state in iree_runtime_session_t
//===----------------------------------------------------------------------===//

static void iree_runtime_demo_run_session(iree_runtime_instance_t* instance) {
  // TODO(#5724): move device selection into the compiled modules.

  // Create devices : cpu and hip
  // To create other non-default devices, 
  // get hal_driver from device_registry and 
  // use `iree_hal_driver_create_device_by_<ordinal/uri/path/id>` API
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("local-task"), &device));

  iree_hal_device_t* hip_device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("hip"), &hip_device));


  // Create one session per loaded module to hold the module state.

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));
  iree_hal_device_release(device);

  iree_runtime_session_options_t hip_session_options;
  iree_runtime_session_options_initialize(&hip_session_options);
  iree_runtime_session_t* hip_session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &hip_session_options, hip_device,
      iree_runtime_instance_host_allocator(instance), &hip_session));
  iree_hal_device_release(hip_device);


  // Load your user module into the session (from memory, from file, etc).
  const iree_file_toc_t* module_file =
      iree_runtime_demo_simple_mul_module_create();
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_memory(
      session, iree_make_const_byte_span(module_file->data, module_file->size),
      iree_allocator_null()));

  const iree_file_toc_t* hip_module_file =
      iree_runtime_demo_simple_mul_module_hip_create();
  IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_memory(
      hip_session, iree_make_const_byte_span(hip_module_file->data, hip_module_file->size),
      iree_allocator_null()));

  // Run your functions; you should reuse the session to make multiple calls.
  iree_runtime_demo_perform_mul_dual(session, hip_session);

  iree_runtime_session_release(session);
  iree_runtime_session_release(hip_session);
}

//===----------------------------------------------------------------------===//
// 3. Call a function within a module with buffer views
//===----------------------------------------------------------------------===//

// func.func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) ->
// tensor<4xf32>
static void iree_runtime_demo_perform_mul_dual(iree_runtime_session_t* cpu_session, iree_runtime_session_t* hip_session)
{
  iree_runtime_call_t call;
  IREE_CHECK_OK(iree_runtime_call_initialize_by_name(
      cpu_session, iree_make_cstring_view("module.simple_mul"), &call));

  iree_runtime_call_t hip_call;
  IREE_CHECK_OK(iree_runtime_call_initialize_by_name(
      hip_session, iree_make_cstring_view("module.simple_mul"), &hip_call));


  // CPU -> GPU
  // CPU execution (arg0, arg1) -> ret0

  // %arg0: tensor<4xf32> 
  fprintf(stdout, "\nCPU Exec Begin\n");

  iree_hal_buffer_view_t* arg0 = NULL;
  static const iree_hal_dim_t arg0_shape[1] = {4};
  static const float arg0_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(cpu_session),
      iree_runtime_session_device_allocator(cpu_session),
      IREE_ARRAYSIZE(arg0_shape), arg0_shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(arg0_data, sizeof(arg0_data)), &arg0));
  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, arg0, /*max_element_count=*/4096,
      iree_runtime_session_host_allocator(cpu_session)));
  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, arg0));
  iree_hal_buffer_view_release(arg0);

  fprintf(stdout, "\n * \n");

  // %arg1: tensor<4xf32>
  iree_hal_buffer_view_t* arg1 = NULL;
  static const iree_hal_dim_t arg1_shape[1] = {4};
  static const float arg1_data[4] = {10.0f, 100.0f, 1000.0f, 10000.0f};
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(cpu_session),
      iree_runtime_session_device_allocator(cpu_session),
      IREE_ARRAYSIZE(arg1_shape), arg1_shape,
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(arg1_data, sizeof(arg1_data)), &arg1));
  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, arg1, /*max_element_count=*/4096,
      iree_runtime_session_host_allocator(cpu_session)));
  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&call, arg1));
  iree_hal_buffer_view_release(arg1);

  IREE_CHECK_OK(iree_runtime_call_invoke(&call, /*flags=*/0));

  fprintf(stdout, "\n = \n");

  // -> tensor<4xf32>
  iree_hal_buffer_view_t* ret0_out = NULL;
  IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret0_out));

  
  IREE_CHECK_OK(iree_hal_buffer_view_fprint(
      stdout, ret0_out, /*max_element_count=*/4096,
      iree_runtime_session_host_allocator(cpu_session)));
  fprintf(stdout, "\nCPU Exec Done\n");

  iree_runtime_call_deinitialize(&call);


  // GPU Execution ret0, arg2 -> ret1

  fprintf(stdout, "\nGPU Exec Begin\n");
  // %ret0: tensor<4xf32>
  static const iree_hal_dim_t ret0_shape[1] = {4};
  float ret0_data[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
      iree_runtime_session_device(cpu_session),
      iree_hal_buffer_view_buffer(ret0_out),
      0, ret0_data, sizeof(ret0_data),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
      iree_infinite_timeout()));

  iree_hal_buffer_view_t* ret0 = NULL;
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(hip_session),
      iree_runtime_session_device_allocator(hip_session),
      IREE_ARRAYSIZE(ret0_shape), ret0_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(ret0_data, sizeof(ret0_data)), &ret0));


  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&hip_call, ret0));
  iree_hal_buffer_view_release(ret0);


  // %arg2: tensor<4xf32>
  iree_hal_buffer_view_t* arg2 = NULL;
  static const iree_hal_dim_t arg2_shape[1] = {4};
  static const float arg2_data[4] = {2000.0f, 200.0f, 20.0f, 2.0f};
  IREE_CHECK_OK(iree_hal_buffer_view_allocate_buffer_copy(
      iree_runtime_session_device(hip_session),
      iree_runtime_session_device_allocator(hip_session),
      IREE_ARRAYSIZE(arg2_shape), arg2_shape, 
      IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(arg2_data, sizeof(arg2_data)), &arg2));

  IREE_CHECK_OK(iree_runtime_call_inputs_push_back_buffer_view(&hip_call, arg2));
  iree_hal_buffer_view_release(arg2);

  IREE_CHECK_OK(iree_runtime_call_invoke(&hip_call, /*flags=*/0));

  // ret1 (tensor<4xf32>)
  iree_hal_buffer_view_t* ret1 = NULL;
  IREE_CHECK_OK(iree_runtime_call_outputs_pop_front_buffer_view(&hip_call, &ret1));

  // Read back result from device to host.
  float results[] = {0.0f, 0.0f, 0.0f, 0.0f};
  IREE_CHECK_OK(iree_hal_device_transfer_d2h(
        iree_runtime_session_device(hip_session),
        iree_hal_buffer_view_buffer(ret1), 0, results,
        sizeof(results), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

  iree_hal_buffer_view_release(ret1);
 
  // TODO : find a way to `iree_hal_buffer_view_fprint` a device buffer
  fprintf(stdout, "\nresult = [ "); 
  for (int i = 0; i < 4; ++i){
    fprintf(stdout, "%f, ", results[i]);
  }
  fprintf(stdout, " ]\n");
  fprintf(stdout, "\nGPU Exec Done\n");

  iree_runtime_call_deinitialize(&hip_call);

}