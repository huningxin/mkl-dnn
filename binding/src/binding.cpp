#include <emscripten/bind.h>

#include "mkldnn.h"

using namespace emscripten;

#define LOG_STATUS(s) do {\
    printf("[%s:%d] error: %s returns %d\n", __FILE__, __LINE__, __func__, s); \
} while (0)

namespace utils {
    int dimsFromJSArray(mkldnn_dims_t& dims, val js_array) {
        std::vector<int> vec = vecFromJSArray<int>(js_array);
        assert(vec.size() < TENSOR_MAX_DIMS);
        for (int i = 0; i < vec.size(); ++i) {
            dims[i] = vec[i];
        }
        return vec.size();
    }

    intptr_t mkldnn_engine_create_helper(int kind, size_t index) {
        mkldnn_engine_t aengine;
        mkldnn_status_t status = mkldnn_engine_create(&aengine, (mkldnn_engine_kind_t)kind, index);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)aengine;
    }

    int mkldnn_engine_destroy_helper(intptr_t engine) {
        return mkldnn_engine_destroy((mkldnn_engine_t)engine);
    }

    intptr_t mkldnn_memory_desc_create_helper(val js_dims,
                                       int data_type, int format) {
        mkldnn_memory_desc_t* memory_desc = new mkldnn_memory_desc_t;
        mkldnn_dims_t dims;
        int ndims = dimsFromJSArray(dims, js_dims);
        mkldnn_status_t status = mkldnn_memory_desc_init(
            memory_desc, ndims, dims, (mkldnn_data_type_t)data_type, (mkldnn_memory_format_t)format);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            delete memory_desc;
            return 0;
        }
        return (intptr_t)memory_desc;
    }

    int mkldnn_memory_desc_destroy_helper(intptr_t memory_desc) {
        delete (mkldnn_memory_desc_t*)memory_desc;
        return mkldnn_success;
    }

    intptr_t mkldnn_memory_primitive_desc_create_helper(intptr_t memory_desc, intptr_t engine) {
        mkldnn_primitive_desc_t primitive_desc;
        mkldnn_status_t status = mkldnn_memory_primitive_desc_create(
            &primitive_desc, (const mkldnn_memory_desc_t*)memory_desc, (mkldnn_engine_t)engine);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)primitive_desc;
    }

    int mkldnn_primitive_desc_destroy_helper(intptr_t primitive_desc) {
        return mkldnn_primitive_desc_destroy((mkldnn_primitive_desc_t)primitive_desc);
    }

    intptr_t mkldnn_primitive_create_helper(intptr_t primitive_desc, val js_inputs, val js_outputs) {
        mkldnn_primitive_t primitive;
        std::vector<intptr_t> inputs_vec = vecFromJSArray<intptr_t>(js_inputs);
        std::vector<mkldnn_primitive_at_t> inputs;
        for (int i = 0; i < inputs_vec.size(); ++i) {
            inputs.push_back(mkldnn_primitive_at((const_mkldnn_primitive_t)inputs_vec[i], 0));
        }
        std::vector<const_mkldnn_primitive_t> outputs;
        std::vector<intptr_t> outputs_vec = vecFromJSArray<intptr_t>(js_outputs);
        for (int i = 0; i < outputs_vec.size(); ++i) {
            outputs.push_back((const_mkldnn_primitive_t)outputs_vec[i]);
        }
        mkldnn_status_t status = mkldnn_primitive_create(
            &primitive, (const_mkldnn_primitive_desc_t)primitive_desc,
            inputs.empty() ? NULL : inputs.data(),
            outputs.empty() ? NULL : outputs.data());
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)primitive;
    }

    int mkldnn_primitive_destroy_helper(intptr_t primitive) {
        return mkldnn_primitive_destroy((mkldnn_primitive_t)primitive);
    }

    intptr_t mkldnn_memory_get_data_handle_helper(intptr_t memory) {
        void* handle;
        mkldnn_status_t status = mkldnn_memory_get_data_handle((const_mkldnn_primitive_t)memory, &handle);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)handle;
    }

    int mkldnn_memory_set_data_handle_helper(intptr_t memory, intptr_t handle) {
        return mkldnn_memory_set_data_handle((mkldnn_primitive_t)memory, (void*)handle);
    }

    intptr_t mkldnn_convolution_forward_desc_create_helper(
            int prop_kind, int alg_kind, intptr_t src_desc, intptr_t weights_desc, intptr_t bias_desc,
            intptr_t dst_desc, val js_strides, val js_padding_l, val js_padding_r, int padding_kind) {
       mkldnn_convolution_desc_t* conv_desc = new mkldnn_convolution_desc_t;
       mkldnn_dims_t strides, padding_l, padding_r;
       dimsFromJSArray(strides, js_strides);
       dimsFromJSArray(padding_l, js_padding_l);
       dimsFromJSArray(padding_r, js_padding_r);
       mkldnn_status_t status = mkldnn_convolution_forward_desc_init(
           conv_desc, (mkldnn_prop_kind_t)prop_kind, (mkldnn_alg_kind_t)alg_kind,
           (const mkldnn_memory_desc_t*)src_desc, (const mkldnn_memory_desc_t*)weights_desc,
           (const mkldnn_memory_desc_t*)bias_desc, (const mkldnn_memory_desc_t*)dst_desc,
           strides, padding_l, padding_r, (mkldnn_padding_kind_t)padding_kind);
       if (status != mkldnn_success) {
           LOG_STATUS(status);
           delete conv_desc;
           return 0;
       }
       return (intptr_t)conv_desc;
    }

    int mkldnn_convolution_forward_desc_destory_helper(intptr_t conv_desc) {
        delete (mkldnn_convolution_desc_t*)conv_desc;
        return mkldnn_success;
    }

    intptr_t mkldnn_primitive_desc_create_helper(intptr_t op_desc, intptr_t engine, intptr_t hint_forward_primitive_desc) {
        mkldnn_primitive_desc_t primitive_desc;
        mkldnn_status_t status = mkldnn_primitive_desc_create(
            &primitive_desc, (const_mkldnn_op_desc_t)op_desc, (mkldnn_engine_t)engine,
            (const_mkldnn_primitive_desc_t)hint_forward_primitive_desc);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)primitive_desc;
    }

    intptr_t mkldnn_primitive_desc_query_pd_helper(intptr_t primitive_desc, int what, int index) {
        return (intptr_t)mkldnn_primitive_desc_query_pd((const_mkldnn_primitive_desc_t)primitive_desc, (mkldnn_query_t)what, index);
    }

    intptr_t mkldnn_primitive_get_primitive_desc_helper(intptr_t primitive) {
        const_mkldnn_primitive_desc_t primitive_desc;
        mkldnn_status_t status = mkldnn_primitive_get_primitive_desc((const_mkldnn_primitive_t)primitive, &primitive_desc);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)primitive_desc;
    }

    bool mkldnn_memory_primitive_desc_equal_helper(intptr_t lhs, intptr_t rhs) {
        return mkldnn_memory_primitive_desc_equal((const_mkldnn_primitive_desc_t)lhs, (const_mkldnn_primitive_desc_t)rhs) == 1;
    }

    intptr_t mkldnn_reorder_primitive_desc_create_helper(intptr_t input, intptr_t output) {
        mkldnn_primitive_desc_t reorder_primitive_desc;
        mkldnn_status_t status = mkldnn_reorder_primitive_desc_create(
            &reorder_primitive_desc, (const_mkldnn_primitive_desc_t)input, (const_mkldnn_primitive_desc_t)output);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)reorder_primitive_desc;
    }

    intptr_t mkldnn_stream_create_helper(int stream_kind) {
        mkldnn_stream_t stream;
        mkldnn_status_t status = mkldnn_stream_create(&stream, (mkldnn_stream_kind_t)stream_kind);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)stream;
    }

    int mkldnn_stream_submit_helper(intptr_t stream, val js_primitives) {
        std::vector<int> primitives_vec = vecFromJSArray<int>(js_primitives);
        size_t n = primitives_vec.size();
        mkldnn_primitive_t primitives[n];
        for (int i = 0; i < n; ++i) {
            primitives[i] = (mkldnn_primitive_t)primitives_vec[i];
        }
        return (int)mkldnn_stream_submit((mkldnn_stream_t)stream, n, primitives, NULL);
    }

    int mkldnn_stream_wait_helper(intptr_t stream, int block) {
        return (int)mkldnn_stream_wait((mkldnn_stream_t)stream, block, NULL);
    }

    int mkldnn_stream_destroy_helper(intptr_t stream) {
        return (int)mkldnn_stream_destroy((mkldnn_stream_t)stream);
    }

    intptr_t mkldnn_primitive_desc_query_memory_d_helper(intptr_t primitive_desc) {
        return (intptr_t)mkldnn_primitive_desc_query_memory_d((const_mkldnn_primitive_desc_t)primitive_desc);
    }

    intptr_t mkldnn_eltwise_forward_desc_create_helper(int prop_kind, int alg_kind, intptr_t data_desc, float alpha, float beta) {
        mkldnn_eltwise_desc_t* eltwise_desc = new mkldnn_eltwise_desc_t;
        mkldnn_status_t status = mkldnn_eltwise_forward_desc_init(eltwise_desc, (mkldnn_prop_kind_t)prop_kind,
            (mkldnn_alg_kind_t)alg_kind, (const mkldnn_memory_desc_t*)data_desc, alpha, beta);
        if (status != mkldnn_success) {
            LOG_STATUS(status);
            return 0;
        }
        return (intptr_t)eltwise_desc;
    }
}

EMSCRIPTEN_BINDINGS(mkldnn)
{
    // mkldnn_engine_kind_t
    constant("mkldnn_any_engine", (int)mkldnn_any_engine);
    constant("mkldnn_cpu", (int)mkldnn_cpu);

    // mkldnn_data_type_t
    constant("mkldnn_data_type_undef", (int)mkldnn_data_type_undef);
    constant("mkldnn_f32", (int)mkldnn_f32);
    constant("mkldnn_s32", (int)mkldnn_s32);
    constant("mkldnn_s16", (int)mkldnn_s16);
    constant("mkldnn_s8", (int)mkldnn_s8);
    constant("mkldnn_u8", (int)mkldnn_u8);

    // mkldnn_memory_format_t
    constant("mkldnn_format_undef", (int)mkldnn_format_undef);
    constant("mkldnn_any", (int)mkldnn_any);
    constant("mkldnn_blocked", (int)mkldnn_blocked);
    constant("mkldnn_x", (int)mkldnn_x);
    constant("mkldnn_nc", (int)mkldnn_nc);
    constant("mkldnn_nchw", (int)mkldnn_nchw);
    constant("mkldnn_nhwc", (int)mkldnn_nhwc);
    constant("mkldnn_chwn", (int)mkldnn_chwn);
    constant("mkldnn_oi", (int)mkldnn_oi);
    constant("mkldnn_io", (int)mkldnn_io);
    constant("mkldnn_oihw", (int)mkldnn_oihw);
    constant("mkldnn_ihwo", (int)mkldnn_ihwo);
    constant("mkldnn_hwio", (int)mkldnn_hwio);
    // TODO: more enums of mkldnn_memory_format_t

    // mkldnn_prop_kind_t
    constant("mkldnn_prop_kind_undef", (int)mkldnn_prop_kind_undef);
    constant("mkldnn_forward_training", (int)mkldnn_forward_training);
    constant("mkldnn_forward_inference", (int)mkldnn_forward_inference);
    constant("mkldnn_forward_scoring", (int)mkldnn_forward_scoring);
    constant("mkldnn_forward", (int)mkldnn_forward);
    constant("mkldnn_backward", (int)mkldnn_backward);
    constant("mkldnn_backward_data", (int)mkldnn_backward_data);
    constant("mkldnn_backward_weights", (int)mkldnn_backward_weights);
    constant("mkldnn_backward_bias", (int)mkldnn_backward_bias);

    // mkldnn_alg_kind_t
    constant("mkldnn_convolution_direct", (int)mkldnn_convolution_direct);
    constant("mkldnn_convolution_winograd", (int)mkldnn_convolution_winograd);
    constant("mkldnn_eltwise_relu", (int)mkldnn_eltwise_relu);
    constant("mkldnn_eltwise_tanh", (int)mkldnn_eltwise_tanh);
    constant("mkldnn_eltwise_elu", (int)mkldnn_eltwise_elu);
    constant("mkldnn_eltwise_square", (int)mkldnn_eltwise_square);
    constant("mkldnn_eltwise_abs", (int)mkldnn_eltwise_abs);
    constant("mkldnn_eltwise_sqrt", (int)mkldnn_eltwise_sqrt);
    constant("mkldnn_eltwise_linear", (int)mkldnn_eltwise_linear);
    constant("mkldnn_eltwise_bounded_relu", (int)mkldnn_eltwise_bounded_relu);
    constant("mkldnn_eltwise_soft_relu", (int)mkldnn_eltwise_soft_relu);
    constant("mkldnn_eltwise_logistic", (int)mkldnn_eltwise_logistic);
    constant("mkldnn_pooling_max", (int)mkldnn_pooling_max);
    constant("mkldnn_pooling_avg_include_padding", (int)mkldnn_pooling_avg_include_padding);
    constant("mkldnn_pooling_avg_exclude_padding", (int)mkldnn_pooling_avg_exclude_padding);
    constant("mkldnn_pooling_avg", (int)mkldnn_pooling_avg);
    constant("mkldnn_lrn_across_channels", (int)mkldnn_lrn_across_channels);
    constant("mkldnn_lrn_within_channel", (int)mkldnn_lrn_within_channel);

    // mkldnn_padding_kind_t
    constant("mkldnn_padding_zero", (int)mkldnn_padding_zero);

    // mkldnn_query_t
    constant("mkldnn_query_undef", (int)mkldnn_query_undef);
    constant("mkldnn_query_engine", (int)mkldnn_query_engine);
    constant("mkldnn_query_primitive_kind", (int)mkldnn_query_primitive_kind);
    constant("mkldnn_query_num_of_inputs_s32", (int)mkldnn_query_num_of_inputs_s32);
    constant("mkldnn_query_num_of_outputs_s32", (int)mkldnn_query_num_of_outputs_s32);
    constant("mkldnn_query_time_estimate_f64", (int)mkldnn_query_time_estimate_f64);
    constant("mkldnn_query_memory_consumption_s64", (int)mkldnn_query_memory_consumption_s64);
    constant("mkldnn_query_impl_info_str", (int)mkldnn_query_impl_info_str);
    constant("mkldnn_query_some_d", (int)mkldnn_query_some_d);
    constant("mkldnn_query_memory_d", (int)mkldnn_query_memory_d);
    constant("mkldnn_query_convolution_d", (int)mkldnn_query_convolution_d);
    constant("mkldnn_query_eltwise_d", (int)mkldnn_query_eltwise_d);
    constant("mkldnn_query_relu_d", (int)mkldnn_query_relu_d);
    constant("mkldnn_query_softmax_d", (int)mkldnn_query_softmax_d);
    constant("mkldnn_query_pooling_d", (int)mkldnn_query_pooling_d);
    constant("mkldnn_query_lrn_d", (int)mkldnn_query_lrn_d);
    constant("mkldnn_query_batch_normalization_d", (int)mkldnn_query_batch_normalization_d);
    constant("mkldnn_query_inner_product_d", (int)mkldnn_query_inner_product_d);
    constant("mkldnn_query_convolution_relu_d", (int)mkldnn_query_convolution_relu_d);
    constant("mkldnn_query_some_pd", (int)mkldnn_query_some_pd);
    constant("mkldnn_query_input_pd", (int)mkldnn_query_input_pd);
    constant("mkldnn_query_output_pd", (int)mkldnn_query_output_pd);
    constant("mkldnn_query_src_pd", (int)mkldnn_query_src_pd);
    constant("mkldnn_query_diff_src_pd", (int)mkldnn_query_diff_src_pd);
    constant("mkldnn_query_weights_pd", (int)mkldnn_query_weights_pd);
    constant("mkldnn_query_diff_weights_pd", (int)mkldnn_query_diff_weights_pd);
    constant("mkldnn_query_dst_pd", (int)mkldnn_query_dst_pd);
    constant("mkldnn_query_diff_dst_pd", (int)mkldnn_query_diff_dst_pd);
    constant("mkldnn_query_workspace_pd", (int)mkldnn_query_workspace_pd);

    // mkldnn_stream_kind_t
    constant("mkldnn_any_stream", (int)mkldnn_any_stream);
    constant("mkldnn_eager", (int)mkldnn_eager);
    constant("mkldnn_lazy", (int)mkldnn_lazy);

    // functions
    function("mkldnn_engine_create", &utils::mkldnn_engine_create_helper, allow_raw_pointers());
    function("mkldnn_engine_destroy", &utils::mkldnn_engine_destroy_helper, allow_raw_pointers());
    function("mkldnn_memory_desc_create", &utils::mkldnn_memory_desc_create_helper, allow_raw_pointers());
    function("mkldnn_memory_desc_destroy", &utils::mkldnn_memory_desc_destroy_helper, allow_raw_pointers());
    function("mkldnn_memory_primitive_desc_create", &utils::mkldnn_memory_primitive_desc_create_helper, allow_raw_pointers());
    function("mkldnn_primitive_desc_destroy", &utils::mkldnn_primitive_desc_destroy_helper, allow_raw_pointers());
    function("mkldnn_primitive_create", &utils::mkldnn_primitive_create_helper, allow_raw_pointers());
    function("mkldnn_primitive_destroy", &utils::mkldnn_primitive_destroy_helper, allow_raw_pointers());
    function("mkldnn_memory_get_data_handle", &utils::mkldnn_memory_get_data_handle_helper, allow_raw_pointers());
    function("mkldnn_memory_set_data_handle", &utils::mkldnn_memory_set_data_handle_helper, allow_raw_pointers());
    function("mkldnn_convolution_forward_desc_create", &utils::mkldnn_convolution_forward_desc_create_helper, allow_raw_pointers());
    function("mkldnn_convolution_forward_desc_destroy", &utils::mkldnn_convolution_forward_desc_destory_helper, allow_raw_pointers());
    function("mkldnn_primitive_desc_create", &utils::mkldnn_primitive_desc_create_helper, allow_raw_pointers());
    function("mkldnn_primitive_desc_query_pd", &utils::mkldnn_primitive_desc_query_pd_helper, allow_raw_pointers());
    function("mkldnn_primitive_get_primitive_desc", &utils::mkldnn_primitive_get_primitive_desc_helper, allow_raw_pointers());
    function("mkldnn_memory_primitive_desc_equal", &utils::mkldnn_memory_primitive_desc_equal_helper, allow_raw_pointers());
    function("mkldnn_reorder_primitive_desc_create", &utils::mkldnn_reorder_primitive_desc_create_helper, allow_raw_pointers());
    function("mkldnn_stream_create", &utils::mkldnn_stream_create_helper, allow_raw_pointers());
    function("mkldnn_stream_submit", &utils::mkldnn_stream_submit_helper, allow_raw_pointers());
    function("mkldnn_stream_wait", &utils::mkldnn_stream_wait_helper, allow_raw_pointers());
    function("mkldnn_stream_destroy", &utils::mkldnn_stream_destroy_helper, allow_raw_pointers());
    function("mkldnn_primitive_desc_query_memory_d", &utils::mkldnn_primitive_desc_query_memory_d_helper, allow_raw_pointers());
    function("mkldnn_eltwise_forward_desc_create", &utils::mkldnn_eltwise_forward_desc_create_helper, allow_raw_pointers());
}