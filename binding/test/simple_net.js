var mkldnn;

function main() {
  mkldnn = Module;
  
  let engine = mkldnn.mkldnn_engine_create(mkldnn.mkldnn_cpu, 0);
  console.log(`engine: ${engine}`);

  let net_src = mkldnn._malloc(1*3*227*227*Float32Array.BYTES_PER_ELEMENT);
  let net_dst = mkldnn._malloc(1*96*27*27*Float32Array.BYTES_PER_ELEMENT);

  /* AlexNet: conv
  * {BATCH, 3, 227, 227} (x) {96, 3, 11, 11} -> {BATCH, 96, 55, 55}
  * strides: {4, 4}
  */
  let conv_src_sizes = [1, 3, 227, 227];
  let conv_weights_sizes = [96, 3, 11, 11];
  let conv_bias_sizes = [96];
  let conv_dst_sizes = [1, 96, 55, 55];
  let conv_strides = [4, 4];
  let conv_padding = [0, 0];

  let conv_src = net_src;
  let conv_weights = mkldnn._malloc(product(conv_weights_sizes)*Float32Array.BYTES_PER_ELEMENT);
  let conv_bias = mkldnn._malloc(product(conv_bias_sizes)*Float32Array.BYTES_PER_ELEMENT);

  /* create memory for user data */
  let conv_user_src_memory = init_data_memory(
      conv_src_sizes, mkldnn.mkldnn_nchw, mkldnn.mkldnn_f32, engine, conv_src);
  let conv_user_weights_memory = init_data_memory(
      conv_weights_sizes, mkldnn.mkldnn_oihw, mkldnn.mkldnn_f32, engine, conv_weights);
  let conv_user_bias_memory = init_data_memory(
    conv_bias_sizes, mkldnn.mkldnn_x, mkldnn.mkldnn_f32, engine, conv_bias);

  /* create data descriptors for convolution w/ no specified format */
  let conv_src_md = mkldnn.mkldnn_memory_desc_create(conv_src_sizes, mkldnn.mkldnn_f32, mkldnn.mkldnn_any);
  let conv_weights_md = mkldnn.mkldnn_memory_desc_create(conv_weights_sizes, mkldnn.mkldnn_f32, mkldnn.mkldnn_any);
  let conv_bias_md = mkldnn.mkldnn_memory_desc_create(conv_bias_sizes, mkldnn.mkldnn_f32, mkldnn.mkldnn_any);
  let conv_dst_md = mkldnn.mkldnn_memory_desc_create(conv_dst_sizes, mkldnn.mkldnn_f32, mkldnn.mkldnn_any);

  /* create a convolution */
  let conv_any_desc = mkldnn.mkldnn_convolution_forward_desc_create(
      mkldnn.mkldnn_forward, mkldnn.mkldnn_convolution_direct, conv_src_md,
      conv_weights_md, conv_bias_md, conv_dst_md, conv_strides, conv_padding,
      conv_padding, mkldnn.mkldnn_padding_zero);

  let conv_pd = mkldnn.mkldnn_primitive_desc_create(conv_any_desc, engine, 0);

  let conv_src_buffer = mkldnn._malloc(product(conv_src_sizes)*Float32Array.BYTES_PER_ELEMENT);
  let conv_weights_buffer = mkldnn._malloc(product(conv_weights_sizes)*Float32Array.BYTES_PER_ELEMENT);
  let conv_dst_buffer = mkldnn._malloc(product(conv_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);
  mkldnn._memset(conv_src_buffer, 0, product(conv_src_sizes)*Float32Array.BYTES_PER_ELEMENT);
  mkldnn._memset(conv_weights_buffer, 0, product(conv_weights_sizes)*Float32Array.BYTES_PER_ELEMENT);
  mkldnn._memset(conv_dst_buffer, 0, product(conv_src_sizes)*Float32Array.BYTES_PER_ELEMENT);

  /* create memory for dst data, we don't need reorder it to user data */
  let conv_internal_dst_memory_md = mkldnn.mkldnn_primitive_desc_query_pd(conv_pd, mkldnn.mkldnn_query_dst_pd, 0);
  let conv_internal_dst_memory = mkldnn.mkldnn_primitive_create(conv_internal_dst_memory_md, [], []);
  
  mkldnn.mkldnn_memory_set_data_handle(conv_internal_dst_memory, conv_dst_buffer);

  /* create reorder primitives between user data and convolution srcs
     * if required */
  
}

function init_data_memory(dims, user_fmt, data_type, engine, data) {
  let prim_md = mkldnn.mkldnn_memory_desc_create(dims, data_type, user_fmt);
  let user_pd = mkldnn.mkldnn_memory_primitive_desc_create(prim_md, engine);
  let memory = mkldnn.mkldnn_primitive_create(user_pd, [], []);

  let req = mkldnn.mkldnn_memory_get_data_handle(memory);
  console.log(`req: ${req}`);
  mkldnn.mkldnn_memory_set_data_handle(memory, data);
  req = mkldnn.mkldnn_memory_get_data_handle(memory);
  console.log(`req: ${req}`);
  mkldnn.mkldnn_primitive_desc_destroy(user_pd);
  mkldnn.mkldnn_memory_desc_destroy(prim_md);

  return memory;
}

function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator + currentValue);
}
