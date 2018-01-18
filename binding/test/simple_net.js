var mkldnn;

function prepare_reorder(user_memory, prim_memory_pd, dir_is_user_to_prim, buffer) {
  let user_memory_pd = mkldnn.mkldnn_primitive_get_primitive_desc(user_memory);

  let prim_memory = 0;
  let reorder = 0;
  let equal = mkldnn.mkldnn_memory_primitive_desc_equal(user_memory_pd, prim_memory_pd);
  if (!equal) {
    prim_memory = mkldnn.mkldnn_primitive_create(prim_memory_pd, [], []);
    let reorder_pd;
    if (dir_is_user_to_prim) {
      reorder_pd = mkldnn.mkldnn_reorder_primitive_desc_create(user_memory_pd, prim_memory_pd);
      reorder = mkldnn.mkldnn_primitive_create(reorder_pd, [user_memory], [prim_memory]);
    } else {
      reorder_pd = mkldnn.mkldnn_reorder_primitive_desc_create(prim_memory_pd, user_memory_pd);
      reorder = mkldnn.mkldnn_primitive_create(reorder_pd, [prim_memory], [user_memory]);
    }
    mkldnn.mkldnn_memory_set_data_handle(prim_memory, buffer);
    mkldnn.mkldnn_primitive_desc_destroy(reorder_pd);
  }

  return [prim_memory, reorder];
}

function init_data_memory(dims, user_fmt, data_type, engine, data) {
  let prim_md = mkldnn.mkldnn_memory_desc_create(dims, data_type, user_fmt);
  let user_pd = mkldnn.mkldnn_memory_primitive_desc_create(prim_md, engine);
  let memory = mkldnn.mkldnn_primitive_create(user_pd, [], []);

  let req = mkldnn.mkldnn_memory_get_data_handle(memory);
  mkldnn.mkldnn_memory_set_data_handle(memory, data);
  req = mkldnn.mkldnn_memory_get_data_handle(memory);
  mkldnn.mkldnn_primitive_desc_destroy(user_pd);
  mkldnn.mkldnn_memory_desc_destroy(prim_md);

  return memory;
}

function product(array) {
  return array.reduce((accumulator, currentValue) => accumulator * currentValue);
}

function main() {
  mkldnn = Module;
  
  const BATCH = 1;

  let engine = mkldnn.mkldnn_engine_create(mkldnn.mkldnn_cpu, 0);
  console.log(`engine: ${engine}`);

  let net_src = mkldnn._malloc(BATCH*3*227*227*Float32Array.BYTES_PER_ELEMENT);
  let net_dst = mkldnn._malloc(BATCH*96*27*27*Float32Array.BYTES_PER_ELEMENT);

  /* AlexNet: conv
   * {BATCH, 3, 227, 227} (x) {96, 3, 11, 11} -> {BATCH, 96, 55, 55}
   * strides: {4, 4}
   */
  let conv_src_sizes = [BATCH, 3, 227, 227];
  let conv_weights_sizes = [96, 3, 11, 11];
  let conv_bias_sizes = [96];
  let conv_dst_sizes = [BATCH, 96, 55, 55];
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
  let conv_internal_dst_memory_pd = mkldnn.mkldnn_primitive_desc_query_pd(conv_pd, mkldnn.mkldnn_query_dst_pd, 0);
  let conv_internal_dst_memory = mkldnn.mkldnn_primitive_create(conv_internal_dst_memory_pd, [], []);
  
  mkldnn.mkldnn_memory_set_data_handle(conv_internal_dst_memory, conv_dst_buffer);

  /* create reorder primitives between user data and convolution srcs
   * if required */
  let src_pd = mkldnn.mkldnn_primitive_desc_query_pd(conv_pd, mkldnn.mkldnn_query_src_pd, 0);
  [conv_internal_src_memory, conv_reorder_src] = prepare_reorder(conv_user_src_memory, src_pd, 1, conv_src_buffer);
  let weights_pd = mkldnn.mkldnn_primitive_desc_query_pd(conv_pd, mkldnn.mkldnn_query_weights_pd, 0);
  [conv_internal_weights_memory, conv_reorder_weights] = prepare_reorder(conv_user_weights_memory, weights_pd, 1, conv_weights_buffer);

  let conv_src_memory = conv_internal_src_memory ? conv_internal_src_memory : conv_user_src_memory;
  let conv_weights_memory = conv_internal_weights_memory ? conv_internal_weights_memory : conv_user_weights_memory;

  /* finally create a convolution primitive */
  let conv = mkldnn.mkldnn_primitive_create(conv_pd, [conv_src_memory, conv_weights_memory, conv_user_bias_memory], [conv_internal_dst_memory]);

  /* AlexNet: relu
   * {BATCH, 96, 55, 55} -> {BATCH, 96, 55, 55}
   */
  let negative_slope = 1.0;
  
  let relu_dst_sizes = conv_dst_sizes;
  let relu_dst_buffer = mkldnn._malloc(product(relu_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);
  mkldnn._memset(relu_dst_buffer, 0, product(relu_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);

  /* create relu memory descriptor on dst memory descriptor
   * from previos primitive */
  let conv_dst_pd = mkldnn.mkldnn_primitive_desc_query_pd(conv_pd, mkldnn.mkldnn_query_dst_pd, 0);
  let relu_src_md = mkldnn.mkldnn_primitive_desc_query_memory_d(conv_dst_pd);

  /* create a relu */
  let relu_desc = mkldnn.mkldnn_eltwise_forward_desc_create(mkldnn.mkldnn_forward, mkldnn.mkldnn_eltwise_relu, relu_src_md, negative_slope, 0);

  let relu_pd = mkldnn.mkldnn_primitive_desc_create(relu_desc, engine, 0);

  let relu_dst_pd = mkldnn.mkldnn_primitive_desc_query_pd(relu_pd, mkldnn.mkldnn_query_dst_pd, 0);
  let relu_dst_memory = mkldnn.mkldnn_primitive_create(relu_dst_pd, [], []);
  mkldnn.mkldnn_memory_set_data_handle(relu_dst_memory, relu_dst_buffer);

  /* finally create a relu primitive */
  let relu = mkldnn.mkldnn_primitive_create(relu_pd, [conv_internal_dst_memory], [relu_dst_memory]);

  /* AlexNet: lrn
   * {BATCH, 96, 55, 55} -> {BATCH, 96, 55, 55}
   * local size: 5
   * alpha: 0.0001
   * beta: 0.75
   */
  let local_size = 5;
  let alpha = 0.0001;
  let beta = 0.75;
  let k = 1.0;

  let lrn_dst_sizes = relu_dst_sizes;

  let lrn_dst_buffer = mkldnn._malloc(product(lrn_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);
  mkldnn._memset(lrn_dst_buffer, 0, product(lrn_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);

  /* create lrn memory descriptor on dst memory descriptor
   *  from previos primitive */
  let lrn_src_md = mkldnn.mkldnn_primitive_desc_query_memory_d(relu_dst_pd);

  /* create a lrn */
  let lrn_desc = mkldnn.mkldnn_lrn_forward_desc_create(
    mkldnn.mkldnn_forward, mkldnn.mkldnn_lrn_across_channels, lrn_src_md, local_size, alpha, beta, k);

  let lrn_pd = mkldnn.mkldnn_primitive_desc_create(lrn_desc, engine, 0);

  let lrn_dst_pd = mkldnn.mkldnn_primitive_desc_query_pd(lrn_pd, mkldnn.mkldnn_query_dst_pd, 0);
  let lrn_dst_memory = mkldnn.mkldnn_primitive_create(lrn_dst_pd, [], []);
  mkldnn.mkldnn_memory_set_data_handle(lrn_dst_memory, lrn_dst_buffer);

  let lrn_scratch_pd = mkldnn.mkldnn_primitive_desc_query_pd(lrn_pd, mkldnn.mkldnn_query_workspace_pd, 0);
  let lrn_scratch_memory = mkldnn.mkldnn_primitive_create(lrn_scratch_pd, [], []);
  let lrn_scratch_size = mkldnn.mkldnn_memory_primitive_desc_get_size(lrn_scratch_pd);
  let lrn_scratch_buffer = mkldnn._malloc(lrn_scratch_size);
  mkldnn._memset(lrn_scratch_buffer, 0, lrn_scratch_size);
  mkldnn.mkldnn_memory_set_data_handle(lrn_scratch_memory, lrn_scratch_buffer);

  /* finally create a lrn primitive */
  let lrn = mkldnn.mkldnn_primitive_create(lrn_pd, [relu_dst_memory], [lrn_dst_memory, lrn_scratch_memory]);

  /* AlexNet: pool
   * {BATCH, 96, 55, 55} -> {BATCH, 96, 27, 27}
   * kernel: {3, 3}
   * strides: {2, 2}
   */
  let pool_dst_sizes = [BATCH, 96, 27, 27];
  let pool_kernel = [3, 3];
  let pool_strides = [2, 2];
  let pool_padding = [0, 0];

  let pool_dst_buffer = mkldnn._malloc(product(pool_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);
  mkldnn._memset(pool_dst_buffer, 0, product(pool_dst_sizes)*Float32Array.BYTES_PER_ELEMENT);

  /* create pooling memory descriptor on dst descriptor
   *  from previos primitive */
  let pool_src_md = mkldnn.mkldnn_primitive_desc_query_memory_d(lrn_dst_pd);

  /* create descriptors for dst pooling data */
  let pool_dst_md = mkldnn.mkldnn_memory_desc_create(pool_dst_sizes, mkldnn.mkldnn_f32, mkldnn.mkldnn_any);

  /* create memory for user data */
  let pool_user_dst_memory = init_data_memory(pool_dst_sizes, mkldnn.mkldnn_nchw, mkldnn.mkldnn_f32, engine, net_dst);

  /* create a pooling */
  let pool_desc = mkldnn.mkldnn_pooling_forward_desc_create(
      mkldnn.mkldnn_forward, mkldnn.mkldnn_pooling_max, pool_src_md, pool_dst_md, pool_strides,
      pool_kernel, pool_padding, pool_padding, mkldnn.mkldnn_padding_zero);

  let pool_pd = mkldnn.mkldnn_primitive_desc_create(pool_desc, engine, 0);

  /* create memory for workspace */
  let pool_indices_pd = mkldnn.mkldnn_primitive_desc_query_pd(pool_pd, mkldnn.mkldnn_query_workspace_pd, 0);
  let pool_indices_memory = mkldnn.mkldnn_primitive_create(pool_indices_pd, [], []);
  let pool_indices_size = mkldnn.mkldnn_memory_primitive_desc_get_size(pool_indices_pd);
  let pool_indices_buffer = mkldnn._malloc(pool_indices_size);
  mkldnn._memset(pool_indices_buffer, 0, pool_indices_size);
  mkldnn.mkldnn_memory_set_data_handle(pool_indices_memory, pool_indices_buffer);

  /* create reorder primitives between user data and pooling dsts
   * if required */
  let pool_dst_pd = mkldnn.mkldnn_primitive_desc_query_pd(pool_pd, mkldnn.mkldnn_query_dst_pd, 0);
  [pool_internal_dst_memory, pool_reorder_dst] = prepare_reorder(pool_user_dst_memory, pool_dst_pd, 0, pool_dst_buffer);

  let pool_dst_memory = pool_internal_dst_memory ? pool_internal_dst_memory : pool_user_dst_memory;

  /* finally create a pooling primitive */
  let pool = mkldnn.mkldnn_primitive_create(pool_pd, [lrn_dst_memory], [pool_dst_memory, pool_indices_memory]);

  /* build a simple net */
  let net = [conv, relu, lrn, pool];
  let stream = mkldnn.mkldnn_stream_create(mkldnn.mkldnn_eager);
  console.log(`stream: ${stream}`);
  let start = performance.now();
  mkldnn.mkldnn_stream_submit(stream, net);
  let status = mkldnn.mkldnn_stream_wait(stream, 1);
  console.log(`mkldnn_stream_wait: ${status}`);
  let end = performance.now();
  console.log(`elapsed time: ${end - start} ms`);

  /* clean-up */
  mkldnn.mkldnn_primitive_desc_destroy(conv_pd);

  mkldnn.mkldnn_stream_destroy(stream);
  
  mkldnn._free(net_src);
  mkldnn._free(net_dst);

  mkldnn.mkldnn_primitive_destroy(conv_user_src_memory);
  mkldnn.mkldnn_primitive_destroy(conv_user_weights_memory);
  mkldnn.mkldnn_primitive_destroy(conv_user_bias_memory);
  mkldnn.mkldnn_primitive_destroy(conv_internal_src_memory);
  mkldnn.mkldnn_primitive_destroy(conv_internal_weights_memory);
  mkldnn.mkldnn_primitive_destroy(conv_internal_dst_memory);
  mkldnn.mkldnn_primitive_destroy(conv_reorder_src);
  mkldnn.mkldnn_primitive_destroy(conv_reorder_weights);
  mkldnn.mkldnn_primitive_destroy(conv);

  mkldnn._free(conv_weights);
  mkldnn._free(conv_bias);

  mkldnn._free(conv_src_buffer);
  mkldnn._free(conv_weights_buffer);
  mkldnn._free(conv_dst_buffer);

  mkldnn.mkldnn_primitive_destroy(relu_dst_memory);
  mkldnn.mkldnn_primitive_destroy(relu);

  mkldnn._free(relu_dst_buffer);

  mkldnn.mkldnn_primitive_destroy(lrn_scratch_memory);
  mkldnn.mkldnn_primitive_destroy(lrn_dst_memory);
  mkldnn.mkldnn_primitive_destroy(lrn);

  mkldnn._free(lrn_scratch_buffer);
  mkldnn._free(lrn_dst_buffer);

  mkldnn.mkldnn_engine_destroy(engine);
}