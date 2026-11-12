#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void direct_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool init_mask = asc_create_mask_b32(PAT_ALL);
  auto zero = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), init_mask, MODE_ZEROING);
  auto initial_m = simd_inst::vdup<float>(tl::limits::kFloatNInf, init_mask, MODE_ZEROING);
  simd_inst::vsts_norm(zero, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49536])), 0, init_mask);
  simd_inst::vsts_norm(initial_m, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49472])), 0, init_mask);
}

__simd_vf__ inline void direct_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool clear_mask = asc_create_mask_b32(PAT_ALL);
  auto output_zero = simd_inst::vdup<float>(float(0x0p+0f/*0.000000e+00*/), clear_mask, MODE_ZEROING);
  for (int32_t i = 0; i < 64; ++i) {
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      simd_inst::vsts_norm(output_zero, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 41088)])), 0, clear_mask);
    }
  }
}

__simd_vf__ inline void direct_kernel_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem, int32_t tile_1) {
  __ubuf__ void* dst_nz_ptr = nullptr;
  vector_bool softmax_full = asc_create_mask_b32(PAT_ALL);
  vector_bool softmax_one = asc_create_mask_b32(PAT_VL1);
  vector_bool softmax_b16 = asc_create_mask_b16(PAT_VL128);
  dst_nz_ptr = (&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((tile_1 & 1) * 8320) + 49152)]));
  simd_inst::vsts_norm(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49472])), 0), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49600])), 0, softmax_full);
  for (int32_t _tmp = 0; _tmp < 32; ++_tmp) {
    auto s00 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((tile_1 % 3) * 8192) + (_tmp * 256))])), 0), float(0x1.6a09e667f3bcdp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    auto s01 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp * 256)) + 64)])), 0), float(0x1.6a09e667f3bcdp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    auto s10 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp * 256)) + 128)])), 0), float(0x1.6a09e667f3bcdp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    auto s11 = simd_inst::vmuls(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp * 256)) + 192)])), 0), float(0x1.6a09e667f3bcdp-4f/*8.838835e-02*/), softmax_full, MODE_ZEROING);
    simd_inst::vsts_norm(s00, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((tile_1 % 3) * 8192) + (_tmp * 256))])), 0, softmax_full);
    simd_inst::vsts_norm(s01, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp * 256)) + 64)])), 0, softmax_full);
    simd_inst::vsts_norm(s10, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp * 256)) + 128)])), 0, softmax_full);
    simd_inst::vsts_norm(s11, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp * 256)) + 192)])), 0, softmax_full);
    auto max0 = simd_inst::vcmax(simd_inst::vmax(s00, s01, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    auto max1 = simd_inst::vcmax(simd_inst::vmax(s10, s11, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    simd_inst::vsts_1st(max0, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp * 2) + 49472)])), 0, softmax_one);
    simd_inst::vsts_1st(max1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp * 2) + 49473)])), 0, softmax_one);
  }
  simd_inst::mem_bar(VST_VLD);
  auto merged_max = simd_inst::vmax(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49472])), 0), simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49600])), 0), softmax_full, MODE_ZEROING);
  simd_inst::vsts_norm(merged_max, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49472])), 0, softmax_full);
  simd_inst::mem_bar(VST_VLD);
  for (int32_t _tmp_1 = 0; _tmp_1 < 32; ++_tmp_1) {
    auto max0_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 49472)])), 0);
    auto max1_1 = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 49473)])), 0);
    auto v = simd_inst::vld_x2<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((tile_1 % 3) * 8192) + (_tmp_1 * 256))])), DINTLV_B32);
    auto v_1 = simd_inst::vld_x2<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((tile_1 % 3) * 8192) + (_tmp_1 * 256)) + 128)])), DINTLV_B32);
    auto e00 = simd_inst::vexpdif(v.v0, max0_1, softmax_full);
    auto e10 = simd_inst::vexpdif(v_1.v0, max1_1, softmax_full);
    auto e01 = simd_inst::vexpdif(v.v1, max0_1, softmax_full);
    auto e11 = simd_inst::vexpdif(v_1.v1, max1_1, softmax_full);
    auto row_sum0 = simd_inst::vcadd(simd_inst::vadd(e00, e01, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    auto row_sum1 = simd_inst::vcadd(simd_inst::vadd(e10, e11, softmax_full, MODE_ZEROING), softmax_full, MODE_ZEROING);
    simd_inst::vsts_1st(row_sum0, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 49664)])), 0, softmax_one);
    simd_inst::vsts_1st(row_sum1, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((_tmp_1 * 2) + 49665)])), 0, softmax_one);
    auto even_bf16_0 = simd_inst::vcvt<bfloat16_t>(e00, softmax_full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
    auto even_bf16_1 = simd_inst::vcvt<bfloat16_t>(e10, softmax_full, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
    auto odd_bf16_0 = simd_inst::vcvt<bfloat16_t>(e01, softmax_full, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
    auto odd_bf16_1 = simd_inst::vcvt<bfloat16_t>(e11, softmax_full, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
    auto merged_bits0 = simd_inst::vor((*(vector_u16 *)(&(even_bf16_0))), (*(vector_u16 *)(&(odd_bf16_0))), softmax_b16, MODE_ZEROING);
    auto merged_bits1 = simd_inst::vor((*(vector_u16 *)(&(even_bf16_1))), (*(vector_u16 *)(&(odd_bf16_1))), softmax_b16, MODE_ZEROING);
    dst_nz_ptr = simd_inst::vsstb((*(vector_bf16 *)(&(merged_bits0))), (__ubuf__ bfloat16_t*)dst_nz_ptr, 4259841, softmax_b16, POST_UPDATE);
    dst_nz_ptr = simd_inst::vsstb((*(vector_bf16 *)(&(merged_bits1))), (__ubuf__ bfloat16_t*)dst_nz_ptr, 4259841, softmax_b16, POST_UPDATE);
  }
  simd_inst::mem_bar(VST_VLD);
  auto alpha_reg = simd_inst::vexpdif(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49600])), 0), simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49472])), 0), softmax_full);
  simd_inst::vsts_norm(alpha_reg, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((tile_1 % 3) * 64) + 49280)])), 0, softmax_full);
  auto l_scaled = simd_inst::vmul(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49536])), 0), alpha_reg, softmax_full, MODE_ZEROING);
  auto l_updated = simd_inst::vadd(l_scaled, simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49664])), 0), softmax_full, MODE_ZEROING);
  simd_inst::vsts_norm(l_updated, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[49536])), 0, softmax_full);
}

__simd_vf__ inline void direct_kernel_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem, int32_t tile_1) {
  vector_bool rescale_mask = asc_create_mask_b32(PAT_ALL);
  for (int32_t i = 0; i < 64; ++i) {
    auto alpha = simd_inst::vlds_brc_elem<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((((tile_1 + 1) % 3) * 64) + i) + 49280)])), 0);
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      auto prev = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 41088)])), 0);
      auto cur = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 32896)])), 0);
      auto updated = simd_inst::vadd(simd_inst::vmul(alpha, prev, rescale_mask, MODE_ZEROING), cur, rescale_mask, MODE_ZEROING);
      simd_inst::vsts_norm(updated, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((i * 128) + (_tmp * 64)) + 41088)])), 0, rescale_mask);
    }
  }
}

__simt_vf__ __launch_bounds__(128) inline void direct_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  #pragma unroll
  for (int32_t i = 0; i < 32; ++i) {
    float2 _;
      float2 v_ = *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 41088));
      float2 v__1 = make_float2(max(((__ubuf__ float*)buf_dyn_shmem)[(((i * 2) + (((int32_t)threadIdx.x) >> 6)) + 49536)], float(0x1.4484bfeebc2ap-100f/*1.000000e-30*/)), max(((__ubuf__ float*)buf_dyn_shmem)[(((i * 2) + (((int32_t)threadIdx.x) >> 6)) + 49536)], float(0x1.4484bfeebc2ap-100f/*1.000000e-30*/)));
      _.x = (v_.x/v__1.x);
      _.y = (v_.y/v__1.y);
    *(__ubuf__ float2*)(((__ubuf__ float*)buf_dyn_shmem) + (((i * 256) + (((int32_t)threadIdx.x) * 2)) + 41088)) = _;
  }
}

__simd_vf__ inline void direct_kernel_simd_vf_4(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool mask = asc_create_mask_b32(PAT_ALL);
  for (int32_t r = 0; r < 64; ++r) {
    for (int32_t _tmp = 0; _tmp < 2; ++_tmp) {
      auto value = simd_inst::vcvt<bfloat16_t>(simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((r * 128) + (_tmp * 64)) + 41088)])), 0), mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
      simd_inst::vsts_pack_b32(value, (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((r * 128) + (_tmp * 64)) + 82176)])), 0, mask);
    }
  }
}

extern "C" __global__ __mix__(1, 2) void direct_kernel(__gm__ bfloat16_t* K, __gm__ int32_t* Lengths, __gm__ bfloat16_t* O, __gm__ bfloat16_t* Q, __gm__ int32_t* Table, __gm__ bfloat16_t* V) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ uint8_t *buf_dyn_l0c = (__cc__ uint8_t *)0;
  __cb__ uint8_t *buf_dyn_l0b = (__cb__ uint8_t *)0;
  __ca__ uint8_t *buf_dyn_l0a = (__ca__ uint8_t *)0;
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  if ASC_IS_AIC {
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(3));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4));
    asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(5));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(3));
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_intra_arrive(PIPE_MTE1, 4);
    asc_sync_intra_arrive(PIPE_MTE1, 20);
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), (&(Q[((((((int32_t)block_idx) >> 8) * 4194304) + ((((int32_t)block_idx) & 7) * 524288)) + (((((int32_t)block_idx) & 255) >> 3) * 16384))])), 256, static_cast<asc_load_l2_cache_mode>(0), 128, 128, 0, 0);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(6));
    int32_t end = Lengths[(((int32_t)block_idx) >> 8)];
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(6));
    asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 98304), 0, 0, 8, 8, 8, 8);
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(2));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(2));
    for (int32_t tile = 0; tile < (max((end / 128), 0) + 7); ++tile) {
      int32_t condval;
      if (((end < 2176) || (tile < 19))) {
        condval = Table[(((((int32_t)block_idx) >> 8) * 16) + max(0, min((max((end / 128), 0) - 1), (tile - 3))))];
      } else {
        condval = 0;
      }
      int32_t physical = (condval * 128);
      if (((tile < ((end >> 7) + 3)) || (tile < 3)) && (3 <= tile)) {
        asc_sync_intra_wait(PIPE_FIX, ((tile % 3) + 16));
        asc_sync_intra_wait(PIPE_FIX, (tile % 3));
        asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>((((tile + 1) & 1) + 2)));
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[((tile % 3) * 8192)])), ((__cc__ float*)buf_dyn_l0c + (((tile + 1) & 1) * 16384)), 128, 128, 128, 128, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, ((tile % 3) + 5));
        asc_sync_intra_arrive(PIPE_FIX, ((tile % 3) + 21));
        asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>((((tile + 1) & 1) + 2)));
      }
      if (((tile <= (end >> 7)) || (tile < 1)) && (1 <= tile)) {
        asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>((((tile + 2) % 3) + 3)));
        asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((tile + 2) % 3) * 16384)), 0, 0, 8, 8, 8, 8);
        asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>((((tile + 2) % 3) + 3)));
      }
      int32_t condval_1;
      if (((end < 2176) || (tile < 16))) {
        condval_1 = Table[(((((int32_t)block_idx) >> 8) * 16) + max(0, min((max((end / 128), 0) - 1), tile)))];
      } else {
        condval_1 = 0;
      }
      int32_t physical_1 = (condval_1 * 128);
      if (tile < (end >> 7)) {
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(((tile % 3) + 3)));
        if (physical_1 < 4096) {
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
          asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((tile % 3) * 16384)), (&(K[((((int64_t)physical_1) * (int64_t)1024) + ((((int64_t)((int32_t)block_idx)) & (int64_t)7) * (int64_t)128))])), 2048, static_cast<asc_load_l2_cache_mode>(0), min(128, (4096 - physical_1)), 128, 0, 0);
        }
      }
      if (((tile < ((end >> 7) + 6)) || (tile < 6)) && (6 <= tile)) {
        asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
        asc_sync_intra_wait(PIPE_MTE1, 25);
        asc_sync_intra_wait(PIPE_MTE1, 9);
        asc_copy_l12l0a(((__ca__ bfloat16_t*)buf_dyn_l0a + 16384), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 114688), 0, 0, 8, 8, 8, 8);
        asc_sync_intra_arrive(PIPE_MTE1, 4);
        asc_sync_intra_arrive(PIPE_MTE1, 20);
      }
      if (7 <= tile) {
        asc_sync_intra_wait(PIPE_FIX, 19);
        asc_sync_intra_wait(PIPE_FIX, 3);
        asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(((tile + 1) & 1)));
        asc_set_l0c_copy_nz_para(1, 0, 0);
        asc_copy_l0c2ub((&(((__ubuf__ float*)buf_dyn_shmem)[32896])), ((__cc__ float*)buf_dyn_l0c + ((((tile + 1) & 1) * 16384) + 32768)), 128, 128, 128, 128, 0, static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
        asc_sync_intra_arrive(PIPE_FIX, 8);
        asc_sync_intra_arrive(PIPE_FIX, 24);
        asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(((tile + 1) & 1)));
      }
      if (((tile < ((end >> 7) + 6)) || (tile < 6)) && (6 <= tile)) {
        asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>((tile % 3)));
        asc_copy_l12l0b(((__cb__ bfloat16_t*)buf_dyn_l0b + 16384), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((tile % 3) * 16384) + 49152)), 0, 0, 8, 8, 8, 8);
        asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
        asc_sync_notify(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>((tile % 3)));
      }
      if (((tile <= (end >> 7)) || (tile < 1)) && (1 <= tile)) {
        asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>((((tile + 1) & 1) + 2)));
        asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(1));
        asc_mmad(((__cc__ float*)buf_dyn_l0c + (((tile + 1) & 1) * 16384)), ((__ca__ bfloat16_t*)buf_dyn_l0a), ((__cb__ bfloat16_t*)buf_dyn_l0b), 128, 128, 128, 0, 1, 0, (bool)1);
        asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>((((tile + 1) & 1) + 2)));
        asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
      }
      if (tile < (end >> 7)) {
        asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((tile % 3) * 16384)) + (min(128, max((4096 - physical_1), 0)) * 32)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(8), .blk_num = static_cast<uint64_t>(max(0, (128 - max((4096 - physical_1), 0)))), .dst_gap = static_cast<uint64_t>((128 - max(0, (128 - max((4096 - physical_1), 0)))))});
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(((tile % 3) + 3)));
      }
      if (((tile < ((end >> 7) + 3)) || (tile < 3)) && (3 <= tile)) {
        asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>((tile % 3)));
        if (physical < 4096) {
          asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(128), 0);
          asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((tile % 3) * 16384) + 49152)), (&(V[((((int64_t)physical) * (int64_t)1024) + ((((int64_t)((int32_t)block_idx)) & (int64_t)7) * (int64_t)128))])), 2048, static_cast<asc_load_l2_cache_mode>(0), 128, min(128, (4096 - physical)), 0, 0);
        }
      }
      if (((tile < ((end >> 7) + 6)) || (tile < 6)) && (6 <= tile)) {
        asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>((tile & 1)));
        asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
        asc_mmad(((__cc__ float*)buf_dyn_l0c + (((tile & 1) * 16384) + 32768)), ((__ca__ bfloat16_t*)buf_dyn_l0a + 16384), ((__cb__ bfloat16_t*)buf_dyn_l0b + 16384), 128, 128, 128, 0, 1, 0, (bool)1);
        asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>((tile & 1)));
        asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
      }
      if (((tile < ((end >> 7) + 3)) || (tile < 3)) && (3 <= tile)) {
        asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + (((tile % 3) * 16384) + 49152)) + (((min(128, max((4096 - physical), 0)) + 15) >> 4) * 4096)), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>((8 - ((min(128, max((4096 - physical), 0)) + 15) >> 4))), .blk_num = static_cast<uint64_t>(128), .dst_gap = static_cast<uint64_t>(0)});
        asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>((tile % 3)));
      }
    }
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(1));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(2));
    asc_sync_intra_wait(PIPE_FIX, 0);
    asc_sync_intra_wait(PIPE_FIX, 1);
    asc_sync_intra_wait(PIPE_FIX, 2);
    asc_sync_intra_wait(PIPE_FIX, 16);
    asc_sync_intra_wait(PIPE_FIX, 17);
    asc_sync_intra_wait(PIPE_FIX, 18);
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(3));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(4));
    asc_sync_wait(PIPE_MTE1, PIPE_MTE2, static_cast<event_t>(5));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(2));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(3));
    asc_sync_intra_wait(PIPE_FIX, 3);
    asc_sync_intra_wait(PIPE_FIX, 19);
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(1));
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_intra_arrive(PIPE_V, 0);
    asc_sync_intra_arrive(PIPE_V, 1);
    asc_sync_intra_arrive(PIPE_V, 2);
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_intra_arrive(PIPE_V, 3);
    direct_kernel_simd_vf_0(buf_dyn_shmem);
    int32_t end_1 = Lengths[(((int32_t)block_idx) >> 8)];
    direct_kernel_simd_vf_1(buf_dyn_shmem);
    for (int32_t tile_1 = 0; tile_1 < (max((end_1 / 128), 0) + 2); ++tile_1) {
      if (((tile_1 <= (end_1 >> 7)) || (tile_1 < 1)) && (1 <= tile_1)) {
        asc_sync_intra_wait(PIPE_MTE3, 4);
        asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(((tile_1 + 1) & 1)));
        asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)buf_dyn_l1 + ((sid * 1024) + 114688))), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((((tile_1 + 1) & 1) * 8320) + 49152)]))), 8, 64, 1, 64);
        asc_sync_intra_arrive(PIPE_MTE3, 9);
        asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(((tile_1 + 1) & 1)));
      }
      if (tile_1 < (end_1 >> 7)) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((tile_1 & 1)));
        asc_sync_intra_wait(PIPE_V, ((tile_1 % 3) + 5));
        direct_kernel_simd_vf_2(buf_dyn_shmem, tile_1);
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((tile_1 & 1)));
        asc_sync_intra_arrive(PIPE_V, (tile_1 % 3));
      }
      if (2 <= tile_1) {
        asc_sync_intra_wait(PIPE_V, 8);
        direct_kernel_simd_vf_3(buf_dyn_shmem, tile_1);
        asc_sync_intra_arrive(PIPE_V, 3);
      }
    }
    asc_vf_call<direct_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem);
    direct_kernel_simd_vf_4(buf_dyn_shmem);
    asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
    asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(2));
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(O[((((((int32_t)block_idx) >> 3) * 131072) + (sid * 65536)) + ((((int32_t)block_idx) & 7) * 128))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[82176]))), 64, 256, static_cast<asc_store_l2_cache_mode>(0), 2048, 256);
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
    asc_sync_intra_wait(PIPE_MTE3, 4);
  }
}

