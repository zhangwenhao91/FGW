# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def direct(Q_handle: T.handle, K_handle: T.handle, V_handle: T.handle, Table_handle: T.handle, Lengths_handle: T.handle, O_handle: T.handle):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["ascend"], "kind": "ascend", "launch_grid_only": True, "max_vector_bits": 64, "supports_kernel_status_return": False, "supports_vector_predicate": True, "tag": ""}), "tilelang_callee_allocated_outputs": 1, "tilelang_out_idx": [-1]})
        Q = T.match_buffer(Q_handle, (2, 8, 4096, 128), "bfloat16", strides=(4194304, 524288, 128, 1))
        K = T.match_buffer(K_handle, (4096, 8, 128), "bfloat16", strides=(1024, 128, 1))
        V = T.match_buffer(V_handle, (4096, 8, 128), "bfloat16", strides=(1024, 128, 1))
        Table = T.match_buffer(Table_handle, (2, 16), "int32", strides=(16, 1))
        Lengths = T.match_buffer(Lengths_handle, (2,), "int32", strides=(1,))
        O = T.match_buffer(O_handle, (2, 4096, 1024), "bfloat16", strides=(4194304, 1024, 1))
        with T.sblock("root"):
            T.reads()
            T.writes()
            probs = T.Buffer((65, 128), "bfloat16", scope="shared.dyn")
            pv = T.Buffer((128, 128), scope="shared.l0c")
            qk = T.Buffer((128, 128), scope="shared.l0c")
            Q_l1 = T.Buffer((128, 128), "bfloat16", scope="shared.l1")
            K_l1 = T.Buffer((128, 128), "bfloat16", scope="shared.l1")
            qa = T.Buffer((128, 128), "bfloat16", scope="shared.l0a")
            kb = T.Buffer((128, 128), "bfloat16", scope="shared.l0b")
            P_l1 = T.Buffer((128, 128), "bfloat16", scope="shared.l1")
            V_l1 = T.Buffer((128, 128), "bfloat16", scope="shared.l1")
            pa = T.Buffer((128, 128), "bfloat16", scope="shared.l0a")
            vb = T.Buffer((128, 128), "bfloat16", scope="shared.l0b")
            T.sblock_attr({"layout_map": {probs: metadata["tl.Layout"][0], pv: metadata["tl.AscendFractalLayout"][0], qk: metadata["tl.AscendFractalLayout"][1], Q_l1: metadata["tl.AscendFractalLayout"][2], K_l1: metadata["tl.AscendFractalLayout"][3], qa: metadata["tl.AscendFractalLayout"][4], kb: metadata["tl.AscendFractalLayout"][5], P_l1: metadata["tl.AscendFractalLayout"][6], V_l1: metadata["tl.AscendFractalLayout"][7], pa: metadata["tl.AscendFractalLayout"][8], vb: metadata["tl.AscendFractalLayout"][9]}})
            task = T.launch_thread("blockIdx.x", 512)
            sid = T.launch_thread("cthread", 2)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                total = T.handle("float32", "shared.dyn")
                maximum = T.handle("float32", "shared.dyn")
                cast_view = T.handle("float32", "shared.dyn")
                scores = T.handle("float32", "shared.dyn")
                old_max = T.handle("float32", "shared.dyn")
                row_sum = T.handle("float32", "shared.dyn")
                alpha = T.handle("float32", "shared.dyn")
                current = T.handle("float32", "shared.dyn")
                dst_nz_ptr = T.handle("handle", "local.var")
                T.sblock_attr({"layout_map": {probs: metadata["tl.Layout"][0], pv: metadata["tl.AscendFractalLayout"][0], qk: metadata["tl.AscendFractalLayout"][1], Q_l1: metadata["tl.AscendFractalLayout"][2], K_l1: metadata["tl.AscendFractalLayout"][3], qa: metadata["tl.AscendFractalLayout"][4], kb: metadata["tl.AscendFractalLayout"][5], P_l1: metadata["tl.AscendFractalLayout"][6], V_l1: metadata["tl.AscendFractalLayout"][7], pa: metadata["tl.AscendFractalLayout"][8], vb: metadata["tl.AscendFractalLayout"][9]}, "tl.buffer_alias_map": {Q_l1.data: [], total: [], maximum: [], cast_view: [], qa.data: [], probs.data: [], P_l1.data: [], scores: [], old_max: [], row_sum: [], alpha: [], qk.data: [], K_l1.data: [], kb.data: [], pa.data: [], pv.data: [], current: [], V_l1.data: [], vb.data: []}, "tl.buffer_versions_map": {K_l1.data: 3, kb.data: 1, qk.data: 2, scores: 3, old_max: 1, row_sum: 1, probs.data: 2, P_l1.data: 1, V_l1.data: 3, pa.data: 1, vb.data: 1, pv.data: 2, alpha: 3}, "tl.simd_pointer_dtypes": {dst_nz_ptr: "bfloat16"}})
                Q_l1 = T.sblock_alloc_buffer((128, 128), "bfloat16", data=Q_l1.data, scope="shared.l1")
                K_l1 = T.sblock_alloc_buffer((128, 128), "bfloat16", data=K_l1.data, scope="shared.l1")
                V_l1 = T.sblock_alloc_buffer((128, 128), "bfloat16", data=V_l1.data, scope="shared.l1")
                P_l1 = T.sblock_alloc_buffer((128, 128), "bfloat16", data=P_l1.data, scope="shared.l1")
                qa = T.sblock_alloc_buffer((128, 128), "bfloat16", data=qa.data, scope="shared.l0a")
                kb = T.sblock_alloc_buffer((128, 128), "bfloat16", data=kb.data, scope="shared.l0b")
                pa = T.sblock_alloc_buffer((128, 128), "bfloat16", data=pa.data, scope="shared.l0a")
                vb = T.sblock_alloc_buffer((128, 128), "bfloat16", data=vb.data, scope="shared.l0b")
                qk = T.sblock_alloc_buffer((128, 128), data=qk.data, scope="shared.l0c")
                pv = T.sblock_alloc_buffer((128, 128), data=pv.data, scope="shared.l0c")
                scores_1 = T.sblock_alloc_buffer((64, 128), data=scores, scope="shared.dyn")
                probs = T.sblock_alloc_buffer((65, 128), "bfloat16", data=probs.data, scope="shared.dyn")
                current_1 = T.sblock_alloc_buffer((64, 128), data=current, scope="shared.dyn")
                output = T.sblock_alloc_buffer((64, 128), data=cast_view, scope="shared.dyn")
                maximum_1 = T.sblock_alloc_buffer((64,), data=maximum, scope="shared.dyn")
                total_1 = T.sblock_alloc_buffer((64,), data=total, scope="shared.dyn")
                alpha_1 = T.sblock_alloc_buffer((64,), data=alpha, scope="shared.dyn")
                old_max_1 = T.sblock_alloc_buffer((64,), data=old_max, scope="shared.dyn")
                row_sum_1 = T.sblock_alloc_buffer((64,), data=row_sum, scope="shared.dyn")
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_MTE2", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_MTE2", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_MTE2", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_MTE2", 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_MTE2", 4)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_MTE2", 5)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("FIX_M", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("FIX_M", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE3_V", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE3_V", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("FIX_M", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("FIX_M", 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_V", 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("M_MTE1", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 4)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 20)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("M_MTE1", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2), "ii": T.int64(328), "latency": T.int64(518)}, "tl.ascend_task", 1)
                    T.ascend_copy(T.region(Q[task // 256, task % 8, task % 256 // 8 * 128, 0], 1, 1, 1, 128, 128), T.region(Q_l1[0, 0], 2, 128, 128))
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE2_MTE1", 6)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5), "ii": T.int64(400), "latency": T.int64(400)}, "tl.ascend_task", 1)
                    with T.sblock("SIMD_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"tl.vf_latency": T.int64(400), "tl.vf_source_index": T.int64(0)})
                        T.attr("simdvf", "tl.simdvf_scope", 1)
                        init_mask: T.boolx256 = T.tl.simd.pset(32, "PAT_ALL")
                        zero: T.float32x64 = T.tl.simd.vdup(T.float32(0.0), init_mask, "MODE_ZEROING")
                        initial_m: T.float32x64 = T.tl.simd.vdup(T.infinity("float32") * T.float32(-1.0), init_mask, "MODE_ZEROING")
                        T.tl.simd.vsts(T.access_ptr(total_1[0], 64, 2), zero, init_mask, "NORM_B32")
                        T.tl.simd.vsts(T.access_ptr(maximum_1[0], 64, 2), initial_m, init_mask, "NORM_B32")
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(7), "ii": T.int64(1), "latency": T.int64(1)}, "tl.ascend_task", 1)
                    end: T.int32 = Lengths[task // 256]
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5), "ii": T.int64(142), "latency": T.int64(142)}, "tl.ascend_task", 1)
                    with T.sblock("SIMD_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"tl.vf_latency": T.int64(142), "tl.vf_source_index": T.int64(1)})
                        T.attr("simdvf", "tl.simdvf_scope", 1)
                        clear_mask: T.boolx256 = T.tl.simd.pset(32, "PAT_ALL")
                        output_zero: T.float32x64 = T.tl.simd.vdup(T.float32(0.0), clear_mask, "MODE_ZEROING")
                        for i, _tmp in T.grid(64, 2):
                            T.tl.simd.vsts(T.access_ptr(output[i, _tmp * 64], 64, 2), output_zero, clear_mask, "NORM_B32")
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE2_MTE1", 6)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2), "ii": T.int64(131), "latency": T.int64(133)}, "tl.ascend_task", 1)
                    T.ascend_copy(T.region(Q_l1[0, 0], 1, 128, 128), T.region(qa[0, 0], 2, 128, 128))
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("MTE1_M", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_M", 2)
                end = T.int32()
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    for tile in T.serial(T.max(end // 128, 0), annotations={"enable_offset": True, "multi_buffer_eligible": [K_l1.data, kb.data, qk.data, scores, old_max, row_sum, probs.data, P_l1.data, V_l1.data, pa.data, vb.data, pv.data, alpha], "num_stages": 2}):
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(1), "latency": T.int64(1)}, "tl.ascend_task", 1)
                            physical: T.int32 = Table[task // 256, tile] * 128
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 4)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("V_MTE3", tile % 2)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5), "ii": T.int64(128), "latency": T.int64(171)}, "tl.ascend_task", 1)
                            T.ascend_nd2nz_post_copy(T.access_ptr(P_l1[sid * 64, 0], 15360, 2), T.access_ptr(probs[0, 0], 8304, 1), 64, 128, 65, 128, "bfloat16_t", nd2nz_logical_regions=3)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_MTE3", 9)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("MTE3_V", tile % 2)
                        with T.attr({"stage": 5}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE3_V", tile % 2)
                        with T.attr({"stage": 5}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_V", tile % 3 + 5)
                        with T.attr({"stage": 5}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5), "ii": T.int64(735), "latency": T.int64(735)}, "tl.ascend_task", 1)
                            with T.sblock("SIMD_VF", no_realize=True):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"tl.vf_latency": T.int64(735), "tl.vf_source_index": T.int64(2)})
                                dst_nz_ptr_1 = T.sblock_alloc_buffer((1,), "handle", data=dst_nz_ptr, scope="local.var")
                                T.attr("simdvf", "tl.simdvf_scope", 1)
                                softmax_full: T.boolx256 = T.tl.simd.pset(32, "PAT_ALL")
                                softmax_one: T.boolx256 = T.tl.simd.pset(32, "PAT_VL1")
                                softmax_b16: T.boolx256 = T.tl.simd.pset(16, "PAT_VL128")
                                dst_nz_ptr_1[0] = T.access_ptr(probs[0, 0], 8320, 2)
                                T.tl.simd.vsts(T.access_ptr(old_max_1[0], 64, 2), T.tl.simd.vld(T.access_ptr(maximum_1[0], 64, 1), "NORM"), softmax_full, "NORM_B32")
                                for _tmp in range(32):
                                    s00: T.float32x64 = T.tl.simd.vmuls(T.tl.simd.vld(T.access_ptr(scores_1[_tmp * 2, 0], 64, 1), "NORM"), T.float32(0.088388347648318447), softmax_full, "MODE_ZEROING")
                                    s01: T.float32x64 = T.tl.simd.vmuls(T.tl.simd.vld(T.access_ptr(scores_1[_tmp * 2, 64], 64, 1), "NORM"), T.float32(0.088388347648318447), softmax_full, "MODE_ZEROING")
                                    s10: T.float32x64 = T.tl.simd.vmuls(T.tl.simd.vld(T.access_ptr(scores_1[_tmp * 2 + 1, 0], 64, 1), "NORM"), T.float32(0.088388347648318447), softmax_full, "MODE_ZEROING")
                                    s11: T.float32x64 = T.tl.simd.vmuls(T.tl.simd.vld(T.access_ptr(scores_1[_tmp * 2 + 1, 64], 64, 1), "NORM"), T.float32(0.088388347648318447), softmax_full, "MODE_ZEROING")
                                    T.tl.simd.vsts(T.access_ptr(scores_1[_tmp * 2, 0], 64, 2), s00, softmax_full, "NORM_B32")
                                    T.tl.simd.vsts(T.access_ptr(scores_1[_tmp * 2, 64], 64, 2), s01, softmax_full, "NORM_B32")
                                    T.tl.simd.vsts(T.access_ptr(scores_1[_tmp * 2 + 1, 0], 64, 2), s10, softmax_full, "NORM_B32")
                                    T.tl.simd.vsts(T.access_ptr(scores_1[_tmp * 2 + 1, 64], 64, 2), s11, softmax_full, "NORM_B32")
                                    max0: T.float32x64 = T.tl.simd.vcmax(T.tl.simd.vmax(s00, s01, softmax_full, "MODE_ZEROING"), softmax_full, "MODE_ZEROING")
                                    max1: T.float32x64 = T.tl.simd.vcmax(T.tl.simd.vmax(s10, s11, softmax_full, "MODE_ZEROING"), softmax_full, "MODE_ZEROING")
                                    T.tl.simd.vsts(T.access_ptr(maximum_1[_tmp * 2], 1, 2), max0, softmax_one, "ONEPT_B32")
                                    T.tl.simd.vsts(T.access_ptr(maximum_1[_tmp * 2 + 1], 1, 2), max1, softmax_one, "ONEPT_B32")
                                T.tl.simd.mem_bar("VST_VLD")
                                merged_max: T.float32x64 = T.tl.simd.vmax(T.tl.simd.vld(T.access_ptr(maximum_1[0], 64, 1), "NORM"), T.tl.simd.vld(T.access_ptr(old_max_1[0], 64, 1), "NORM"), softmax_full, "MODE_ZEROING")
                                T.tl.simd.vsts(T.access_ptr(maximum_1[0], 64, 2), merged_max, softmax_full, "NORM_B32")
                                T.tl.simd.mem_bar("VST_VLD")
                                for _tmp in range(32):
                                    max0: T.float32x64 = T.tl.simd.vld(T.access_ptr(maximum_1[_tmp * 2], 1, 1), "BRC_B32")
                                    max1: T.float32x64 = T.tl.simd.vld(T.access_ptr(maximum_1[_tmp * 2 + 1], 1, 1), "BRC_B32")
                                    v: T.float32x64 = T.tl.simd.vld2(T.access_ptr(scores_1[_tmp * 2, 0], 128, 1), "DINTLV_B32")
                                    p00: T.float32x64 = T.tl.simd.pair_get(v, 0)
                                    p01: T.float32x64 = T.tl.simd.pair_get(v, 1)
                                    v_1: T.float32x64 = T.tl.simd.vld2(T.access_ptr(scores_1[_tmp * 2 + 1, 0], 128, 1), "DINTLV_B32")
                                    p10: T.float32x64 = T.tl.simd.pair_get(v_1, 0)
                                    p11: T.float32x64 = T.tl.simd.pair_get(v_1, 1)
                                    e00: T.float32x64 = T.tl.simd.vexpdif(p00, max0, softmax_full)
                                    e10: T.float32x64 = T.tl.simd.vexpdif(p10, max1, softmax_full)
                                    e01: T.float32x64 = T.tl.simd.vexpdif(p01, max0, softmax_full)
                                    e11: T.float32x64 = T.tl.simd.vexpdif(p11, max1, softmax_full)
                                    row_sum0: T.float32x64 = T.tl.simd.vcadd(T.tl.simd.vadd(e00, e01, softmax_full, "MODE_ZEROING"), softmax_full, "MODE_ZEROING")
                                    row_sum1: T.float32x64 = T.tl.simd.vcadd(T.tl.simd.vadd(e10, e11, softmax_full, "MODE_ZEROING"), softmax_full, "MODE_ZEROING")
                                    T.tl.simd.vsts(T.access_ptr(row_sum_1[_tmp * 2], 1, 2), row_sum0, softmax_one, "ONEPT_B32")
                                    T.tl.simd.vsts(T.access_ptr(row_sum_1[_tmp * 2 + 1], 1, 2), row_sum1, softmax_one, "ONEPT_B32")
                                    even_bf16_0: T.bfloat16x128 = T.tl.simd.vcvt(e00, softmax_full, "ROUND_R", "RS_DISABLE", "PART_EVEN", "MODE_ZEROING")
                                    even_bf16_1: T.bfloat16x128 = T.tl.simd.vcvt(e10, softmax_full, "ROUND_R", "RS_DISABLE", "PART_EVEN", "MODE_ZEROING")
                                    odd_bf16_0: T.bfloat16x128 = T.tl.simd.vcvt(e01, softmax_full, "ROUND_R", "RS_DISABLE", "PART_ODD", "MODE_ZEROING")
                                    odd_bf16_1: T.bfloat16x128 = T.tl.simd.vcvt(e11, softmax_full, "ROUND_R", "RS_DISABLE", "PART_ODD", "MODE_ZEROING")
                                    merged_bits0: T.uint16x128 = T.tl.simd.vor(T.reinterpret("uint16x128", even_bf16_0), T.reinterpret("uint16x128", odd_bf16_0), softmax_b16, "MODE_ZEROING")
                                    merged_bits1: T.uint16x128 = T.tl.simd.vor(T.reinterpret("uint16x128", even_bf16_1), T.reinterpret("uint16x128", odd_bf16_1), softmax_b16, "MODE_ZEROING")
                                    dst_nz_ptr_1[0] = T.tl.simd.vsstb(T.reinterpret("bfloat16x128", merged_bits0), dst_nz_ptr_1[0], 4259841, softmax_b16, "POST_UPDATE")
                                    dst_nz_ptr_1[0] = T.tl.simd.vsstb(T.reinterpret("bfloat16x128", merged_bits1), dst_nz_ptr_1[0], 4259841, softmax_b16, "POST_UPDATE")
                                T.tl.simd.mem_bar("VST_VLD")
                                alpha_reg: T.float32x64 = T.tl.simd.vexpdif(T.tl.simd.vld(T.access_ptr(old_max_1[0], 64, 1), "NORM"), T.tl.simd.vld(T.access_ptr(maximum_1[0], 64, 1), "NORM"), softmax_full)
                                T.tl.simd.vsts(T.access_ptr(alpha_1[0], 64, 2), alpha_reg, softmax_full, "NORM_B32")
                                l_scaled: T.float32x64 = T.tl.simd.vmul(T.tl.simd.vld(T.access_ptr(total_1[0], 64, 1), "NORM"), alpha_reg, softmax_full, "MODE_ZEROING")
                                l_updated: T.float32x64 = T.tl.simd.vadd(l_scaled, T.tl.simd.vld(T.access_ptr(row_sum_1[0], 64, 1), "NORM"), softmax_full, "MODE_ZEROING")
                                T.tl.simd.vsts(T.access_ptr(total_1[0], 64, 2), l_updated, softmax_full, "NORM_B32")
                        with T.attr({"stage": 5}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("V_MTE3", tile % 2)
                        with T.attr({"stage": 5}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_V", tile % 3)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_FIX", tile % 3 + 16)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_FIX", tile % 3)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("M_FIX", tile % 2 + 2)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(258), "latency": T.int64(311)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(qk[0, 0], 1, 128, 128), T.region(scores_1[0, 0], 2, 64, 128), dual_dst_ctl=1, l2_cache_ctrl=0)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_FIX", tile % 3 + 5)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_FIX", tile % 3 + 21)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("FIX_M", tile % 2 + 2)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("M_MTE1", 1)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE2_MTE1", tile % 3 + 3)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(131), "latency": T.int64(133)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(K_l1[0, 0], 1, 128, 128), T.region(kb[0, 0], 2, 128, 128))
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("MTE1_M", 1)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("MTE1_MTE2", tile % 3 + 3)
                        with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(1), "latency": T.int64(1)}, "tl.ascend_task", 1)
                            physical: T.int32 = Table[task // 256, tile] * 128
                        with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE1_MTE2", tile % 3 + 3)
                        physical = T.int32()
                        with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(328), "latency": T.int64(518)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(K[physical, task % 8, 0], 1, T.min(128, T.max(4096 - physical, 0)), 1, 128), T.region(K_l1[0, 0], 2, T.min(128, T.max(4096 - physical, 0)), 128))
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("M_MTE1", 0)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 25)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_MTE1", 9)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(131), "latency": T.int64(133)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(P_l1[0, 0], 1, 128, 128), T.region(pa[0, 0], 2, 128, 128))
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 4)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_MTE1", 20)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 19)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 3)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("M_FIX", tile % 2)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(258), "latency": T.int64(311)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(pv[0, 0], 1, 128, 128), T.region(current_1[0, 0], 2, 64, 128), dual_dst_ctl=1, l2_cache_ctrl=0)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_FIX", 8)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_FIX", 24)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("FIX_M", tile % 2)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE2_MTE1", tile % 3)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(131), "latency": T.int64(133)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(V_l1[0, 0], 1, 128, 128), T.region(vb[0, 0], 2, 128, 128))
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("MTE1_M", 0)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("MTE1_MTE2", tile % 3)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("FIX_M", tile % 2 + 2)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE1_M", 1)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(512), "latency": T.int64(520)}, "tl.ascend_task", 1)
                            T.gemm(T.region(qa[0, 0], 1, 128, 128), T.region(kb[0, 0], 1, 128, 128), T.region(qk[0, 0], 2, 128, 128), T.bool(False), T.bool(True), 128, 128, 128, 0, T.bool(True), 0, 0, 0)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("M_FIX", tile % 2 + 2)
                        with T.attr({"stage": 1}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("M_MTE1", 1)
                        with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(328), "latency": T.int64(518)}, "tl.ascend_task", 1)
                            T.fill(T.region(K_l1[T.min(128, T.max(4096 - physical, 0)), 0], 2, 128 - T.min(128, T.max(4096 - physical, 0)), T.FloorDiv(143, 16) * 16), T.bfloat16(0.0))
                        with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("MTE2_MTE1", tile % 3 + 3)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE1_MTE2", tile % 3)
                        physical_1 = T.int32()
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(328), "latency": T.int64(518)}, "tl.ascend_task", 1)
                            T.ascend_copy(T.region(V[physical_1, task % 8, 0], 1, T.min(128, T.max(4096 - physical_1, 0)), 1, 128), T.region(V_l1[0, 0], 2, 128, T.min(128, T.max(4096 - physical_1, 0))), transpose=1)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_wait_flag(4, "PIPE_V", 8)
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5), "ii": T.int64(318), "latency": T.int64(318)}, "tl.ascend_task", 1)
                            with T.sblock("SIMD_VF", no_realize=True):
                                T.reads()
                                T.writes()
                                T.sblock_attr({"tl.vf_latency": T.int64(318), "tl.vf_source_index": T.int64(3)})
                                T.attr("simdvf", "tl.simdvf_scope", 1)
                                rescale_mask: T.boolx256 = T.tl.simd.pset(32, "PAT_ALL")
                                for i in range(64):
                                    alpha_2: T.float32x64 = T.tl.simd.vld(T.access_ptr(alpha_1[i], 1, 1), "BRC_B32")
                                    for _tmp in range(2):
                                        prev: T.float32x64 = T.tl.simd.vld(T.access_ptr(output[i, _tmp * 64], 64, 1), "NORM")
                                        cur: T.float32x64 = T.tl.simd.vld(T.access_ptr(current_1[i, _tmp * 64], 64, 1), "NORM")
                                        updated: T.float32x64 = T.tl.simd.vadd(T.tl.simd.vmul(alpha_2, prev, rescale_mask, "MODE_ZEROING"), cur, rescale_mask, "MODE_ZEROING")
                                        T.tl.simd.vsts(T.access_ptr(output[i, _tmp * 64], 64, 2), updated, rescale_mask, "NORM_B32")
                        with T.attr({"stage": 7}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                            T.ascend_cross_core_set_flag(4, "PIPE_V", 3)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("FIX_M", tile % 2)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_wait_flag("MTE1_M", 0)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(512), "latency": T.int64(520)}, "tl.ascend_task", 1)
                            T.gemm(T.region(pa[0, 0], 1, 128, 128), T.region(vb[0, 0], 1, 128, 128), T.region(pv[0, 0], 2, 128, 128), T.bool(False), T.bool(True), 128, 128, 128, 0, T.bool(True), 0, 0, 0)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("M_FIX", tile % 2)
                        with T.attr({"stage": 6}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                            T.ascend_set_flag("M_MTE1", 0)
                        with T.attr({"stage": 3}, "tl.schedule_unit", 1):
                            T.attr({"core_mask": T.int64(2), "ii": T.int64(328), "latency": T.int64(518)}, "tl.ascend_task", 1)
                            T.fill(T.region(V_l1[0, (T.min(128, T.max(4096 - physical_1, 0)) + 16 - 1) // 16 * 16], 2, T.FloorDiv(143, 16) * 16, (T.FloorDiv(143, 16) - (T.min(128, T.max(4096 - physical_1, 0)) + 16 - 1) // 16) * 16), T.bfloat16(0.0))
                        T.attr({"stage": 3}, "tl.schedule_unit", 1)
                        T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                        T.ascend_set_flag("MTE2_MTE1", tile % 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5), "ii": T.int64(5840), "latency": T.int64(5840)}, "tl.ascend_task", 1)
                    with T.sblock("SIMT_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"layout_map": {probs: metadata["tl.Layout"][0], pv: metadata["tl.AscendFractalLayout"][0], qk: metadata["tl.AscendFractalLayout"][1], Q_l1: metadata["tl.AscendFractalLayout"][2], K_l1: metadata["tl.AscendFractalLayout"][3], qa: metadata["tl.AscendFractalLayout"][4], kb: metadata["tl.AscendFractalLayout"][5], P_l1: metadata["tl.AscendFractalLayout"][6], V_l1: metadata["tl.AscendFractalLayout"][7], pa: metadata["tl.AscendFractalLayout"][8], vb: metadata["tl.AscendFractalLayout"][9]}, "tl.vf_source_index": T.int64(0)})
                        simtvf_tx = T.launch_thread("threadIdx.x", 128)
                        simtvf_ty = T.launch_thread("threadIdx.y", 1)
                        simtvf_tz = T.launch_thread("threadIdx.z", 1)
                        T.attr("simtvf", "tl.simtvf_scope", 1)
                        for r_d_fused in T.parallel(8192, annotations={"parallel_loop_layout": metadata["tl.Fragment"][0], "parallel_loop_requires_padding_guard": T.bool(True)}):
                            output[r_d_fused % 8192 // 128, r_d_fused % 128] = output[r_d_fused % 8192 // 128, r_d_fused % 128] / T.max(total_1[r_d_fused % 8192 // 128], T.float32(1.0000000000000001e-30))
                cast_view_1 = T.Buffer((128, 128), "bfloat16", data=cast_view, strides=(128, 1), scope="shared.dyn")
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5), "ii": T.int64(193), "latency": T.int64(193)}, "tl.ascend_task", 1)
                    with T.sblock("SIMD_VF", no_realize=True):
                        T.reads()
                        T.writes()
                        T.sblock_attr({"tl.vf_latency": T.int64(193), "tl.vf_source_index": T.int64(4)})
                        T.attr("simdvf", "tl.simdvf_scope", 1)
                        mask: T.boolx256 = T.tl.simd.pset(32, "PAT_ALL")
                        for r, _tmp in T.grid(64, 2):
                            value: T.bfloat16x128 = T.tl.simd.vcvt(T.tl.simd.vld(T.access_ptr(output[r, _tmp * 64], 64, 1), "NORM"), mask, "ROUND_R", "RS_ENABLE", "PART_EVEN", "MODE_ZEROING")
                            T.tl.simd.vsts(T.access_ptr(cast_view_1[r, _tmp * 64], 64, 2), value, mask, "PK_B32")
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_set_flag("V_MTE3", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("V_MTE3", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5), "ii": T.int64(298), "latency": T.int64(478)}, "tl.ascend_task", 1)
                    T.ascend_copy(T.region(cast_view_1[0, 0], 1, 64, 128), T.region(O[task // 256, task % 256 // 8 * 128 + sid * 64, task % 8 * 128], 2, 1, 64, 128), l2_cache_ctrl=0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 16)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 17)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 18)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 4)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE1_MTE2", 5)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("FIX_M", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("FIX_M", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE3_V", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("MTE3_V", 1)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("FIX_M", 2)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("FIX_M", 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 3)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 19)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                    T.ascend_wait_flag("M_MTE1", 0)
                with T.attr({"stage": 0}, "tl.schedule_unit", 1):
                    T.attr({"core_mask": T.int64(5)}, "tl.ascend_task", 1)
                    T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", 4)
                T.attr({"stage": 0}, "tl.schedule_unit", 1)
                T.attr({"core_mask": T.int64(2)}, "tl.ascend_task", 1)
                T.ascend_wait_flag("M_MTE1", 1)

# Metadata omitted. Use show_meta=True in script() method to show it.