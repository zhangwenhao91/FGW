"""Two-phase launch experiment; original raw SIMD math and task ownership."""
import torch
import tilelang
from operators.mha.phase_schedule import split_tables


def build_overlap(base, device):
    if not base.raw_partials:
        raise ValueError('Requires original raw A/max/sum baseline')
    tables = split_tables(base.plan)
    uploaded = tuple(tuple(torch.tensor(value,dtype=torch.int32,device=device) for value in phase)
                     for phase in tables)
    merges = base.prepare_plan(device)[2]
    inplace = tilelang.compile(base.local_kernel.prim_func, out_idx=[], target='ascend',
                               execution_backend='auto', pass_configs={})
    if inplace.get_kernel_source() != base.local_kernel.get_kernel_source():
        raise ValueError('In-place local device source differs from baseline')
    side = torch.npu.Stream(device=device)
    ready,done = torch.npu.Event(),torch.npu.Event()
    sq,slots = base.plan.q,max(1,base.plan.partial_count)

    def launch(parallel,q,k,v,table,lengths):
        # Allocation/preparation contract matches the baseline; both phases write
        # into the same output. Slots and split/direct Q rows are disjoint.
        prepared = q.permute(0,2,1,3).contiguous()
        parts = torch.empty((slots,128,128),dtype=torch.float32,device=q.device)
        stats = torch.empty((2*slots*128,),dtype=torch.float32,device=q.device)
        output = torch.empty((1,sq,128),dtype=torch.bfloat16,device=q.device)
        inplace(prepared,k,v,table,*uploaded[0],parts,stats,output)
        if parallel:
            current = torch.npu.current_stream(q.device)
            ready.record(current)
            # All tensors consumed across streams stay live through that use.
            for tensor in (parts,stats,output):
                tensor.record_stream(side)
            with torch.npu.stream(side):
                side.wait_event(ready)
                base.merge_kernel(parts,stats,merges,output)
                done.record(side)
            inplace(prepared,k,v,table,*uploaded[1],parts,stats,output)
            current.wait_event(done)
        else:
            # Same three kernels/buffers as parallel, with no overlap.
            inplace(prepared,k,v,table,*uploaded[1],parts,stats,output)
            base.merge_kernel(parts,stats,merges,output)
        return output.view(1,sq,1,128)

    import functools
    serial = functools.partial(launch,False)
    parallel = functools.partial(launch,True)
    return dict(serial=serial,parallel=parallel,kernel=inplace,tables=tables,
                scope='Single host thread/caller stream; reused events/side stream; not a concurrent-call API')
