"""Collect actual FIA runtime names/logs and current v7 generated code; no tuning changes."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "mha_kernel_inspect"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str)+"\n", encoding="utf-8")


def child(mode, directory, device_index):
    result = {"status": "running", "mode": mode}
    path = directory/(mode+".json")
    write_json(path, result)
    try:
        import torch
        import torch_npu
        torch.npu.set_device(device_index)
        device = torch.device("npu",device_index)
        result["versions"] = {"torch":torch.__version__,"torch_npu":torch_npu.__version__}
        result["cann_environment"] = {key:os.environ.get(key) for key in ("ASCEND_HOME_PATH","ASCEND_TOOLKIT_HOME","ASCEND_OPP_PATH","ASCEND_RT_VISIBLE_DEVICES")}
        try:
            result["cann_version"] = str(torch_npu.npu.get_cann_version())
        except Exception as exc:
            result["cann_version_probe_error"] = str(exc)
        result["device"] = str(torch.npu.get_device_properties(device_index))
        b,sq,h,d,kv,page = 2,4096,8,128,2048,128
        if mode == "v7-export":
            import tilelang
            from operators.mha.kernels import mha_paged_decode_ascend_v7 as v7
            program = v7.build_kernel(b,sq,h,d,b*kv,kv//page,page,False,"bfloat16",device_index,
                schedule_mode="manual",full_kv_cache=True,dump_ir_dir=str(directory/"v7_passes"))
            if program.execution_mode != "direct_single_split":
                raise ValueError("Expected direct path")
            source = program.direct_kernel.get_kernel_source()
            if not source:
                raise ValueError("No generated source returned")
            (directory/"v7_device.cpp").write_text(source,encoding="utf-8")
            result.update(tilelang=tilelang.__version__,split_policy=program.split_policy,
                num_splits=program.num_splits,query_tile_tasks=program.query_tile_tasks,
                kernel_source_sha256=hashlib.sha256(Path(v7.__file__).read_bytes()).hexdigest(),
                generated_source_sha256=hashlib.sha256(source.encode()).hexdigest())
            result["ir_files"] = [str(p.relative_to(directory)) for p in (directory/"v7_passes").rglob("*.py")]
        else:
            from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
            gen = torch.Generator().manual_seed(7)
            table = torch.randperm(b*(kv//page),generator=gen).reshape(b,kv//page).int()
            lengths = torch.full((b,),kv,dtype=torch.int32)
            q = torch.randn(b,sq,h,d,generator=gen).bfloat16()
            k = torch.randn(b*kv,h,d,generator=gen).bfloat16()
            v = torch.randn(b*kv,h,d,generator=gen).bfloat16()
            expected = chunked_cpu_reference(q,k,v,table,lengths,page,False,128)
            inputs = [x.to(device) for x in (q,k,v,table)]
            def call():
                return torch_npu.npu_fused_infer_attention_score_v2(inputs[0],
                    inputs[1].view(b*kv//page,page,h*d),inputs[2].view(b*kv//page,page,h*d),
                    block_table=inputs[3],num_query_heads=h,num_key_value_heads=h,
                    input_layout="BSND",block_size=page,softmax_scale=d**-0.5,
                    actual_seq_qlen=[sq]*b,actual_seq_kvlen=[kv]*b,
                    sparse_mode=0,inner_precise=0,return_softmax_lse=False)[0]
            got = call().cpu()
            torch.testing.assert_close(got,expected,rtol=1e-2,atol=4e-3)
            result["accuracy"] = {"status":"passed","max_abs_error":(got.float()-expected.float()).abs().max().item()}
            result["fia_schema"] = str(torch.ops.npu.npu_fused_infer_attention_score_v2._schemas)
            if mode == "fia-profile":
                for _ in range(5):
                    call()
                torch.npu.synchronize()
                trace_dir = directory/"fia_trace"
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,with_stack=False,profile_memory=False,
                    schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1,skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False,data_simplification=False)) as prof:
                    with torch.profiler.record_function("FGW_FIA_Q4096_KV2048"):
                        output = call()
                        torch.npu.synchronize()
                    prof.step()
                result["trace_directory"] = str(trace_dir)
                # CSV export via the installed tensorboard handler is primary.
                result["trace_files"] = [str(p.relative_to(directory)) for p in trace_dir.rglob("*") if p.is_file()]
        result["status"] = "passed"
    except Exception:
        result.update(status="error",error=traceback.format_exc())
        traceback.print_exc()
    write_json(path,result)
    return 0 if result["status"]=="passed" else 1


def collect(output, directory, device_index):
    report = {"status":"running","artifact_dir":str(directory),
        "shape":{"B":2,"Q":4096,"H":8,"D":128,"KV":2048,"page_size":128},
        "notes":["Inspection only: DEBUG/profiler collection is not a latency benchmark.",
            "Runtime log fields are observations, not automatically decoded tiling bytes.",
            "blockSize is the cache page size; s1Size/s2Size may be input lengths. Do not equate them to compute tiles.",
            "Missing fields remain unknown; installed CANN may differ from public master.",
            "No source/kernel/model/system configuration changes; DEBUG env applies to FIA child only."],
        "children":{},"runtime_kernel_rows":[],"trace_kernel_events":[]}
    def save():
        write_json(output/(PREFIX+"_summary.json"),report)
    save()
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(ROOT)+os.pathsep+env_base.get("PYTHONPATH","")
    for mode in ("fia-debug","fia-profile","v7-export"):
        env = env_base.copy()
        # Keep the normal profile separate from the diagnostic DEBUG execution.
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "0" if mode=="fia-debug" else "3"
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        command = [sys.executable,"-u","-m","tools.mha.inspect_runtime","--child",mode,
            "--artifact-dir",str(directory),"--device",str(device_index)]
        log = directory/(mode+".log")
        print(f"Collecting {mode} ...",flush=True)
        try:
            with log.open("w",encoding="utf-8") as handle:
                result = subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=600)
            item = json.loads((directory/(mode+".json")).read_text())
            item["returncode"] = result.returncode
        except Exception:
            item = {"status":"error","error":traceback.format_exc()}
        report["children"][mode] = item
        save()
    debug_log = directory/"fia-debug.log"
    interesting = []
    pattern = re.compile(r"tiling|fused.?infer|flash.?attention|s[12].*(?:base|inner|outer)|base[MN]|kernel.?name|block.?dim",re.I)
    if debug_log.exists():
        with debug_log.open(encoding="utf-8",errors="replace") as handle:
            for line in handle:
                if pattern.search(line):
                    interesting.append(line.rstrip())
    # Bounded upload artifact; the full child log remains on the NPU machine.
    primary = [line for line in interesting if re.search(r"fused.?infer|flash.?attention|fia.?tiling",line,re.I)]
    primary_set = set(primary)
    secondary = [line for line in interesting if line not in primary_set]
    retained = primary[-1200:]+secondary[-300:]
    filtered = output/(PREFIX+"_runtime_tiling.log")
    filtered.write_text("\n".join(retained)+"\n",encoding="utf-8")
    report["runtime_log"] = {"matched_lines":len(interesting),"retained_lines":len(retained),
        "truncated":len(interesting)>len(retained),"path":str(filtered),"raw_path":str(debug_log)}
    observed = {}
    for line in retained:
        for key,value in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*(?:\[\d+\])?)\s*[:=]\s*(-?0x[0-9a-fA-F]+|-?\d+(?:\.\d+)?)",line):
            if any(word in key.lower() for word in ("size","base","tile","tiling","block","core","split","buf","num")):
                observed.setdefault(key,set()).add(value)
    report["runtime_numeric_fields"] = {key:sorted(values) for key,values in observed.items()}
    report["tiling_status"] = "runtime_log_fields_available" if observed else "not_observed"
    trace_dir = directory/"fia_trace"
    for path in sorted(trace_dir.rglob("*.csv")):
        if any(word in path.name.lower() for word in ("kernel_details","op_summary","opbasicinfo")):
            try:
                with path.open(encoding="utf-8-sig",errors="replace",newline="") as handle:
                    rows = list(csv.DictReader(handle))
                report["runtime_kernel_rows"].append({"source":str(path),"rows":rows[:128],
                    "total_rows":len(rows),"truncated":len(rows)>128})
            except Exception as exc:
                report.setdefault("parse_errors",[]).append({"source":str(path),"error":str(exc)})
    if not any(item["rows"] for item in report["runtime_kernel_rows"]):
        # Trace fallback preserves exact event names/args without claiming their meaning.
        for path in trace_dir.rglob("*.json"):
            if path.stat().st_size>64*1024*1024:
                continue
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
                for event in trace.get("traceEvents",[]) if isinstance(trace,dict) else []:
                    if "kernel" in str(event.get("cat","")).lower():
                        report["trace_kernel_events"].append({k:event.get(k) for k in ("name","cat","dur","args")})
                        if len(report["trace_kernel_events"])>=128:
                            break
            except Exception:
                pass
    rows_path = output/(PREFIX+"_kernel_rows.json")
    write_json(rows_path,{"csv":report["runtime_kernel_rows"],"trace":report["trace_kernel_events"]})
    source_file = directory/"v7_device.cpp"
    if source_file.exists():
        shutil.copyfile(source_file,output/(PREFIX+"_v7_device.cpp"))
    for tag in ("AutoSchedule","InsertSync"):
        found = sorted((directory/"v7_passes").rglob("*"+tag+"*.py"))
        if found:
            dest = output/(PREFIX+"_v7_"+tag+".py")
            shutil.copyfile(found[-1],dest)
            report.setdefault("v7_ir_snapshots",{})[tag]=str(dest)
    names_available = any(item["rows"] for item in report["runtime_kernel_rows"]) or bool(report["trace_kernel_events"])
    report["kernel_names_status"] = "profiler_records_available" if names_available else "not_observed"
    report["status"] = "passed" if all(x.get("status")=="passed" for x in report["children"].values()) and names_available else "partial_error"
    save()
    print(f"{report['status']}: {output/(PREFIX+'_summary.json')}; tiling={report['tiling_status']}",flush=True)
    return 0 if report["status"]=="passed" else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child",choices=["fia-debug","fia-profile","v7-export"])
    parser.add_argument("--artifact-dir",type=Path)
    parser.add_argument("--device",type=int,default=0)
    args=parser.parse_args()
    output=Path.cwd()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory=(args.artifact_dir or output/(PREFIX+"_"+stamp)).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if args.child:
        return child(args.child,directory,args.device)
    return collect(output,directory,args.device)


if __name__=="__main__":
    raise SystemExit(main())
