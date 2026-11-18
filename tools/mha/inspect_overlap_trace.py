"""Inspect retained CANN trace schema without importing torch or running NPU work."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def brief(event):
    result = {k:event[k] for k in ('name','cat','ph','pid','tid','ts','dur') if k in event}
    result['args'] = {str(k):str(v)[:240] for k,v in list(event.get('args',{}).items())[:12]}
    return result


def inspect(path):
    payload = json.loads(path.read_text(encoding='utf8'))
    events = payload.get('traceEvents',[]) if isinstance(payload,dict) else payload if isinstance(payload,list) else []
    events = [e for e in events if isinstance(e,dict)]
    categories = Counter((str(e.get('cat','')),str(e.get('ph',''))) for e in events)
    matched = [e for e in events if any(term in (str(e.get('name',''))+' '+str(e.get('args',{}))).lower()
                                      for term in ('local_split','merge_simd'))]
    selected = [e for e in matched if e.get('ph')=='X'][:16]
    samples = defaultdict(list)
    for e in events:
        if e.get('ph')=='X' and len(samples[str(e.get('cat',''))])<2:
            samples[str(e.get('cat',''))].append(e)
    for group in list(samples.values())[:16]:
        selected.extend(group)
    process_ids = {e.get('pid') for e in selected}
    metadata = [e for e in events if e.get('ph')=='M' and e.get('pid') in process_ids
                and e.get('name') in ('process_name','thread_name')][:32]
    return dict(file=str(path),events=len(events),matched_names_or_args=len(matched),
                categories=[dict(cat=cat,ph=ph,count=count) for (cat,ph),count in categories.most_common(40)],
                metadata=[brief(e) for e in metadata],samples=[brief(e) for e in selected])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=sorted(p for p in args.directory.rglob('*.json')
                 if p.name=='trace_view.json' or p.name.startswith('msprof_'))
    if not paths:
        parser.error('No retained trace_view.json or msprof_*.json found')
    lines=['Read-only schema inspection. No kernel execution or profiler export.']
    for path in paths:
        lines.append(json.dumps(inspect(path),ensure_ascii=False))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(args.output,flush=True)


if __name__=='__main__':
    main()
