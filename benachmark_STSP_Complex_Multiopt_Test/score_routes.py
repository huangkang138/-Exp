#!/usr/bin/env python3
"""Score .jsonl route proposals against certified global optimal LENGTHS.
Input line: {"instance":"mstsp_n032_rectangle_01", "routes":[[0,1,...],...]}
Routes must have zero-based indices in .tsp line order; closed edge is implicit.
This reports exact-gap, best-distance hits, and number of distinct proven-optimal
routes FOUND, NOT recall/F-beta (the full optimal set is unknown).
"""
import argparse
import csv
import json
from math import isqrt
from pathlib import Path


def canonical(route):
    j=route.index(min(route));rt=route[j:]+route[:j]
    back=[rt[0]]+rt[:0:-1]
    return tuple(min(rt,back))


def integer_euc_len(route,pts):
    result=0;n=len(route)
    for i in range(n):
        x1,y1=pts[route[i]];x2,y2=pts[route[(i+1)%n]]
        ds=(x1-x2)**2+(y1-y2)**2
        # floor(sqrt(ds)+0.5), integer-arithmetic exact rounding for ds integer
        k=isqrt(ds)
        result += k+int(4*ds >= (2*k+1)**2)
    return result


def score(root,pred,out):
    with (root/'manifest.csv').open(encoding='utf-8',newline='') as f: manifest={r['name']:r for r in csv.DictReader(f)}
    entries={}
    for line in pred.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        row=json.loads(line);name=row['instance']
        if name not in manifest:raise ValueError(f'unknown instance {name}')
        if name in entries:raise ValueError(f'duplicate instance line {name}')
        entries[name]=row['routes']
    result=[]
    for name,m in manifest.items():
        n=int(m['n']);optimal=int(m['exact_optimal_length']);proposed=entries.get(name,[])
        coords=[tuple(map(int,s.split())) for s in (root/m['file']).read_text(encoding='utf-8').splitlines() if s.strip()]
        best=None;unique=set();perfect=set()
        for route in proposed:
            if len(route)!=n or set(route)!=set(range(n)):raise ValueError(f'invalid Hamiltonian cycle in {name}')
            key=canonical(route);unique.add(key)
            l=integer_euc_len(route,coords)
            best=l if best is None else min(best,l)
            if l<optimal:raise ValueError(f'PROOF CONTRADICTION {name}: {l} < {optimal}')
            if l==optimal:perfect.add(key)
        result.append({'name':name,'tier':m['tier'],'n':n,'shape':m['shape'],
                       'certified_optimal_length':optimal,'best_length':best if best is not None else '',
                       'gap_pct':(100*(best/optimal-1)) if best is not None else '',
                       'hit_exact_optimal':int(best==optimal),'routes_evaluated':len(proposed),
                       'distinct_generated_routes':len(unique),'distinct_proven_optimal_found':len(perfect),
                       'certified_witness_lower_bound':int(m['certified_optimal_witness_count'])})
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    checked=sum(bool(entries.get(x)) for x in manifest)
    print(f'{checked}/{len(manifest)} instances evaluated, exact optimal distance hit: '+str(sum(x['hit_exact_optimal'] for x in result)))
    print('Distinct proven-optimal cycles found (sum):',sum(x['distinct_proven_optimal_found'] for x in result))
    print('Output:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();score(Path(__file__).resolve().parent,Path(args.predictions),Path(args.out))
