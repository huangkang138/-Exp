#!/usr/bin/env python3
"""Deterministic, certified multi-optimum integer-EUC MSTSP test suite.
Each 2x2 block of lattice nodes has a 4-edge unit cycle. Replacing shared side
edges between tiles along a spanning tree connects these cycles into one
Hamiltonian cycle with only lattice-neighbor edges. Different spanning trees
produce distinct optimum cycles. Every nonzero lattice distance is >= unit.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from pathlib import Path

SIZES = [32,36,40,48,60,64,72,80,96,100,120,128,144,160,192,196,240,256,320,400,512,640,800]
TYPES = ['rectangle','strip','compact','elongated','ring','rooms']
VECTORS = [(1,0),(3,4),(5,12),(8,15),(7,24),(20,21)]
WITNESSES = 16
SEED = 20261015


def normalized(tile):
    x0=min(x for x,y in tile); y0=min(y for x,y in tile)
    return frozenset((x-x0,y-y0) for x,y in tile)


def adjacencies(cells):
    s=set(cells)
    return [(u,v) for u in sorted(s) for v in [(u[0]+1,u[1]),(u[0],u[1]+1)] if v in s]


def rank(cells):
    return len(adjacencies(cells)) - len(cells) + 1


def add_frontier(shape, target, rng, mode='compact', forbidden=frozenset()):
    s=set(shape)
    while len(s)<target:
        f=set()
        for x,y in s:
            for t in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
                if t not in s and t not in forbidden: f.add(t)
        if not f: raise ValueError('no frontier')
        def weight(t):
            x,y=t
            n=sum((x+dx,y+dy) in s for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])
            if mode=='compact':return n**3 * (1/(1+.02*(x*x+y*y)))
            if mode=='elongated':return n**2 * (1/(1+.18*(y*y))) * (1+.03*abs(x))
            if mode=='ring':return n**2 * (1+.04*(abs(x)+abs(y)))
            if mode=='rooms':return n**2 * (4 if x<=2 or x>=12 else .3)
            return n*n
        arr=sorted(f); w=[max(.0001,weight(t)) for t in arr]
        s.add(rng.choices(arr,weights=w,k=1)[0])
    return s


def make_shape(m, kind, rng):
    assert m>=8
    if kind=='rectangle':
        opts=[(w,m//w) for w in range(2,math.isqrt(m)+1) if m%w==0 and m//w>=2]
        w,h=rng.choice(opts)
        if rng.randrange(2):w,h=h,w
        s={(x,y) for x in range(w) for y in range(h)}
    elif kind=='strip':
        w=rng.choice([2,3]) if m>=12 else 2
        h=max(2,(m-2)//w)
        s={(x,y) for x in range(w) for y in range(h)}
        s=add_frontier(s,m,rng,'elongated')
    elif kind=='compact':
        s=add_frontier({(0,0),(0,1),(1,0),(1,1)},m,rng,'compact')
    elif kind=='elongated':
        s=add_frontier({(0,0),(0,1),(1,0),(1,1)},m,rng,'elongated')
    elif kind=='ring':
        # A permanent central missing tile (or 3x3 missing patch), with a complete surrounding cycle.
        if m>=48:
            base={(x,y) for x in range(7) for y in range(7) if not(2<=x<=4 and 2<=y<=4)}
            hole={(x,y) for x in range(2,5) for y in range(2,5)}
        elif m>=16:
            base={(x,y) for x in range(5) for y in range(5) if not(1<=x<=3 and 1<=y<=3)}
            hole={(x,y) for x in range(1,4) for y in range(1,4)}
        else:
            base={(x,y) for x in range(3) for y in range(3) if (x,y)!=(1,1)}
            hole={(1,1)}
        s=add_frontier(base,m,rng,'ring',frozenset(hole))
    elif kind=='rooms':
        if m<16:
            s=add_frontier({(0,0),(0,1),(1,0),(1,1)},m,rng,'compact')
        else:
            # Two 2x2 rooms, narrow (one tile) connector, then irregular growth in the rooms.
            corridor=min(6,max(2,m//18))
            far=3+corridor
            s={(x,y) for x in (0,1) for y in (0,1)}
            s |= {(x,0) for x in range(2,far)}
            s |= {(x,y) for x in (far,far+1) for y in (0,1)}
            s=add_frontier(s,m,rng,'rooms')
    else:raise ValueError(kind)
    assert len(s)==m
    return normalized(s)


def random_tree(cells, rng):
    arr=adjacencies(cells)
    # Uniform-ish randomized Kruskal, randomness strictly independent of train/model.
    rng.shuffle(arr)
    par={v:v for v in cells}; sz={v:1 for v in cells}
    def find(a):
        while par[a]!=a:par[a]=par[par[a]];a=par[a]
        return a
    out=[]
    for u,v in arr:
        ru,rv=find(u),find(v)
        if ru==rv:continue
        if sz[ru]<sz[rv]:ru,rv=rv,ru
        par[rv]=ru;sz[ru]+=sz[rv];out.append((u,v))
        if len(out)==len(cells)-1:break
    if len(out)!=len(cells)-1:raise ValueError('disconnected')
    return out


def tour_from_tree(cells, tree):
    s=set(cells)
    edges=set()
    def edge(u,v):return tuple(sorted((u,v)))
    for bx,by in s:
        a=(2*bx,2*by);b=(2*bx+1,2*by);c=(2*bx+1,2*by+1);d=(2*bx,2*by+1)
        edges.update((edge(a,b),edge(b,c),edge(c,d),edge(d,a)))
    for u,v in tree:
        ux,uy=u;vx,vy=v
        if uy==vy:
            if ux>vx:u,v=v,u;ux,uy=u;vx,vy=v
            left1=(2*ux+1,2*uy);left2=(2*ux+1,2*uy+1)
            right1=(2*vx,2*vy);right2=(2*vx,2*vy+1)
            edges.remove(edge(left1,left2));edges.remove(edge(right1,right2))
            edges.add(edge(left1,right1));edges.add(edge(left2,right2))
        else:
            if uy>vy:u,v=v,u;ux,uy=u;vx,vy=v
            low1=(2*ux,2*uy+1);low2=(2*ux+1,2*uy+1)
            high1=(2*vx,2*vy);high2=(2*vx+1,2*vy)
            edges.remove(edge(low1,low2));edges.remove(edge(high1,high2))
            edges.add(edge(low1,high1));edges.add(edge(low2,high2))
    nb=defaultdict(list)
    for u,v in edges:nb[u].append(v);nb[v].append(u)
    assert len(nb)==4*len(s) and all(len(v)==2 for v in nb.values())
    start=min(nb); route=[start];prev=None;cur=start
    for _ in range(len(nb)-1):
        cand=[z for z in nb[cur] if z!=prev]
        assert cand
        nxt=min(cand)
        assert nxt not in route
        route.append(nxt);prev,cur=cur,nxt
    assert start in nb[cur] and len(route)==4*len(s)
    return route


def canonical(route):
    i=route.index(min(route));r=route[i:]+route[:i]
    return tuple(min(r, [r[0]]+list(reversed(r[1:]))))


def write_instance(root, n, kind, serial, rng, known_signatures):
    m=n//4
    shape=None
    for _ in range(150):
        proposal=make_shape(m,kind,rng)
        # Do not duplicate the same tile arrangement at this size (up to translation).
        sig=(n,proposal)
        if sig not in known_signatures and rank(proposal)>=1:
            shape=proposal;known_signatures.add(sig);break
    if shape is None:raise RuntimeError(f'could not obtain distinct {n} {kind}')
    nodes=[(2*bx+dx,2*by+dy) for bx,by in sorted(shape) for dx,dy in [(0,0),(1,0),(0,1),(1,1)]]
    assert len(set(nodes))==n
    node_order=nodes[:];rng.shuffle(node_order)
    inds={pt:i for i,pt in enumerate(node_order)}
    a,b=rng.choice(VECTORS);scale=math.isqrt(a*a+b*b)
    assert scale*scale==a*a+b*b
    # randomly include reflection, rotation already supplied by (a,b).
    mir=-1 if rng.randrange(2) else 1
    coords=[(a*x-b*y, mir*(b*x+a*y)) for x,y in node_order]
    minx=min(x for x,y in coords);miny=min(y for x,y in coords)
    offx=rng.randrange(500,3000);offy=rng.randrange(500,3000)
    coords=[(x-minx+offx,y-miny+offy) for x,y in coords]
    name=f'mstsp_n{n:03d}_{kind}_{serial:02d}'
    level='core_032_128' if n<=128 else ('advanced_144_400' if n<=400 else 'stress_512_800')
    rel=Path('test')/level/f'n{n:03d}'/(name+'.tsp')
    path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(f'{x} {y}\n' for x,y in coords),encoding='utf-8')
    opt=n*scale
    target_witnesses=8 if n<64 else WITNESSES
    tours={}
    for i in range(360):
        rt=tour_from_tree(shape,random_tree(shape,rng))
        ix=[inds[p] for p in rt];canon=canonical(ix)
        assert len(canon)==n and len(set(canon))==n
        assert all(sum((coords[canon[j]][z]-coords[canon[(j+1)%n]][z])**2 for z in range(2))==scale*scale for j in range(n))
        tours[canon]=1
        if len(tours)>=target_witnesses:break
    if len(tours)<target_witnesses:raise RuntimeError(f'insufficient distinct witnesses n={n} {kind}: {len(tours)}')
    # Store 0-index tour arrays; do not name them complete optimal solutions.
    proof={'instance':name,'proof_kind':'nearest_lattice_step_lower_bound_and_spanning_tree_cycles',
           'city_count':n,'known_exact_optimal_length':opt,
           'unit_edge_length':scale,'verified_distinct_optimal_tours_lower_bound':len(tours),
           'proof_explanation':'All coordinates are unique sites of an orthogonal integer lattice with nearest-neighbor distance h; rounded Euclidean d_ij>=h for i!=j. Every Hamiltonian tour has N edges, hence length>=N*h. Each stored tour visits all cities once and uses only edges of length h, so attains N*h and is globally optimal. Tour list is a certified SUBSET, not the complete optimal set.',
           'tour_indexing':'zero_based, ordered city rows of the matching .tsp file; closure implicit',
           'optimal_tours':[list(t) for t in sorted(tours)[:target_witnesses]]}
    proof_rel=Path('proofs')/(name+'.json');(root/proof_rel).parent.mkdir(parents=True,exist_ok=True)
    (root/proof_rel).write_text(json.dumps(proof,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    return {'name':name,'split':'test','tier':level,'file':rel.as_posix(), 'proof_file':proof_rel.as_posix(),
            'n':n,'shape':kind,'tile_count':m,'grid_cycle_rank':rank(shape),
            'rotation_a':a,'rotation_b':b,'reflected_y':int(mir<0),
            'unit_edge':scale,'exact_optimal_length':opt,
            'certified_optimal_witness_count':len(tours),'complete_optimal_set_known':'no',
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    root=Path(__file__).resolve().parent
    rng=random.Random(SEED)
    manifest=[];seen=set()
    for n in SIZES:
        for kind in TYPES:
            row=write_instance(root,n,kind,1,rng,seen);manifest.append(row)
        print('generated n',n,flush=True)
    with (root/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
    cfg={'dataset_name':'benachmark_STSP_Complex_Multiopt_Test','generation_seed':SEED,'sizes':SIZES,'shapes':TYPES,'instances':len(manifest),'certified_optimal_witnesses_per_instance':'8 for n<64, 16 for n>=64','algorithm':'2x2-tile spanning-tree splicing on rotated integer lattice','test_only':True}
    (root/'dataset_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('DONE',len(manifest), 'witnesses',len(manifest)*WITNESSES,flush=True)

if __name__=='__main__':main()
