#!/usr/bin/env python3
"""Independently verify exact length certificates and distinct witness tours."""
import csv
import hashlib
import json
import math
from pathlib import Path


def canonical(route):
    i=route.index(min(route))
    r=route[i:]+route[:i]
    rev=[r[0]]+r[:0:-1]
    return tuple(r if r<rev else rev)


def verify(root):
    with (root/'manifest.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==138, len(rows)
    seen_coords=set()
    count_witness=0;worst_n=0
    by_tier={}
    for r in rows:
        p=root/r['file'];data=p.read_bytes()
        assert hashlib.sha256(data).hexdigest()==r['sha256'],r['name']
        coords=[tuple(map(int,line.split())) for line in data.decode().splitlines() if line.strip()]
        n=int(r['n']);h=int(r['unit_edge']);L=n*h
        assert len(coords)==n and len(set(coords))==n and L==int(r['exact_optimal_length'])
        # Verify all differences are orthogonal-lattice differences:
        # [a -b; b a] (dx,dy), reflection allowed, divided by a*a+b*b.
        a=int(r['rotation_a']);b=int(r['rotation_b']);mir=-1 if int(r['reflected_y']) else 1
        base=coords[0];det=a*a+b*b
        assert h*h==det
        lattice=[]
        for x,y in coords:
            dx=x-base[0];dy=mir*(y-base[1])
            q=a*dx+b*dy;w=-b*dx+a*dy
            assert q%det==0 and w%det==0
            lattice.append((q//det,w//det))
        assert len(set(lattice))==n
        # Integer lattice has minimum nonzero distance h, proving d_ij>=h.
        proof=json.loads((root/r['proof_file']).read_text(encoding='utf-8'))
        assert proof['known_exact_optimal_length']==L
        witness=proof['optimal_tours'];canonical_set=set()
        for route in witness:
            assert len(route)==n and set(route)==set(range(n))
            assert all(sum((coords[route[i]][z]-coords[route[(i+1)%n]][z])**2 for z in (0,1))==h*h for i in range(n))
            key=canonical(route);assert key not in canonical_set
            canonical_set.add(key)
        assert len(witness)==int(r['certified_optimal_witness_count'])
        assert int(r['grid_cycle_rank'])>=1
        sorted_lattice=tuple(sorted(lattice))
        # No exact duplicate shapes under translations in lattice coordinates.
        minx=min(x for x,y in sorted_lattice);miny=min(y for x,y in sorted_lattice)
        shape=(n,tuple(sorted((x-minx,y-miny) for x,y in sorted_lattice)))
        assert shape not in seen_coords, f'duplicate normalized dataset {r["name"]}'
        seen_coords.add(shape)
        count_witness+=len(witness);worst_n=max(n,worst_n)
        by_tier[r['tier']]=by_tier.get(r['tier'],0)+1
    print('VERIFIED:',len(rows),'instances,',count_witness,'certified distinct optimum witnesses, max N=',worst_n)
    print('Tier distribution:',by_tier)


if __name__=='__main__':verify(Path(__file__).resolve().parent)
