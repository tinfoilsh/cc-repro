#!/usr/bin/env python3
"""Compare two guidellm concurrency-sweep dirs (CC-off baseline vs CC-on).
Usage: compare_sweep.py <cc-off-dir> <cc-on-dir>
Latency overhead% = (on-off)/off*100. Throughput loss% = (off-on)/off*100."""
import json,sys,os,glob,re
def load(run_dir):
    out={}
    for f in glob.glob(os.path.join(run_dir,"c*.json")):
        c=int(re.search(r'c(\d+)\.json',os.path.basename(f)).group(1))
        out[c]=json.load(open(f))['benchmarks'][0]['metrics']
    return out
def s(m,name,k='mean'):
    b=m[name]['successful']; return b.get('mean') if k=='mean' else (b.get('percentiles') or {}).get(k,b.get('median'))
def ov(on,off): return (on-off)/off*100 if (on is not None and off) else None
def loss(on,off): return (off-on)/off*100 if (on is not None and off) else None
def f(x,n=1): return f"{x:.{n}f}" if isinstance(x,(int,float)) else "-"
def main(off_dir,on_dir):
    off=load(off_dir); on=load(on_dir)
    print(f"\nThroughput frontier — CC overhead: {off_dir} (off) vs {on_dir} (on)")
    h=f"{'conc':>4} | {'out_tok/s off':>13} {'on':>9} {'loss%':>6} {'keeps':>6} | {'TTFTp50 off':>11} {'on':>8} {'ovhd%':>6} | {'ITL off':>7} {'on':>7} {'ovhd%':>6}"
    print(h); print("-"*len(h))
    for c in sorted(set(off)&set(on)):
        o,n=off[c],on[c]
        xo,xn=s(o,'output_tokens_per_second'),s(n,'output_tokens_per_second')
        to,tn=s(o,'time_to_first_token_ms','p50'),s(n,'time_to_first_token_ms','p50')
        io,iN=s(o,'inter_token_latency_ms'),s(n,'inter_token_latency_ms')
        keeps=f"{xn/xo:.2f}x" if (xo and xn) else "-"
        print(f"{c:>4} | {f(xo):>13} {f(xn):>9} {f(loss(xn,xo)):>6} {keeps:>6} | {f(to):>11} {f(tn):>8} {f(ov(tn,to)):>6} | {f(io):>7} {f(iN):>7} {f(ov(iN,io)):>6}")
if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "results/cc-off/run1",
         sys.argv[2] if len(sys.argv)>2 else "results/cc-on/run1")
