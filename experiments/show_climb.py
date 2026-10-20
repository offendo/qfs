import json
import sys

r = json.load(open(sys.argv[1]))
names = []
for k in r:
    n = k.split("-", 1)[1].split("@")[0]
    if n not in names:
        names.append(n)
for c in (2, 4, 6, 8):
    print(f"\nceiling {c}: best height reached - parent after 4 / 8 / 16 oracle calls  |  summed reward @16  |  share of parents with some success @16")
    print(f"{'rule':14s} {'free walk':>26s} {'summed':>8s} {'hit':>6s} | {'best of k (independent)':>26s} {'hit':>6s}")
    for n in names:
        w, b = r.get(f"climb-{n}@{c}"), r.get(f"bestof-{n}@{c}")
        if not w:
            continue
        fw = " / ".join(f"{w['gain'][i]:+.3f}" for i in (3, 7, 15))
        bo = " / ".join(f"{b['gain'][i]:+.3f}" for i in (3, 7, 15)) if b else ""
        print(f"{n:14s} {fw:>26s} {w['total'][15]:8.3f} {100 * w['hit'][15]:5.1f}% | {bo:>26s} {100 * b['hit'][15] if b else 0:5.1f}%")
