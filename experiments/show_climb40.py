import json
import sys

r = json.load(open(sys.argv[1]))
names = []
for k in r:
    n = k.split("-", 1)[1].split("@")[0]
    if n not in names:
        names.append(n)
idx = (7, 15, 39)
for c in (2, 4, 6, 8):
    print(f"\nceiling {c}: best height reached - parent after 8 / 16 / 40 calls | hit@40 | summed reward@40 || best-of-k (independent) 8 / 16 / 40, hit@40")
    for n in names:
        w, b = r.get(f"climb-{n}@{c}"), r.get(f"bestof-{n}@{c}")
        if not w:
            continue
        fw = " / ".join(f"{w['gain'][i]:+.3f}" for i in idx)
        bo = (" / ".join(f"{b['gain'][i]:+.3f}" for i in idx) + f"  {100 * b['hit'][39]:5.1f}%") if b else ""
        print(f"{n:16s} {fw}  {100 * w['hit'][39]:5.1f}%  {w['total'][39]:7.3f} || {bo}")
