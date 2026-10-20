import re, sys
import numpy as np

R = "/home/npatel37/.claude/jobs/7b4859dc/tmp/rl/"
pat = re.compile(r"iter (\d+)/\d+ .*?child ([\d.]+) \(parent ([\d.]+)\).*?hamming ([\d.]+)  copied ([\d.]+)%  repeated ([\d.]+)%  new>=6 (\d+)  new>=8 (\d+)  best (\d+)")
for name in sys.argv[1:]:
    rows = [pat.search(l) for l in open(R + name + ".log") if "iter " in l]
    rows = np.array([[float(x) for x in m.groups()] for m in rows if m])
    n = len(rows)
    print(f"== {name}: {n} iterations")
    print(f"total new>=6 {int(rows[:, 6].sum())}  new>=8 {int(rows[:, 7].sum())}  best {int(rows[:, 8].max())}")
    for a, b in ((0, n // 4), (n // 4, n // 2), (n // 2, 3 * n // 4), (3 * n // 4, n)):
        r = rows[a:b]
        print(f"  iters {a+1}-{b}: child {r[:,1].mean():.2f} parent {r[:,2].mean():.2f} hamming {r[:,3].mean():.1f} repeated {r[:,5].mean():.1f}% new>=6/iter {r[:,6].mean():.1f} new>=8/iter {r[:,7].mean():.2f}")
