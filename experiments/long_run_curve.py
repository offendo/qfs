import re
import numpy as np

pat = re.compile(r"iter (\d+)/\d+ .*?gain ([+-][\d.]+)  repeats ([\d.]+)%  success ([\d.]+)% \(step 1: ([\d.]+)%\)  final ([\d.]+) \(parent ([\d.]+)\).*?kl ([-\d.]+)  ceil (\d+)  hit (\d+)%")
rows = np.array([[float(x) for x in m.groups()] for m in (pat.search(l) for l in open("/home/npatel37/.claude/jobs/7b4859dc/tmp/climb/long40.log")) if m])
n = len(rows)
print(n, "iterations")
print("iter range      gain   success  step1  repeats  final-parent  kl      ceiling  group-hit")
for a in range(0, n, n // 10):
    r = rows[a:a + n // 10]
    print(f"{int(r[0,0]):5d}-{int(r[-1,0]):5d}  {r[:,1].mean():+.3f}  {r[:,3].mean():6.1f}%  {r[:,4].mean():4.1f}%  {r[:,2].mean():5.2f}%  {(r[:,5]-r[:,6]).mean():+.3f}       {r[:,7].mean():.4f}  {r[:,8].mean():.1f}      {r[:,9].mean():.0f}%")
