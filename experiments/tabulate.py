"""Markdown tables from the JSON written by experiments/evaluate.py.

    uv run python experiments/tabulate.py results.json [results2.json ...]

Later files override earlier ones for the same method and ceiling, so an extra evaluation can be added to a table.
"""

import json
import sys

results = {}
for path in sys.argv[1:]:
    results.update(json.load(open(path)))
methods = list(dict.fromkeys(k.split("@")[0] for k in results))
ceilings = sorted({int(k.split("@")[1]) for k in results})


def table(title: str, cell) -> None:
    print(f"\n**{title}**\n")
    print("| method | " + " | ".join(f"ceiling {c}" for c in ceilings) + " |")
    print("|---|" + "---|" * len(ceilings))
    for m in methods:
        print(f"| {m} | " + " | ".join(cell(results[f"{m}@{c}"]) if f"{m}@{c}" in results else "" for c in ceilings) + " |")


table("improvement = child score - parent score (higher is better)", lambda r: f"{r['improvement']:+.3f}")
table("share of children at least as high as their parent (up + same, %)", lambda r: f"{100 * (r['up'] + r['same']):.1f}")
table("mean child score (random edits give about 1.2-1.45 whatever the parent)", lambda r: f"{r['child']:.2f}")
table("mean Hamming distance from the parent", lambda r: f"{r['hamming']:.2f}")
print("\nmean parent score at each ceiling: " + ", ".join(f"{c}: {results[f'{methods[0]}@{c}']['parent']:.2f}" for c in ceilings))

# keeps-the-height rate for each exact parent height, in the highest ceiling (which contains every lower height)
top = ceilings[-1]
if all("by_parent" in results.get(f"{m}@{top}", {}) for m in methods):
    heights = sorted({int(h) for m in methods for h in results[f"{m}@{top}"]["by_parent"]})
    key = "retain_nocopy" if all("retain_nocopy" in results[f"{m}@{top}"]["by_parent"][h] for m in methods for h in results[f"{m}@{top}"]["by_parent"]) else "retain"
    print(f"\n**share of children at least as high as their parent, by exact parent height (ceiling {top} pool; "
          f"{'exact copies are not counted' if key == 'retain_nocopy' else 'exact copies are counted'})**\n")
    print("| method | " + " | ".join(f"parent {h}" for h in heights) + " |")
    print("|---|" + "---|" * len(heights))
    for m in methods:
        bp = results[f"{m}@{top}"]["by_parent"]
        print(f"| {m} | " + " | ".join(f"{100 * bp[str(h)][key]:.1f}" if str(h) in bp else "" for h in heights) + " |")
    print("\nparents per height: " + ", ".join(f"{h}: {results[f'{methods[0]}@{top}']['by_parent'].get(str(h), {}).get('n', 0)}" for h in heights))

