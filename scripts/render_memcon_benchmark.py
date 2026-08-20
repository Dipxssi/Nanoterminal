from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

Path("docs").mkdir(exist_ok=True)

rows = [
    ("ls -la", "RETRIEVE:shallow", "+222"),
    ("pwd", "RETRIEVE:medium", "+222"),
    ("git status", "RETRIEVE:deep", "+206"),
    ("how is python packaging configured?", "RETRIEVE:shallow", "+206"),
    ("poetry run pytest", "RETRIEVE:medium", "+190"),
    ("fix test failures with poetry", "RETRIEVE:deep", "+222"),
    ("git commit -m 'fix'", "RETRIEVE:shallow", "+222"),
    ("git push origin main", "RETRIEVE:medium", "+222"),
]

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=160)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#0b1220")
ax.set_facecolor("#0b1220")

ax.text(
    0.5,
    0.955,
    "MemCon vs. Naive Retrieval Benchmark",
    ha="center",
    va="top",
    color="#e8eefc",
    fontsize=16,
    fontweight="bold",
    fontfamily="DejaVu Sans",
)

header = FancyBboxPatch(
    (0.03, 0.82),
    0.94,
    0.08,
    boxstyle="round,pad=0.01,rounding_size=0.01",
    linewidth=0,
    facecolor="#152238",
)
ax.add_patch(header)
cols = [
    (0.05, "Turn / Prompt", "left"),
    (0.48, "Naive Strategy", "left"),
    (0.68, "MemCon Action", "left"),
    (0.90, "Tokens Saved", "center"),
]
for x, label, ha in cols:
    ax.text(
        x,
        0.86,
        label,
        color="#9fb2d9",
        fontsize=9,
        fontweight="bold",
        va="center",
        ha=ha,
        fontfamily="DejaVu Sans",
    )

y = 0.76
for i, (prompt, action, saved) in enumerate(rows):
    bg = "#101a2e" if i % 2 == 0 else "#0d1628"
    ax.add_patch(
        FancyBboxPatch(
            (0.03, y - 0.035),
            0.94,
            0.07,
            boxstyle="square,pad=0",
            linewidth=0,
            facecolor=bg,
        )
    )
    ax.text(0.05, y, prompt, color="#d7e2f7", fontsize=8.5, va="center")
    ax.text(0.48, y, "RETRIEVE (k=5)", color="#ff6b6b", fontsize=8.5, va="center")
    ax.text(0.68, y, action, color="#3dd68c", fontsize=8.5, va="center")
    ax.text(0.90, y, saved, color="#f0b429", fontsize=8.5, va="center", ha="center")
    y -= 0.072

ax.add_patch(
    FancyBboxPatch(
        (0.03, 0.04),
        0.29,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0,
        facecolor="#152238",
    )
)
ax.add_patch(
    FancyBboxPatch(
        (0.355, 0.04),
        0.29,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0,
        facecolor="#152238",
    )
)
ax.add_patch(
    FancyBboxPatch(
        (0.68, 0.04),
        0.29,
        0.12,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0,
        facecolor="#123528",
    )
)

ax.text(0.175, 0.125, "Total Naive Tokens", ha="center", color="#9fb2d9", fontsize=8)
ax.text(0.175, 0.07, "2000", ha="center", color="#ff6b6b", fontsize=14, fontweight="bold")
ax.text(0.5, 0.125, "Total MemCon Tokens", ha="center", color="#9fb2d9", fontsize=8)
ax.text(0.5, 0.07, "288", ha="center", color="#3dd68c", fontsize=14, fontweight="bold")
ax.text(0.825, 0.125, "Token Reduction", ha="center", color="#9fb2d9", fontsize=8)
ax.text(0.825, 0.07, "85.6%", ha="center", color="#5eead4", fontsize=14, fontweight="bold")

fig.tight_layout(pad=0.4)
out = Path("docs/memcon-benchmark.png")
fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
print(f"wrote {out.resolve()} ({out.stat().st_size} bytes)")
