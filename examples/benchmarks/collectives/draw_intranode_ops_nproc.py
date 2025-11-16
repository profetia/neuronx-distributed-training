import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [
    {"op": "all_reduce", "nodes": 2, "throughput": 56.80},
    {"op": "all_reduce", "nodes": 8, "throughput": 66.76},
    {"op": "all_reduce", "nodes": 16, "throughput": 68.39},
    {"op": "all_reduce", "nodes": 32, "throughput": 127.57},
    {"op": "all_gather", "nodes": 2, "throughput": 55.22},
    {"op": "all_gather", "nodes": 8, "throughput": 75.03},
    {"op": "all_gather", "nodes": 16, "throughput": 76.35},
    {"op": "all_gather", "nodes": 32, "throughput": 158.58},
    {"op": "reduce_scatter", "nodes": 2, "throughput": 37.87},
    {"op": "reduce_scatter", "nodes": 8, "throughput": 55.37},
    {"op": "reduce_scatter", "nodes": 16, "throughput": 57.52},
    {"op": "reduce_scatter", "nodes": 32, "throughput": 103.84},
    {"op": "all_to_all", "nodes": 2, "throughput": 69.23},
    {"op": "all_to_all", "nodes": 8, "throughput": 23.91},
    {"op": "all_to_all", "nodes": 16, "throughput": 29.15},
    {"op": "all_to_all", "nodes": 32, "throughput": 27.64},
]

df = pd.DataFrame(data)

rc = {
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.8,
    "grid.color": "#d0d0d0",
    "font.size": 9,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
}
sns.set_theme(context="notebook", style="whitegrid", rc=rc)
ax = sns.lineplot(
    data=df, x="nodes", y="throughput", 
    hue="op", marker="o", linewidth=3.0, markersize=9
)
ax.set_title("Intra-node scaling of collective ops\n(8 MB/rank · Trn1nx32)")
ax.set_xlabel("Number of ranks")
ax.set_ylabel("Bus bandwidth (GB/s)")

ax.legend(loc="upper left", frameon=True, framealpha=1)
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8, 16, 32], labels=["2", "4", "8", "16", "32"])

plt.savefig(__file__.replace(".py", ".png"))
