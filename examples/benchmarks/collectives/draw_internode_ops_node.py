import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = [
    {"op": "all_reduce", "nodes": 1, "throughput": 127.57},
    {"op": "all_reduce", "nodes": 2, "throughput": 98.92},
    {"op": "all_reduce", "nodes": 4, "throughput": 100.25},
    {"op": "all_gather", "nodes": 1, "throughput": 158.58},
    {"op": "all_gather", "nodes": 2, "throughput": 100.53},
    {"op": "all_gather", "nodes": 4, "throughput": 103.16},
    {"op": "reduce_scatter", "nodes": 1, "throughput": 103.84},
    {"op": "reduce_scatter", "nodes": 2, "throughput": 89.49},
    {"op": "reduce_scatter", "nodes": 4, "throughput": 89.82},
    {"op": "all_to_all", "nodes": 1, "throughput": 27.64},
    {"op": "all_to_all", "nodes": 2, "throughput": 8.67},
    {"op": "all_to_all", "nodes": 4, "throughput": 4.29},
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
ax.set_title("Inter-node scaling of collective ops\n(8 MB/rank · 32 ranks/node · Trn1nx32)")
ax.set_xlabel("Number of nodes")
ax.set_ylabel("Bus bandwidth (GB/s)")

ax.legend(loc="upper right", frameon=True, framealpha=1)
ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4], labels=["1", "2", "4"])

plt.savefig(__file__.replace(".py", ".png"))
