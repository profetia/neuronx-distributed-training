# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from collections import defaultdict

# # -----------------------------
# # Input data
# # -----------------------------
# data_a = [
#     # Weak scaling, all gather
#     {"num_nodes": 1, "data_size_bytes": 1 * 1024 * 1024, "bus_bw_gbps": 24.98},
#     {"num_nodes": 2, "data_size_bytes": 1 * 1024 * 1024, "bus_bw_gbps": 1.74},
#     {"num_nodes": 4, "data_size_bytes": 1 * 1024 * 1024, "bus_bw_gbps": 1.14},
#     {"num_nodes": 8, "data_size_bytes": 1 * 1024 * 1024, "bus_bw_gbps": 1.64},
#     {"num_nodes": 16, "data_size_bytes": 1 * 1024 * 1024, "bus_bw_gbps": 0.52},

#     {"num_nodes": 1, "data_size_bytes": 32 * 1024 * 1024, "bus_bw_gbps": 130.89},
#     {"num_nodes": 2, "data_size_bytes": 32 * 1024 * 1024, "bus_bw_gbps": 38.29},
#     {"num_nodes": 4, "data_size_bytes": 32 * 1024 * 1024, "bus_bw_gbps": 55.80},
#     {"num_nodes": 8, "data_size_bytes": 32 * 1024 * 1024, "bus_bw_gbps": 38.24},
#     {"num_nodes": 16, "data_size_bytes": 32 * 1024 * 1024, "bus_bw_gbps": 29.99},

#     {"num_nodes": 1, "data_size_bytes": 512 * 1024 * 1024, "bus_bw_gbps": 158.77},
#     {"num_nodes": 2, "data_size_bytes": 512 * 1024 * 1024, "bus_bw_gbps": 99.87},
#     {"num_nodes": 4, "data_size_bytes": 512 * 1024 * 1024, "bus_bw_gbps": 98.06},
#     {"num_nodes": 8, "data_size_bytes": 512 * 1024 * 1024, "bus_bw_gbps": 89.39},
#     {"num_nodes": 16, "data_size_bytes": 512 * 1024 * 1024, "bus_bw_gbps": 60.99},
# ]

# # -----------------------------
# # Styling (kept close to your script)
# # -----------------------------
# plt.rcParams.update(
#     {
#         "font.size": 31,
#         "font.family": plt.rcParams["font.sans-serif"][0],
#         "axes.linewidth": 2.5,
#         "xtick.major.width": 2.5,
#         "ytick.major.width": 2.5,
#     }
# )

# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
#     }
# )

# # -----------------------------
# # Helpers
# # -----------------------------
# def bytes_to_mib(x_bytes: int) -> float:
#     return x_bytes / (1024 * 1024)

# def fmt_size(bytes_val: int) -> str:
#     mib = bytes_to_mib(bytes_val)
#     # Nice labels for your specific sizes
#     if abs(mib - 1) < 1e-9:
#         return "1 MB"
#     if abs(mib - 32) < 1e-9:
#         return "32 MB"
#     if abs(mib - 512) < 1e-9:
#         return "512 MB"
#     # fallback
#     if mib >= 1024:
#         return f"{mib/1024:.0f} GB"
#     return f"{mib:.0f} MB"

# def label_points(ax, xs, ys, fmt="{:.2f}", dy=12):
#     for xi, yi in zip(xs, ys):
#         ax.annotate(
#             fmt.format(yi),
#             (xi, yi),
#             textcoords="offset points",
#             xytext=(0, dy),
#             ha="center",
#             va="bottom",
#             clip_on=False,
#             fontsize=28,  # slightly smaller than axis font; adjust as desired
#         )

# # -----------------------------
# # Group data by data_size_bytes
# # -----------------------------
# by_size = defaultdict(list)
# for row in data_a:
#     by_size[row["data_size_bytes"]].append(row)

# sizes_sorted = sorted(by_size.keys())
# x_all = sorted({row["num_nodes"] for row in data_a})

# # -----------------------------
# # Plot
# # -----------------------------
# fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

# # If you want deterministic marker choices per line:

# cmap = [
#     "#3e468a",
#     "#f06411",
#     "#ffc20b",
# ]
# ticker = [
#     "o", 
#     "^", 
#     "s"
# ]
# for i, sz in enumerate(sizes_sorted):
#     rows = sorted(by_size[sz], key=lambda r: r["num_nodes"])
#     x = [r["num_nodes"] for r in rows]
#     y = [r["bus_bw_gbps"] for r in rows]

#     ax.plot(
#         x,
#         y,
#         label=fmt_size(sz),
#         linewidth=5,
#         marker=ticker[i % len(ticker)],
#         markersize=25,
#         color=cmap[i % len(cmap)],
#     )
#     # Optional point labels (uncomment if you want them)
#     # label_points(ax, x, y, fmt="{:.2f}", dy=12)

# ax.set_xscale("log", base=2)
# ax.set_xticks(x_all)
# ax.set_xticklabels([str(v) for v in x_all], fontweight="bold", fontsize=35)

# # Optional: fixed y-ticks (comment out if you prefer auto ticks)
# yticks = [0, 40, 80, 120, 160]
# ax.set_yticks(yticks)
# ax.set_yticklabels([str(v) for v in yticks], fontweight="bold", fontsize=35)
# ax.set_ylim(None, 180)

# ax.tick_params(axis="y", length=10, pad=10)
# ax.tick_params(axis="x", length=10, pad=10)

# ax.set_xlabel("Number of Nodes", fontsize=39, fontweight="bold")
# ax.set_ylabel("Bus Bandwidth (GB/s)", fontsize=39, fontweight="bold")

# ax.legend(loc="upper right", prop={"weight": "bold"})
# ax.grid(True, linestyle="-", linewidth=2.5, alpha=0.7)

# plt.tight_layout()
# # plt.show()

# fig.savefig(
#     "AppendixAllGatherInterNodeWeakScaling.pdf",
#     bbox_inches="tight",
#     dpi=300,
# )

import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

# -----------------------------
# Input data
# -----------------------------
data_a = [
    # Strong scaling, all gather
    {"num_nodes": 1, "data_size_bytes": 4 * 1024 * 1024, "bus_bw_gbps": 12.11}, # 128K
    {"num_nodes": 2, "data_size_bytes": 4 * 1024 * 1024, "bus_bw_gbps": 0.07}, # 64K
    {"num_nodes": 4, "data_size_bytes": 4 * 1024 * 1024, "bus_bw_gbps": 0.04}, # 32K
    {"num_nodes": 8, "data_size_bytes": 4 * 1024 * 1024, "bus_bw_gbps": 0.03}, # 16K
    {"num_nodes": 16, "data_size_bytes": 4 * 1024 * 1024, "bus_bw_gbps": 0.00}, # 8K

    {"num_nodes": 1, "data_size_bytes": 128 * 1024 * 1024, "bus_bw_gbps": 27.08}, # 4M
    {"num_nodes": 2, "data_size_bytes": 128 * 1024 * 1024, "bus_bw_gbps": 7.29}, # 2M
    {"num_nodes": 4, "data_size_bytes": 128 * 1024 * 1024, "bus_bw_gbps": 1.14}, # 1M
    {"num_nodes": 8, "data_size_bytes": 128 * 1024 * 1024, "bus_bw_gbps": 1.20}, # 512K
    {"num_nodes": 16, "data_size_bytes": 128 * 1024 * 1024, "bus_bw_gbps": 0.14}, # 256K

    {"num_nodes": 1, "data_size_bytes": 2048 * 1024 * 1024, "bus_bw_gbps": 150.40}, # 64M
    {"num_nodes": 2, "data_size_bytes": 2048 * 1024 * 1024, "bus_bw_gbps": 38.29}, # 32M
    {"num_nodes": 4, "data_size_bytes": 2048 * 1024 * 1024, "bus_bw_gbps": 33.79}, # 16M
    {"num_nodes": 8, "data_size_bytes": 2048 * 1024 * 1024, "bus_bw_gbps": 9.27}, # 8M
    {"num_nodes": 16, "data_size_bytes": 2048 * 1024 * 1024, "bus_bw_gbps": 2.04}, # 4M
]

# -----------------------------
# Styling (kept close to your script)
# -----------------------------
plt.rcParams.update(
    {
        "font.size": 31,
        "font.family": plt.rcParams["font.sans-serif"][0],
        "axes.linewidth": 2.5,
        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
    }
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    }
)

# -----------------------------
# Helpers
# -----------------------------
def bytes_to_mib(x_bytes: int) -> float:
    return x_bytes / (1024 * 1024)

def fmt_size(bytes_val: int) -> str:
    mib = bytes_to_mib(bytes_val)
    # Nice labels for your specific sizes
    if abs(mib - 1) < 1e-9:
        return "1 MB"
    if abs(mib - 32) < 1e-9:
        return "32 MB"
    if abs(mib - 512) < 1e-9:
        return "512 MiB"
    # fallback
    if mib >= 1024:
        return f"{mib/1024:.0f} GB"
    return f"{mib:.0f} MB"

def label_points(ax, xs, ys, fmt="{:.2f}", dy=12):
    for xi, yi in zip(xs, ys):
        ax.annotate(
            fmt.format(yi),
            (xi, yi),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            va="bottom",
            clip_on=False,
            fontsize=22,  # slightly smaller than axis font; adjust as desired
        )

# -----------------------------
# Group data by data_size_bytes
# -----------------------------
by_size = defaultdict(list)
for row in data_a:
    by_size[row["data_size_bytes"]].append(row)

sizes_sorted = sorted(by_size.keys())
x_all = sorted({row["num_nodes"] for row in data_a})

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 8), dpi=300)

# If you want deterministic marker choices per line:

cmap = [
    "#3e468a",
    "#f06411",
    "#ffc20b",
]
ticker = [
    "o", 
    "^", 
    "s"
]
for i, sz in enumerate(sizes_sorted):
    rows = sorted(by_size[sz], key=lambda r: r["num_nodes"])
    x = [r["num_nodes"] for r in rows]
    y = [r["bus_bw_gbps"] for r in rows]

    ax.plot(
        x,
        y,
        label=fmt_size(sz),
        linewidth=5,
        marker=ticker[i % len(cmap)],
        markersize=25,
        color=cmap[i % len(cmap)],
    )
    # Optional point labels (uncomment if you want them)
    # label_points(ax, x, y, fmt="{:.2f}", dy=12)

ax.set_xscale("log", base=2)
ax.set_xticks(x_all)
ax.set_xticklabels([str(v) for v in x_all], fontweight="bold", fontsize=35)

# Optional: fixed y-ticks (comment out if you prefer auto ticks)
yticks = [0, 40, 80, 120, 160]
ax.set_yticks(yticks)
ax.set_yticklabels([str(v) for v in yticks], fontweight="bold", fontsize=35)
ax.set_ylim(None, 180)

ax.tick_params(axis="y", length=10, pad=10)
ax.tick_params(axis="x", length=10, pad=10)

ax.set_xlabel("Number of Nodes", fontsize=39, fontweight="bold")
ax.set_ylabel("Bus Bandwidth (GB/s)", fontsize=39, fontweight="bold")

ax.legend(loc="upper right", prop={"weight": "bold"})
ax.grid(True, linestyle="-", linewidth=2.5, alpha=0.7)

plt.tight_layout()
plt.show()

fig.savefig(
    "AppendixAllGatherInterNodeStrongScaling.pdf",
    bbox_inches="tight",
    dpi=300,
)