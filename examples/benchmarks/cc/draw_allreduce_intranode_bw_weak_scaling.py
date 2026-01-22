import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

K = 1024
M = 1024**2
G = 1024**3

# data = [

#     # 2 ranks
#     # {"num_nodes": 2, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.49},
#     # {"num_nodes": 2, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.98},
#     {"num_nodes": 2, "data_size_bytes": 4 * K,         "bus_bw_gbps": 1.95},
#     {"num_nodes": 2, "data_size_bytes": 8 * K,         "bus_bw_gbps": 3.90},
#     {"num_nodes": 2, "data_size_bytes": 16 * K,        "bus_bw_gbps": 7.71},
#     {"num_nodes": 2, "data_size_bytes": 32 * K,        "bus_bw_gbps": 12.14},
#     {"num_nodes": 2, "data_size_bytes": 64 * K,        "bus_bw_gbps": 18.59},
#     {"num_nodes": 2, "data_size_bytes": 128 * K,        "bus_bw_gbps": 25.45},
#     {"num_nodes": 2, "data_size_bytes": 256 * K,       "bus_bw_gbps": 37.05},
#     {"num_nodes": 2, "data_size_bytes": 512 * K,       "bus_bw_gbps": 47.77},
#     {"num_nodes": 2, "data_size_bytes": 1 * M,       "bus_bw_gbps": 56.00},
#     {"num_nodes": 2, "data_size_bytes": 2 * M,       "bus_bw_gbps": 60.88},
#     {"num_nodes": 2, "data_size_bytes": 4 * M,      "bus_bw_gbps": 63.53},
#     {"num_nodes": 2, "data_size_bytes": 8 * M,      "bus_bw_gbps": 63.89},
#     {"num_nodes": 2, "data_size_bytes": 16 * M,      "bus_bw_gbps": 56.66},
#     {"num_nodes": 2, "data_size_bytes": 32 * M,     "bus_bw_gbps": 57.36},
#     {"num_nodes": 2, "data_size_bytes": 64 * M,     "bus_bw_gbps": 57.41},
#     {"num_nodes": 2, "data_size_bytes": 128 * M,     "bus_bw_gbps": 57.37},
#     {"num_nodes": 2, "data_size_bytes": 256 * M,       "bus_bw_gbps": 57.13},
#     {"num_nodes": 2, "data_size_bytes": 512 * M,       "bus_bw_gbps": 56.61},
#     {"num_nodes": 2, "data_size_bytes": 1 * G,       "bus_bw_gbps": 56.52},
#     {"num_nodes": 2, "data_size_bytes": 2 * G,       "bus_bw_gbps": 56.59},

#     # 8 ranks
#     # {"num_nodes": 8, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.29},
#     # {"num_nodes": 8, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.56},
#     {"num_nodes": 8, "data_size_bytes": 4 * K,         "bus_bw_gbps": 1.05},
#     {"num_nodes": 8, "data_size_bytes": 8 * K,         "bus_bw_gbps": 2.02},
#     {"num_nodes": 8, "data_size_bytes": 16 * K,        "bus_bw_gbps": 3.82},
#     {"num_nodes": 8, "data_size_bytes": 32 * K,        "bus_bw_gbps": 6.38},
#     {"num_nodes": 8, "data_size_bytes": 64 * K,        "bus_bw_gbps": 9.91},
#     {"num_nodes": 8, "data_size_bytes": 128 * K,        "bus_bw_gbps": 13.85},
#     {"num_nodes": 8, "data_size_bytes": 256 * K,       "bus_bw_gbps": 19.46},
#     {"num_nodes": 8, "data_size_bytes": 512 * K,       "bus_bw_gbps": 23.45},
#     {"num_nodes": 8, "data_size_bytes": 1 * M,       "bus_bw_gbps": 26.11},
#     {"num_nodes": 8, "data_size_bytes": 2 * M,       "bus_bw_gbps": 42.30},
#     {"num_nodes": 8, "data_size_bytes": 4 * M,      "bus_bw_gbps": 54.97},
#     {"num_nodes": 8, "data_size_bytes": 8 * M,      "bus_bw_gbps": 60.33},
#     {"num_nodes": 8, "data_size_bytes": 16 * M,      "bus_bw_gbps": 62.96},
#     {"num_nodes": 8, "data_size_bytes": 32 * M,     "bus_bw_gbps": 65.78},
#     {"num_nodes": 8, "data_size_bytes": 64 * M,     "bus_bw_gbps": 66.76},
#     {"num_nodes": 8, "data_size_bytes": 128 * M,     "bus_bw_gbps": 66.97},
#     {"num_nodes": 8, "data_size_bytes": 256 * M,       "bus_bw_gbps": 67.04},
#     {"num_nodes": 8, "data_size_bytes": 512 * M,       "bus_bw_gbps": 67.04},
#     {"num_nodes": 8, "data_size_bytes": 1 * G,       "bus_bw_gbps": 67.06},
#     {"num_nodes": 8, "data_size_bytes": 2 * G,       "bus_bw_gbps": 67.07},

#     # 16 ranks
#     # {"num_nodes": 16, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.27},
#     # {"num_nodes": 16, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.53},
#     {"num_nodes": 16, "data_size_bytes": 4 * K,         "bus_bw_gbps": 1.01},
#     {"num_nodes": 16, "data_size_bytes": 8 * K,         "bus_bw_gbps": 1.91},
#     {"num_nodes": 16, "data_size_bytes": 16 * K,        "bus_bw_gbps": 3.78},
#     {"num_nodes": 16, "data_size_bytes": 32 * K,        "bus_bw_gbps": 6.74},
#     {"num_nodes": 16, "data_size_bytes": 64 * K,        "bus_bw_gbps": 10.55},
#     {"num_nodes": 16, "data_size_bytes": 128 * K,        "bus_bw_gbps": 15.02},
#     {"num_nodes": 16, "data_size_bytes": 256 * K,       "bus_bw_gbps": 22.49},
#     {"num_nodes": 16, "data_size_bytes": 512 * K,       "bus_bw_gbps": 29.00},
#     {"num_nodes": 16, "data_size_bytes": 1 * M,       "bus_bw_gbps": 32.64},
#     {"num_nodes": 16, "data_size_bytes": 2 * M,       "bus_bw_gbps": 33.69},
#     {"num_nodes": 16, "data_size_bytes": 4 * M,      "bus_bw_gbps": 48.13},
#     {"num_nodes": 16, "data_size_bytes": 8 * M,      "bus_bw_gbps": 62.21},
#     {"num_nodes": 16, "data_size_bytes": 16 * M,      "bus_bw_gbps": 65.55},
#     {"num_nodes": 16, "data_size_bytes": 32 * M,     "bus_bw_gbps": 67.35},
#     {"num_nodes": 16, "data_size_bytes": 64 * M,     "bus_bw_gbps": 68.13},
#     {"num_nodes": 16, "data_size_bytes": 128 * M,     "bus_bw_gbps": 68.41},
#     {"num_nodes": 16, "data_size_bytes": 256 * M,       "bus_bw_gbps": 68.52},
#     {"num_nodes": 16, "data_size_bytes": 512 * M,       "bus_bw_gbps": 68.61},
#     {"num_nodes": 16, "data_size_bytes": 1 * G,       "bus_bw_gbps": 68.75},
#     {"num_nodes": 16, "data_size_bytes": 2 * G,       "bus_bw_gbps": 68.84},

#     # 32 ranks
#     # {"num_nodes": 32, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.22},
#     # {"num_nodes": 32, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.43},
#     {"num_nodes": 32, "data_size_bytes": 4 * K,         "bus_bw_gbps": 0.83},
#     {"num_nodes": 32, "data_size_bytes": 8 * K,         "bus_bw_gbps": 1.63},
#     {"num_nodes": 32, "data_size_bytes": 16 * K,        "bus_bw_gbps": 3.13},
#     {"num_nodes": 32, "data_size_bytes": 32 * K,        "bus_bw_gbps": 5.91},
#     {"num_nodes": 32, "data_size_bytes": 64 * K,        "bus_bw_gbps": 10.19},
#     {"num_nodes": 32, "data_size_bytes": 128 * K,        "bus_bw_gbps": 15.79},
#     {"num_nodes": 32, "data_size_bytes": 256 * K,       "bus_bw_gbps": 22.48},
#     {"num_nodes": 32, "data_size_bytes": 512 * K,       "bus_bw_gbps": 28.30},
#     {"num_nodes": 32, "data_size_bytes": 1 * M,       "bus_bw_gbps": 31.80},
#     {"num_nodes": 32, "data_size_bytes": 2 * M,       "bus_bw_gbps": 33.64},
#     {"num_nodes": 32, "data_size_bytes": 4 * M,      "bus_bw_gbps": 34.14},
#     {"num_nodes": 32, "data_size_bytes": 8 * M,      "bus_bw_gbps": 52.92},
#     {"num_nodes": 32, "data_size_bytes": 16 * M,      "bus_bw_gbps": 83.02},
#     {"num_nodes": 32, "data_size_bytes": 32 * M,     "bus_bw_gbps": 113.05},
#     {"num_nodes": 32, "data_size_bytes": 64 * M,     "bus_bw_gbps": 123.39},
#     {"num_nodes": 32, "data_size_bytes": 128 * M,     "bus_bw_gbps": 125.93},
#     {"num_nodes": 32, "data_size_bytes": 256 * M,       "bus_bw_gbps": 127.62},
#     {"num_nodes": 32, "data_size_bytes": 512 * M,       "bus_bw_gbps": 127.94},
#     {"num_nodes": 32, "data_size_bytes": 1 * G,       "bus_bw_gbps": 128.27},
#     {"num_nodes": 32, "data_size_bytes": 2 * G,       "bus_bw_gbps": 128.41},
# ]

data = [

    # 2 ranks
    # {"num_nodes": 2, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.20},
    # {"num_nodes": 2, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.41},
    {"num_nodes": 2, "data_size_bytes": 4 * K,         "bus_bw_gbps": 0.77},
    {"num_nodes": 2, "data_size_bytes": 8 * K,         "bus_bw_gbps": 1.53},
    {"num_nodes": 2, "data_size_bytes": 16 * K,        "bus_bw_gbps": 2.93},
    {"num_nodes": 2, "data_size_bytes": 32 * K,        "bus_bw_gbps": 5.29},
    {"num_nodes": 2, "data_size_bytes": 64 * K,        "bus_bw_gbps": 9.71},
    {"num_nodes": 2, "data_size_bytes": 128 * K,        "bus_bw_gbps": 14.56},
    {"num_nodes": 2, "data_size_bytes": 256 * K,       "bus_bw_gbps": 24.97},
    {"num_nodes": 2, "data_size_bytes": 512 * K,       "bus_bw_gbps": 38.55},
    {"num_nodes": 2, "data_size_bytes": 1 * M,       "bus_bw_gbps": 51.40},
    {"num_nodes": 2, "data_size_bytes": 2 * M,       "bus_bw_gbps": 62.14},
    {"num_nodes": 2, "data_size_bytes": 4 * M,      "bus_bw_gbps": 69.67},
    {"num_nodes": 2, "data_size_bytes": 8 * M,      "bus_bw_gbps": 72.13},
    {"num_nodes": 2, "data_size_bytes": 16 * M,      "bus_bw_gbps": 69.01},
    {"num_nodes": 2, "data_size_bytes": 32 * M,     "bus_bw_gbps": 55.70},
    {"num_nodes": 2, "data_size_bytes": 64 * M,     "bus_bw_gbps": 62.33},
    {"num_nodes": 2, "data_size_bytes": 128 * M,     "bus_bw_gbps": 65.27},
    {"num_nodes": 2, "data_size_bytes": 256 * M,       "bus_bw_gbps": 68.74},
    {"num_nodes": 2, "data_size_bytes": 512 * M,       "bus_bw_gbps": 72.18},
    {"num_nodes": 2, "data_size_bytes": 1 * G,       "bus_bw_gbps": 74.63},
    {"num_nodes": 2, "data_size_bytes": 2 * G,       "bus_bw_gbps": 76.72},

    # 8 ranks
    # {"num_nodes": 8, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.22},
    # {"num_nodes": 8, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.44},
    {"num_nodes": 8, "data_size_bytes": 4 * K,         "bus_bw_gbps": 0.87},
    {"num_nodes": 8, "data_size_bytes": 8 * K,         "bus_bw_gbps": 1.71},
    {"num_nodes": 8, "data_size_bytes": 16 * K,        "bus_bw_gbps": 2.94},
    {"num_nodes": 8, "data_size_bytes": 32 * K,        "bus_bw_gbps": 5.07},
    {"num_nodes": 8, "data_size_bytes": 64 * K,        "bus_bw_gbps": 8.13},
    {"num_nodes": 8, "data_size_bytes": 128 * K,        "bus_bw_gbps": 12.14},
    {"num_nodes": 8, "data_size_bytes": 256 * K,       "bus_bw_gbps": 16.77},
    {"num_nodes": 8, "data_size_bytes": 512 * K,       "bus_bw_gbps": 19.89},
    {"num_nodes": 8, "data_size_bytes": 1 * M,       "bus_bw_gbps": 22.16},
    {"num_nodes": 8, "data_size_bytes": 2 * M,       "bus_bw_gbps": 23.09},
    {"num_nodes": 8, "data_size_bytes": 4 * M,      "bus_bw_gbps": 23.58},
    {"num_nodes": 8, "data_size_bytes": 8 * M,      "bus_bw_gbps": 23.68},
    {"num_nodes": 8, "data_size_bytes": 16 * M,      "bus_bw_gbps": 23.79},
    {"num_nodes": 8, "data_size_bytes": 32 * M,     "bus_bw_gbps": 23.84},
    {"num_nodes": 8, "data_size_bytes": 64 * M,     "bus_bw_gbps": 23.91},
    {"num_nodes": 8, "data_size_bytes": 128 * M,     "bus_bw_gbps": 23.93},
    {"num_nodes": 8, "data_size_bytes": 256 * M,       "bus_bw_gbps": 23.96},
    {"num_nodes": 8, "data_size_bytes": 512 * M,       "bus_bw_gbps": 23.97},
    {"num_nodes": 8, "data_size_bytes": 1 * G,       "bus_bw_gbps": 23.98},
    {"num_nodes": 8, "data_size_bytes": 2 * G,       "bus_bw_gbps": 23.97},

    # 16 ranks
    # {"num_nodes": 16, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.19},
    # {"num_nodes": 16, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.38},
    {"num_nodes": 16, "data_size_bytes": 4 * K,         "bus_bw_gbps": 0.76},
    {"num_nodes": 16, "data_size_bytes": 8 * K,         "bus_bw_gbps": 1.51},
    {"num_nodes": 16, "data_size_bytes": 16 * K,        "bus_bw_gbps": 2.99},
    {"num_nodes": 16, "data_size_bytes": 32 * K,        "bus_bw_gbps": 5.08},
    {"num_nodes": 16, "data_size_bytes": 64 * K,        "bus_bw_gbps": 8.50},
    {"num_nodes": 16, "data_size_bytes": 128 * K,        "bus_bw_gbps": 12.72},
    {"num_nodes": 16, "data_size_bytes": 256 * K,       "bus_bw_gbps": 18.93},
    {"num_nodes": 16, "data_size_bytes": 512 * K,       "bus_bw_gbps": 24.19},
    {"num_nodes": 16, "data_size_bytes": 1 * M,       "bus_bw_gbps": 26.62},
    {"num_nodes": 16, "data_size_bytes": 2 * M,       "bus_bw_gbps": 27.84},
    {"num_nodes": 16, "data_size_bytes": 4 * M,      "bus_bw_gbps": 28.40},
    {"num_nodes": 16, "data_size_bytes": 8 * M,      "bus_bw_gbps": 28.76},
    {"num_nodes": 16, "data_size_bytes": 16 * M,      "bus_bw_gbps": 28.95},
    {"num_nodes": 16, "data_size_bytes": 32 * M,     "bus_bw_gbps": 29.07},
    {"num_nodes": 16, "data_size_bytes": 64 * M,     "bus_bw_gbps": 29.12},
    {"num_nodes": 16, "data_size_bytes": 128 * M,     "bus_bw_gbps": 29.14},
    {"num_nodes": 16, "data_size_bytes": 256 * M,       "bus_bw_gbps": 29.16},
    {"num_nodes": 16, "data_size_bytes": 512 * M,       "bus_bw_gbps": 29.16},
    {"num_nodes": 16, "data_size_bytes": 1 * G,       "bus_bw_gbps": 29.16},
    {"num_nodes": 16, "data_size_bytes": 2 * G,       "bus_bw_gbps": 29.16},

    # 32 ranks
    # {"num_nodes": 32, "data_size_bytes": 1 * K,         "bus_bw_gbps": 0.16},
    # {"num_nodes": 32, "data_size_bytes": 2 * K,         "bus_bw_gbps": 0.32},
    {"num_nodes": 32, "data_size_bytes": 4 * K,         "bus_bw_gbps": 0.63},
    {"num_nodes": 32, "data_size_bytes": 8 * K,         "bus_bw_gbps": 1.22},
    {"num_nodes": 32, "data_size_bytes": 16 * K,        "bus_bw_gbps": 2.34},
    {"num_nodes": 32, "data_size_bytes": 32 * K,        "bus_bw_gbps": 4.46},
    {"num_nodes": 32, "data_size_bytes": 64 * K,        "bus_bw_gbps": 7.72},
    {"num_nodes": 32, "data_size_bytes": 128 * K,        "bus_bw_gbps": 12.25},
    {"num_nodes": 32, "data_size_bytes": 256 * K,       "bus_bw_gbps": 17.65},
    {"num_nodes": 32, "data_size_bytes": 512 * K,       "bus_bw_gbps": 21.92},
    {"num_nodes": 32, "data_size_bytes": 1 * M,       "bus_bw_gbps": 24.72},
    {"num_nodes": 32, "data_size_bytes": 2 * M,       "bus_bw_gbps": 26.23},
    {"num_nodes": 32, "data_size_bytes": 4 * M,      "bus_bw_gbps": 26.99},
    {"num_nodes": 32, "data_size_bytes": 8 * M,      "bus_bw_gbps": 27.42},
    {"num_nodes": 32, "data_size_bytes": 16 * M,      "bus_bw_gbps": 27.63},
    {"num_nodes": 32, "data_size_bytes": 32 * M,     "bus_bw_gbps": 27.72},
    {"num_nodes": 32, "data_size_bytes": 64 * M,     "bus_bw_gbps": 27.53},
    {"num_nodes": 32, "data_size_bytes": 128 * M,     "bus_bw_gbps": 27.61},
    {"num_nodes": 32, "data_size_bytes": 256 * M,       "bus_bw_gbps": 27.65},
    {"num_nodes": 32, "data_size_bytes": 512 * M,       "bus_bw_gbps": 27.68},
    {"num_nodes": 32, "data_size_bytes": 1 * G,       "bus_bw_gbps": 27.72},
    {"num_nodes": 32, "data_size_bytes": 2 * G,       "bus_bw_gbps": 27.75},
]

df = pd.DataFrame(data)

# If you already have a CSV, you can instead do:
# df = pd.read_csv("allreduce_results.csv")
# and make sure it has columns: num_nodes, data_size_bytes, bus_bw_gbps

# -------------------------------------------------------------------
# 2. Helper: pretty labels for data size
# -------------------------------------------------------------------
def human_readable_size(bytes_val):
    KB = 1024
    MB = 1024**2
    GB = 1024**3
    if bytes_val < MB:
        return f"{bytes_val / KB:.0f}KB"
    elif bytes_val < GB:
        return f"{bytes_val / MB:.0f}MB"
    else:
        return f"{bytes_val / GB:.0f}GB"

# -------------------------------------------------------------------
# 3. Plot: bus bandwidth vs message size, one line per num_nodes
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3))

# Sort by data size to get nice lines
df = df.sort_values("data_size_bytes")

cmap = plt.get_cmap("Set2")

for i, (num_nodes, group) in enumerate(df.groupby("num_nodes")):
    group = group.sort_values("data_size_bytes")
    ax.plot(
        group["data_size_bytes"],
        group["bus_bw_gbps"],
        marker="o",           # markers to see points
        linestyle="-",        # line style
        label=f"{num_nodes} rank{'s' if num_nodes > 1 else ''}",
        color=cmap(i % cmap.N),
        # markersize=5,
    )

# Log scale on X because range is 1KB → 2GB
ax.set_xscale("log", base=2)
# ax.set_yscale("log")

# Nice x-ticks at powers of two between 1KB and 2GB
xticks = [
    # 1 * 1024,          # 1KB
    # 2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,         # 16KB
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,        # 256KB
    512 * 1024,
    1 * 1024**2,       # 1MB
    2 * 1024**2,
    4 * 1024**2,
    8 * 1024**2,
    16 * 1024**2,
    32 * 1024**2,
    64 * 1024**2,      # 64MB
    128 * 1024**2,
    256 * 1024**2,
    512 * 1024**2,
    1 * 1024**3,       # 1GB
    2 * 1024**3,       # 2GB
]
# Keep only ticks within your data range
data_min = df["data_size_bytes"].min()
data_max = df["data_size_bytes"].max()
xticks = [x for x in xticks if data_min <= x <= data_max]

ax.set_xticks(xticks)
ax.set_xticklabels([human_readable_size(x) for x in xticks], rotation=45, ha="right")

ax.set_xlabel("Message Size")
ax.set_ylabel("Bus Bandwidth (GB/s)")
# ax.set_title("World size = 4")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend()
# ax.axvline(x=1 * M, color="red", linestyle="--", linewidth=1.5)
# ax.axvline(x=56 * M, color="red", linestyle="--", linewidth=1.5)
# ax.text(128 * K, 100, "Mesh", color="red", va="center", ha="center")
# ax.text(8 * M, 100, "RHD", color="red", va="center", ha="center")
# ax.text(512 * M, 100, "KangaRing", color="red", va="center", ha="center")
# ax.set_title("Weak Scaling Bandwidth")

plt.tight_layout()
filename = __file__.replace(".py", "")
plt.savefig(f"{filename}.png", dpi=300)
