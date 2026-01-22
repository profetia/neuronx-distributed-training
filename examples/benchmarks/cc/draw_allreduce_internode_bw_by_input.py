import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Measured by `nccom-test`

# -------------------------------------------------------------------
# 1. Prepare your data
# -------------------------------------------------------------------
# Example: your data as a list of dicts.
# Replace this with your real benchmark results or load from CSV.
data = [
    # 1 node
    {"num_nodes": 1, "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 0.22},
    {"num_nodes": 1, "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 0.43},
    {"num_nodes": 1, "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 0.83},
    {"num_nodes": 1, "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 1.61},
    {"num_nodes": 1, "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 3.13},
    {"num_nodes": 1, "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 5.88},
    {"num_nodes": 1, "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 10.15},
    {"num_nodes": 1, "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 15.75},
    {"num_nodes": 1, "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 22.37},
    {"num_nodes": 1, "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 28.31},
    {"num_nodes": 1, "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 31.80},
    {"num_nodes": 1, "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 33.58},
    {"num_nodes": 1, "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 34.10},
    {"num_nodes": 1, "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 52.70},
    {"num_nodes": 1, "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 82.60},
    {"num_nodes": 1, "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 112.24},
    {"num_nodes": 1, "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 123.13},
    {"num_nodes": 1, "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 125.87},
    {"num_nodes": 1, "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 127.59},
    {"num_nodes": 1, "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 127.81},
    {"num_nodes": 1, "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 128.23},
    {"num_nodes": 1, "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 128.41},

    # 2 nodes
    {"num_nodes": 2, "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 0.00},
    {"num_nodes": 2, "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 0.01},
    {"num_nodes": 2, "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 0.01},
    {"num_nodes": 2, "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 0.02},
    {"num_nodes": 2, "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 0.04},
    {"num_nodes": 2, "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 0.09},
    {"num_nodes": 2, "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 0.17},
    {"num_nodes": 2, "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 0.33},
    {"num_nodes": 2, "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 1.43},
    {"num_nodes": 2, "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 2.98},
    {"num_nodes": 2, "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 12.06},
    {"num_nodes": 2, "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 20.88},
    {"num_nodes": 2, "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 23.11},
    {"num_nodes": 2, "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 35.38},
    {"num_nodes": 2, "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 42.23},
    {"num_nodes": 2, "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 40.93},
    {"num_nodes": 2, "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 45.37},

    # 4 nodes
    {"num_nodes": 4, "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 0.00},
    {"num_nodes": 4, "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 0.00},
    {"num_nodes": 4, "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 0.00},
    {"num_nodes": 4, "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 0.01},
    {"num_nodes": 4, "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 0.02},
    {"num_nodes": 4, "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 0.03},
    {"num_nodes": 4, "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 0.06},
    {"num_nodes": 4, "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 0.28},
    {"num_nodes": 4, "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 0.72},
    {"num_nodes": 4, "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 1.55},
    {"num_nodes": 4, "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 4.27},
    {"num_nodes": 4, "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 7.77},
    {"num_nodes": 4, "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 9.91},
    {"num_nodes": 4, "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 11.07},
    {"num_nodes": 4, "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 12.09},
    {"num_nodes": 4, "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 12.88},
    {"num_nodes": 4, "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 15.15},
    {"num_nodes": 4, "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 15.35},
    {"num_nodes": 4, "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 11.04},
    {"num_nodes": 4, "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 11.52},
    {"num_nodes": 4, "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 11.54},
    {"num_nodes": 4, "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 11.55},
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
fig, ax = plt.subplots(figsize=(8, 5))

# Sort by data size to get nice lines
df = df.sort_values("data_size_bytes")

for num_nodes, group in df.groupby("num_nodes"):
    group = group.sort_values("data_size_bytes")
    ax.plot(
        group["data_size_bytes"],
        group["bus_bw_gbps"],
        marker="o",           # markers to see points
        linestyle="-",        # line style
        label=f"{num_nodes} node{'s' if num_nodes > 1 else ''}",
    )

# Log scale on X because range is 1KB → 2GB
ax.set_xscale("log", base=2)

# Nice x-ticks at powers of two between 1KB and 2GB
xticks = [
    1 * 1024,          # 1KB
    2 * 1024,
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

ax.set_xlabel("Total Input size")
ax.set_ylabel("Bus bandwidth (GB/s)")
ax.set_title("All-Reduce Bus Bandwidth vs Total Input Size\n(Total number of processes = 32)")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend()

plt.tight_layout()
filename = __file__.replace(".py", "")
plt.savefig(f"{filename}.png", dpi=300)
