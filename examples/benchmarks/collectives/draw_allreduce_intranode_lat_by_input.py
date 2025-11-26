import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------
# 1. Prepare your data
# -------------------------------------------------------------------
# Example: your data as a list of dicts.
# Replace this with your real benchmark results or load from CSV.
data = [
    # 2 proc

    {"source": "torch-xla", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 210.2},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 221.9},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 212.1},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 217.4},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 191.4},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 220.3},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 212.6},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 205.3},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 235.8},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 220.3},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 225.1},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 217.5},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 232.8},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 287.9},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 399.6},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 615.4},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 1059.1},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 2123.3},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 4011.2},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 7876.0},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 15623.7},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 31133.0},

    {"source": "nccom-test", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 2.1},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 2.1},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 2.1},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 2.1},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 2.1},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 2.7},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 3.5},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 5.2},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 7.1},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 11.0},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 18.7},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 34.5},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 66.0},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 131.3},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 296.1},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 585.0},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 1168.9},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 2339.3},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 4698.5},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 9483.9},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 18995.9},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 37950.3},
]

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
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

def draw(df, ax, n):
    # Sort by data size to get nice lines
    df = df.sort_values("data_size_bytes")

    for source, group in df.groupby("source"):
        group = group.sort_values("data_size_bytes")
        ax.plot(
            group["data_size_bytes"],
            group["bus_bw_gbps"],
            marker="o",           # markers to see points
            linestyle="-",        # line style
            label=f"{source}",
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

    ax.set_xlabel("Total input size")
    ax.set_ylabel("Latency (us)")
    ax.set_title(f"{n} Ranks")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_ylim(0, 60000)
    # ax.set_ylim(0, 100000)
    # ax.set_yscale("symlog", linthresh=10)
    ax.legend()


df = pd.DataFrame(data)
draw(df, ax[0, 0], 2)

data = [
    # 8 proc

    {"source": "torch-xla", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 198.8},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 197.2},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 197.5},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 197.6},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 196.9},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 197.9},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 198.0},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 197.3},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 198.5},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 187.9},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 195.7},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 200.6},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 250.1},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 254.2},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 298.0},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 374.0},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 521.5},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 844.6},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 1502.3},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 2748.8},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 5223.0},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 10236.4},

    {"source": "nccom-test", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 6.3},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 6.4},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 6.8},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 7.1},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 7.5},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 9.0},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 11.6},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 16.6},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 23.6},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 39.1},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 70.3},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 86.8},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 133.5},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 243.3},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 466.4},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 892.7},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 1759.1},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 3507.1},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 7007.5},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 14014.4},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 28019.1},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 56035.3},
]

df = pd.DataFrame(data)
draw(df, ax[0, 1], 8)

data = [
    # 16 proc

    {"source": "torch-xla", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 222.5},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 206.1},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 210.4},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 213.0},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 212.0},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 211.3},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 210.1},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 208.5},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 207.9},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 201.8},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 209.2},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 201.3},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 200.2},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 258.6},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 266.5},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 326.8},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 395.8},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 562.1},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 891.9},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 1582.1},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 2865.8},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 5436.9},

    {"source": "nccom-test", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 7.1},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 7.2},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 7.6},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 8.1},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 8.1},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 9.1},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 11.7},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 16.3},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 21.9},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 33.9},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 60.2},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 116.7},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 163.4},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 252.8},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 479.9},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 934.1},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 1846.9},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 3678.9},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 7345.0},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 14670.9},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 29283.5},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 58487.2},
]

df = pd.DataFrame(data)
draw(df, ax[1, 0], 16)

data = [
    # 32 proc

    {"source": "torch-xla", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 268.9},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 246.7},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 246.0},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 236.6},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 242.5},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 245.3},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 237.6},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 241.4},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 253.4},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 244.6},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 242.8},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 245.3},
    {"source": "torch-xla", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 245.7},
    {"source": "torch-xla", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 274.1},
    {"source": "torch-xla", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 294.3},
    {"source": "torch-xla", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 312.4},
    {"source": "torch-xla", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 410.1},
    {"source": "torch-xla", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 533.9},
    {"source": "torch-xla", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 637.0},
    {"source": "torch-xla", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 1022.5},
    {"source": "torch-xla", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 1845.9},
    {"source": "torch-xla", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 2565.2},

    {"source": "nccom-test", "data_size_bytes": 1 * 1024,          "bus_bw_gbps": 9.1},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024,          "bus_bw_gbps": 9.1},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024,          "bus_bw_gbps": 9.5},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024,          "bus_bw_gbps": 9.7},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024,         "bus_bw_gbps": 10.1},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024,         "bus_bw_gbps": 10.7},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024,         "bus_bw_gbps": 12.5},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024,        "bus_bw_gbps": 16.1},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024,        "bus_bw_gbps": 22.6},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024,        "bus_bw_gbps": 35.9},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**2,       "bus_bw_gbps": 63.9},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**2,       "bus_bw_gbps": 120.8},
    {"source": "nccom-test", "data_size_bytes": 4 * 1024**2,       "bus_bw_gbps": 238.0},
    {"source": "nccom-test", "data_size_bytes": 8 * 1024**2,       "bus_bw_gbps": 307.1},
    {"source": "nccom-test", "data_size_bytes": 16 * 1024**2,      "bus_bw_gbps": 391.6},
    {"source": "nccom-test", "data_size_bytes": 32 * 1024**2,      "bus_bw_gbps": 575.1},
    {"source": "nccom-test", "data_size_bytes": 64 * 1024**2,      "bus_bw_gbps": 1053.8},
    {"source": "nccom-test", "data_size_bytes": 128 * 1024**2,     "bus_bw_gbps": 2065.0},
    {"source": "nccom-test", "data_size_bytes": 256 * 1024**2,     "bus_bw_gbps": 4075.4},
    {"source": "nccom-test", "data_size_bytes": 512 * 1024**2,     "bus_bw_gbps": 8130.0},
    {"source": "nccom-test", "data_size_bytes": 1 * 1024**3,       "bus_bw_gbps": 16218.4},
    {"source": "nccom-test", "data_size_bytes": 2 * 1024**3,       "bus_bw_gbps": 32401.6},
]

df = pd.DataFrame(data)
draw(df, ax[1, 1], 32)

plt.tight_layout()
filename = __file__.replace(".py", "")
plt.savefig(f"{filename}.png", dpi=300)
