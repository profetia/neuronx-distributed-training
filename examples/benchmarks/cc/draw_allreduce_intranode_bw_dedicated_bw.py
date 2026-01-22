import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

K = 1024
M = 1024**2
G = 1024**3

data = [ # world size = 4

    # 1 ring
    {"num_nodes": 1, "data_size_bytes": 32 * K,        "bus_bw_gbps": 0.9},
    {"num_nodes": 1, "data_size_bytes": 64 * K,        "bus_bw_gbps": 1.9},
    {"num_nodes": 1, "data_size_bytes": 128 * K,        "bus_bw_gbps": 4.2},
    {"num_nodes": 1, "data_size_bytes": 256 * K,       "bus_bw_gbps": 8.1},
    {"num_nodes": 1, "data_size_bytes": 512 * K,       "bus_bw_gbps": 13.1},
    {"num_nodes": 1, "data_size_bytes": 1 * M,       "bus_bw_gbps": 23.3},
    {"num_nodes": 1, "data_size_bytes": 2 * M,       "bus_bw_gbps": 46.7},
    {"num_nodes": 1, "data_size_bytes": 4 * M,      "bus_bw_gbps": 70.2},
    {"num_nodes": 1, "data_size_bytes": 8 * M,      "bus_bw_gbps": 99.4},
    {"num_nodes": 1, "data_size_bytes": 16 * M,      "bus_bw_gbps": 120.4},
    {"num_nodes": 1, "data_size_bytes": 32 * M,     "bus_bw_gbps": 133.2},
    {"num_nodes": 1, "data_size_bytes": 64 * M,     "bus_bw_gbps": 145.2},
    {"num_nodes": 1, "data_size_bytes": 128 * M,     "bus_bw_gbps": 151.6},
    {"num_nodes": 1, "data_size_bytes": 256 * M,       "bus_bw_gbps": 156.5},
    {"num_nodes": 1, "data_size_bytes": 512 * M,       "bus_bw_gbps": 157.5},

    # 2 ring
    {"num_nodes": 2, "data_size_bytes": 32 * K,        "bus_bw_gbps": 1.9},
    {"num_nodes": 2, "data_size_bytes": 64 * K,        "bus_bw_gbps": 3.8},
    {"num_nodes": 2, "data_size_bytes": 128 * K,        "bus_bw_gbps": 7.8},
    {"num_nodes": 2, "data_size_bytes": 256 * K,       "bus_bw_gbps": 14.4},
    {"num_nodes": 2, "data_size_bytes": 512 * K,       "bus_bw_gbps": 24.5},
    {"num_nodes": 2, "data_size_bytes": 1 * M,       "bus_bw_gbps": 49.7},
    {"num_nodes": 2, "data_size_bytes": 2 * M,       "bus_bw_gbps": 86.9},
    {"num_nodes": 2, "data_size_bytes": 4 * M,      "bus_bw_gbps": 130.9},
    {"num_nodes": 2, "data_size_bytes": 8 * M,      "bus_bw_gbps": 175.6},
    {"num_nodes": 2, "data_size_bytes": 16 * M,      "bus_bw_gbps": 208.7},
    {"num_nodes": 2, "data_size_bytes": 32 * M,     "bus_bw_gbps": 213.3},
    {"num_nodes": 2, "data_size_bytes": 64 * M,     "bus_bw_gbps": 233.6},
    {"num_nodes": 2, "data_size_bytes": 128 * M,     "bus_bw_gbps": 245.7},
    {"num_nodes": 2, "data_size_bytes": 256 * M,       "bus_bw_gbps": 252.4},
    {"num_nodes": 2, "data_size_bytes": 512 * M,       "bus_bw_gbps": 238.5},
]

df = pd.DataFrame(data)
df.loc[df["num_nodes"] == 2, "bus_bw_gbps"] /= 2

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

# Sort by data size to get nice lines
df = df.sort_values("data_size_bytes")

for num_nodes, group in df.groupby("num_nodes"):
    group = group.sort_values("data_size_bytes")
    ax[0, 0].plot(
        group["data_size_bytes"],
        group["bus_bw_gbps"],
        marker="o",           # markers to see points
        linestyle="-",        # line style
        label=f"{num_nodes} group{'s' if num_nodes > 1 else ''}",
    )

# Log scale on X because range is 1KB → 2GB
ax[0, 0].set_xscale("log", base=2)

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

ax[0, 0].set_xticks(xticks)
ax[0, 0].set_xticklabels([human_readable_size(x) for x in xticks], rotation=45, ha="right")

ax[0, 0].set_xlabel("Input size per rank")
ax[0, 0].set_ylabel("Bus bandwidth (GB/s)")
# ax[0, 0].set_title("World size = 4")
ax[0, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
ax[0, 0].legend()
ax[0, 0].axvline(x=1 * M, color="red", linestyle="--", linewidth=1.5)
ax[0, 0].axvline(x=56 * M, color="red", linestyle="--", linewidth=1.5)
ax[0, 0].text(128 * K, 40, "Mesh", color="red", va="center", ha="center")
ax[0, 0].text(8 * M, 40, "RHD", color="red", va="center", ha="center")
ax[0, 0].text(256 * M, 40, "KangaRing", color="red", va="center", ha="center")
ax[0, 0].set_title("Bus Bandwidth (World size = 4)")

df_ratio = df[df["num_nodes"] == 1]["bus_bw_gbps"].values / df[df["num_nodes"] == 1]["bus_bw_gbps"].values
ax[0, 1].plot(
    df[df["num_nodes"] == 1]["data_size_bytes"],
    df_ratio,
    marker="o",
    linestyle="-",
    label="1 group",
)
df_ratio = df[df["num_nodes"] == 2]["bus_bw_gbps"].values / df[df["num_nodes"] == 1]["bus_bw_gbps"].values
ax[0, 1].plot(
    df[df["num_nodes"] == 1]["data_size_bytes"],
    df_ratio,
    marker="o",
    linestyle="-",
    label="2 groups",
)
ax[0, 1].set_xscale("log", base=2)
ax[0, 1].set_xticks(xticks)
ax[0, 1].set_ylim(-0.1, 2)
ax[0, 1].set_xticklabels([human_readable_size(x) for x in xticks], rotation=45, ha="right")
ax[0, 1].set_xlabel("Input size per rank")
ax[0, 1].set_ylabel("Bandwidth Ratio")
ax[0, 1].legend()
ax[0, 1].grid(True, which="both", linestyle="--", linewidth=0.5)
ax[0, 1].axvline(x=1 * M, color="red", linestyle="--", linewidth=1.5)
ax[0, 1].axvline(x=56 * M, color="red", linestyle="--", linewidth=1.5)
ax[0, 1].text(128 * K, 0.4, "Mesh", color="red", va="center", ha="center")
ax[0, 1].text(8 * M, 0.4, "RHD", color="red", va="center", ha="center")
ax[0, 1].text(256 * M, 0.4, "KangaRing", color="red", va="center", ha="center")
ax[0, 1].set_title("Bus Bandwidth (World size = 4, Normalized)")


data = [ # world size = 8

    # 1 ring
    {"num_nodes": 1, "data_size_bytes": 32 * K,        "bus_bw_gbps": 1.6},
    {"num_nodes": 1, "data_size_bytes": 64 * K,        "bus_bw_gbps": 3.4},
    {"num_nodes": 1, "data_size_bytes": 128 * K,        "bus_bw_gbps": 6.8},
    {"num_nodes": 1, "data_size_bytes": 256 * K,       "bus_bw_gbps": 13.7},
    {"num_nodes": 1, "data_size_bytes": 512 * K,       "bus_bw_gbps": 27.4},
    {"num_nodes": 1, "data_size_bytes": 1 * M,       "bus_bw_gbps": 54.6},
    {"num_nodes": 1, "data_size_bytes": 2 * M,       "bus_bw_gbps": 91.7},
    {"num_nodes": 1, "data_size_bytes": 4 * M,      "bus_bw_gbps": 146.3},
    {"num_nodes": 1, "data_size_bytes": 8 * M,      "bus_bw_gbps": 215.4},
    {"num_nodes": 1, "data_size_bytes": 16 * M,      "bus_bw_gbps": 248.1},
    {"num_nodes": 1, "data_size_bytes": 32 * M,     "bus_bw_gbps": 281.8},
    {"num_nodes": 1, "data_size_bytes": 64 * M,     "bus_bw_gbps": 312.2},
    {"num_nodes": 1, "data_size_bytes": 128 * M,     "bus_bw_gbps": 328.6},
    {"num_nodes": 1, "data_size_bytes": 256 * M,       "bus_bw_gbps": 338.9},
    {"num_nodes": 1, "data_size_bytes": 512 * M,       "bus_bw_gbps": 343.5},

    # 2 ring
    {"num_nodes": 2, "data_size_bytes": 32 * K,        "bus_bw_gbps": 3.2},
    {"num_nodes": 2, "data_size_bytes": 64 * K,        "bus_bw_gbps": 6.4},
    {"num_nodes": 2, "data_size_bytes": 128 * K,        "bus_bw_gbps": 12.8},
    {"num_nodes": 2, "data_size_bytes": 256 * K,       "bus_bw_gbps": 25.6},
    {"num_nodes": 2, "data_size_bytes": 512 * K,       "bus_bw_gbps": 53.5},
    {"num_nodes": 2, "data_size_bytes": 1 * M,       "bus_bw_gbps": 104.0},
    {"num_nodes": 2, "data_size_bytes": 2 * M,       "bus_bw_gbps": 171.5},
    {"num_nodes": 2, "data_size_bytes": 4 * M,      "bus_bw_gbps": 273.8},
    {"num_nodes": 2, "data_size_bytes": 8 * M,      "bus_bw_gbps": 365.5},
    {"num_nodes": 2, "data_size_bytes": 16 * M,      "bus_bw_gbps": 430.7},
    {"num_nodes": 2, "data_size_bytes": 32 * M,     "bus_bw_gbps": 421.1},
    {"num_nodes": 2, "data_size_bytes": 64 * M,     "bus_bw_gbps": 429.3},
    {"num_nodes": 2, "data_size_bytes": 128 * M,     "bus_bw_gbps": 441.1},
    {"num_nodes": 2, "data_size_bytes": 256 * M,       "bus_bw_gbps": 462.2},
    {"num_nodes": 2, "data_size_bytes": 512 * M,       "bus_bw_gbps": 459.7},
]

df = pd.DataFrame(data)
df.loc[df["num_nodes"] == 2, "bus_bw_gbps"] /= 2


for num_nodes, group in df.groupby("num_nodes"):
    group = group.sort_values("data_size_bytes")
    ax[1, 0].plot(
        group["data_size_bytes"],
        group["bus_bw_gbps"],
        marker="o",           # markers to see points
        linestyle="-",        # line style
        label=f"{num_nodes} group{'s' if num_nodes > 1 else ''}",
    )

# Log scale on X because range is 1KB → 2GB
ax[1, 0].set_xscale("log", base=2)

data_min = df["data_size_bytes"].min()
data_max = df["data_size_bytes"].max()
xticks = [x for x in xticks if data_min <= x <= data_max]

ax[1, 0].set_xticks(xticks)
ax[1, 0].set_xticklabels([human_readable_size(x) for x in xticks], rotation=45, ha="right")

ax[1, 0].set_xlabel("Input size per rank")
ax[1, 0].set_ylabel("Bus bandwidth (GB/s)")
# ax[1, 0].set_title("World size = 8")
ax[1, 0].grid(True, which="both", linestyle="--", linewidth=0.5)
ax[1, 0].legend()
ax[1, 0].axvline(x=1 * M, color="red", linestyle="--", linewidth=1.5)
ax[1, 0].axvline(x=56 * M, color="red", linestyle="--", linewidth=1.5)
ax[1, 0].text(128 * K, 80, "Mesh", color="red", va="center", ha="center")
ax[1, 0].text(8 * M, 80, "RHD", color="red", va="center", ha="center")
ax[1, 0].text(256 * M, 80, "KangaRing", color="red", va="center", ha="center")
ax[1, 0].set_title("Bus Bandwidth (World size = 8)")

df_ratio = df[df["num_nodes"] == 1]["bus_bw_gbps"].values / df[df["num_nodes"] == 1]["bus_bw_gbps"].values
ax[1, 1].plot(
    df[df["num_nodes"] == 1]["data_size_bytes"],
    df_ratio,
    marker="o",
    linestyle="-",
    label="1 group",
)
df_ratio = df[df["num_nodes"] == 2]["bus_bw_gbps"].values / df[df["num_nodes"] == 1]["bus_bw_gbps"].values
ax[1, 1].plot(
    df[df["num_nodes"] == 1]["data_size_bytes"],
    df_ratio,
    marker="o",
    linestyle="-",
    label="2 groups",
)
ax[1, 1].set_xscale("log", base=2)
ax[1, 1].set_xticks(xticks)
ax[1, 1].set_ylim(-0.1, 2)
ax[1, 1].set_xticklabels([human_readable_size(x) for x in xticks], rotation=45, ha="right")
ax[1, 1].set_xlabel("Input size per rank")
ax[1, 1].set_ylabel("Bus Bandwidth (%)")
ax[1, 1].set_title("Bus Bandwidth (World size = 8, Normalized)")
ax[1, 1].legend()
ax[1, 1].grid(True, which="both", linestyle="--", linewidth=0.5)
ax[1, 1].axvline(x=1 * M, color="red", linestyle="--", linewidth=1.5)
ax[1, 1].axvline(x=56 * M, color="red", linestyle="--", linewidth=1.5)
ax[1, 1].text(128 * K, 0.4, "Mesh", color="red", va="center", ha="center")
ax[1, 1].text(8 * M, 0.4, "RHD", color="red", va="center", ha="center")
ax[1, 1].text(256 * M, 0.4, "KangaRing", color="red", va="center", ha="center")


plt.tight_layout()
filename = __file__.replace(".py", "")
plt.savefig(f"{filename}.png", dpi=300)
