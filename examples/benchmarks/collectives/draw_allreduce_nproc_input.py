import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------
# 1. Put your new results into a DataFrame
# ------------------------------------------------------
# Example data – replace with your own measurements
data = [
    # data_size_bytes = 1MB
    {"num_procs": 2,   "data_size_bytes": 1 * 1024**2, "bus_bw_gbps": 56.45},
    {"num_procs": 8,   "data_size_bytes": 1 * 1024**2, "bus_bw_gbps": 26.12},
    {"num_procs": 16,  "data_size_bytes": 1 * 1024**2, "bus_bw_gbps": 32.45},
    {"num_procs": 32,  "data_size_bytes": 1 * 1024**2, "bus_bw_gbps": 31.66},
    {"num_procs": 64,  "data_size_bytes": 1 * 1024**2, "bus_bw_gbps": 5.47},
    {"num_procs": 128, "data_size_bytes": 1 * 1024**2, "bus_bw_gbps": 3.83},

    # data_size_bytes = 4MB
    {"num_procs": 2,   "data_size_bytes": 4 * 1024**2, "bus_bw_gbps": 63.29},
    {"num_procs": 8,   "data_size_bytes": 4 * 1024**2, "bus_bw_gbps": 54.98},
    {"num_procs": 16,  "data_size_bytes": 4 * 1024**2, "bus_bw_gbps": 47.32},
    {"num_procs": 32,  "data_size_bytes": 4 * 1024**2, "bus_bw_gbps": 33.97},
    {"num_procs": 64,  "data_size_bytes": 4 * 1024**2, "bus_bw_gbps": 24.82},
    {"num_procs": 128, "data_size_bytes": 4 * 1024**2, "bus_bw_gbps": 14.35},

    # data_size_bytes = 16MB
    {"num_procs": 2,   "data_size_bytes": 16 * 1024**2, "bus_bw_gbps": 56.66},
    {"num_procs": 8,   "data_size_bytes": 16 * 1024**2, "bus_bw_gbps": 62.95},
    {"num_procs": 16,  "data_size_bytes": 16 * 1024**2, "bus_bw_gbps": 64.05},
    {"num_procs": 32,  "data_size_bytes": 16 * 1024**2, "bus_bw_gbps": 82.60},
    {"num_procs": 64,  "data_size_bytes": 16 * 1024**2, "bus_bw_gbps": 52.21},
    {"num_procs": 128, "data_size_bytes": 16 * 1024**2, "bus_bw_gbps": 31.28},

    # data_size_bytes = 64MB
    {"num_procs": 2,   "data_size_bytes": 64 * 1024**2, "bus_bw_gbps": 57.14},
    {"num_procs": 8,   "data_size_bytes": 64 * 1024**2, "bus_bw_gbps": 66.75},
    {"num_procs": 16,  "data_size_bytes": 64 * 1024**2, "bus_bw_gbps": 68.14},
    {"num_procs": 32,  "data_size_bytes": 64 * 1024**2, "bus_bw_gbps": 123.10},
    {"num_procs": 64,  "data_size_bytes": 64 * 1024**2, "bus_bw_gbps": 76.68},
    {"num_procs": 128, "data_size_bytes": 64 * 1024**2, "bus_bw_gbps": 62.03},

    # data_size_bytes = 256MB
    {"num_procs": 2,   "data_size_bytes": 256 * 1024**2, "bus_bw_gbps": 56.91},
    {"num_procs": 8,   "data_size_bytes": 256 * 1024**2, "bus_bw_gbps": 67.00},
    {"num_procs": 16,  "data_size_bytes": 256 * 1024**2, "bus_bw_gbps": 68.52},
    {"num_procs": 32,  "data_size_bytes": 256 * 1024**2, "bus_bw_gbps": 127.55},
    {"num_procs": 64,  "data_size_bytes": 256 * 1024**2, "bus_bw_gbps": 92.25},
    {"num_procs": 128, "data_size_bytes": 256 * 1024**2, "bus_bw_gbps": 84.55},
]

df = pd.DataFrame(data)

# If you already have a CSV, you can do:
# df = pd.read_csv("allreduce_vs_procs.csv")
# and ensure columns: num_procs, data_size_bytes, bus_bw_gbps

# ------------------------------------------------------
# 2. Create a readable label for each data size
# ------------------------------------------------------
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

df["data_size_label"] = df["data_size_bytes"].apply(human_readable_size)

# ------------------------------------------------------
# 3. Plot: bus BW vs number of processes, grouped by data size
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for data_size, group_data in df.groupby("data_size_bytes"):
    ax.plot(
        group_data["num_procs"],
        group_data["bus_bw_gbps"],
        marker="o",
        label=human_readable_size(data_size),
    )

# Optional: if num_procs grows like 2,4,8,16,... you can uncomment log scale:
ax.set_xscale("log", base=2)
ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
ax.set_xticklabels([2, 4, 8, 16, 32, 64, 128])

ymin, ymax = ax.get_ylim()

ax.axvline(x=3, linestyle="--", linewidth=1)
ax.axvline(x=48, linestyle="--", linewidth=1)
ax.text(2.2, ymin + 1, "Intra-chip", ha="center", va="bottom", fontsize=9)
ax.text(12, ymin + 1, "Intra-node", ha="center", va="bottom", fontsize=9)
ax.text(90, ymin + 1, "Inter-node", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Number of processes")
ax.set_ylabel("Bus bandwidth (GB/s)")
ax.set_title("All-Reduce Bus Bandwidth vs Number of Processes")
ax.grid(True, which="both", ls="--", lw=0.5)
ax.legend(title="Total Input Size")

plt.tight_layout()
filename = __file__.replace(".py", "")
plt.savefig(f"{filename}.png", dpi=300)

