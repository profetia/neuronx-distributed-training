import matplotlib.pyplot as plt

# ==========================
# Replace these with your data
# ==========================
# Example sizes
N = [256, 512, 1024, 2048, 4096, 8192, 16384]

# Corresponding GEMM TFLOPs measurements
tflops = [0.07, 0.56, 4.3, 34.9, 66.8, 75.9, 74.7]
# ==========================

plt.figure(figsize=(6, 4))
plt.plot(N, tflops, marker='o', linewidth=2)
# add text labels to each point
for i, txt in enumerate(tflops):
    plt.annotate(f"{txt}", (N[i], tflops[i]), textcoords="offset points", xytext=(0,10), ha='center')

# plt.title("GEMM Performance vs Matrix Size (N=M=K)")
plt.xlabel("Input Size (N = M = K)")
plt.ylabel("TFLOPs")
plt.grid(True)

# Optional: log-scale for better visualization if range is large
plt.xscale("log", base=2)
plt.xticks([256, 512, 1024, 2048, 4096, 8192, 16384],
           labels=["256", "512", "1024", "2048", "4096", "8192", "16384"])
# plt.yscale("log")
plt.ylim(None, 100)

# draw a horizontal line at y=95
plt.axhline(y=95, color='r', linestyle='--')
# add text label to the line, the spec says bf16 compute capability 95 TFLOPs
plt.text(220, 90, "Compute capability of BF16", color='r')
plt.text(165, 93, "95", color='r')


plt.tight_layout()
filename = __file__.replace(".py", "")
plt.savefig(f"{filename}.png", dpi=300)
