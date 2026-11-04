import matplotlib.pyplot as plt
import numpy as np

# Define performance metrics and their measured values (in milliseconds)
metrics = ['FAISS Search Latency', 'Embedding Generation', 'Chunk Processing', 'Avg Retrieval Time']
values = [7.23, 14.50, 18.20, 40.00]  # Measured and estimated benchmark metrics

# Set figure size and style
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2563eb', '#16a34a', '#d97706', '#9333ea']

# Create vertical bar chart
bars = ax.bar(metrics, values, color=colors, width=0.55, edgecolor='black', linewidth=0.8)

# Add titles and axis labels
ax.set_title('NIT Delhi RAG System - Performance & Latency Benchmarks', fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Latency / Time (ms)', fontsize=11, fontweight='bold')
ax.set_xlabel('System Evaluation Metrics', fontsize=11, fontweight='bold')

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f} ms',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),  # Offset points
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add a clean dashed grid behind the bars
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# Adjust layout for a clean fit
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Save the chart as a high-resolution image for your report
output_filename = 'rag_performance_chart.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✅ Performance chart successfully generated and saved as '{output_filename}'!")

# Display the chart window
plt.show()