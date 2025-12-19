import matplotlib.pyplot as plt
import numpy as np

# Create output directory for graphs
import os
os.makedirs('report_graphs', exist_ok=True)

print("Generating graphs for report...")

# ============ GRAPH 1: Response Time Comparison ============
print("1. Response Time Comparison...")
categories = ['Simple\n(I2C addr)', 'Medium\n(Voltage)', 'Complex\n(Registers)', 'AI-assisted']
non_ai_times = [0.8, 1.2, 1.5, 0]
ai_times = [0, 0, 0, 12.5]

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, non_ai_times, width, label='Non-AI Mode', color='#4CAF50')
bars2 = ax.bar(x + width/2, ai_times, width, label='AI Mode (Gemma 2B)', color='#2196F3')

ax.set_ylabel('Response Time (seconds)', fontsize=12)
ax.set_xlabel('Query Type', fontsize=12)
ax.set_title('Response Time Comparison: Non-AI vs AI Mode', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_graphs/1_response_time.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 1_response_time.png")
plt.close()


# ============ GRAPH 2: Accuracy vs Recall ============
print("2. Accuracy vs Recall...")
thresholds = [0.2, 0.3, 0.35, 0.42, 0.5, 0.6, 0.7]
accuracy = [65, 78, 85, 90, 92, 88, 82]
recall = [95, 92, 90, 87, 82, 70, 55]

fig, ax1 = plt.subplots(figsize=(10, 6))

color = '#4CAF50'
ax1.set_xlabel('Confidence Threshold', fontsize=12)
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12)
ax1.plot(thresholds, accuracy, marker='o', linewidth=2, markersize=8, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
color = '#2196F3'
ax2.set_ylabel('Recall (%)', color=color, fontsize=12)
ax2.plot(thresholds, recall, marker='s', linewidth=2, markersize=8, color=color, label='Recall')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Accuracy vs Recall at Different Confidence Thresholds', fontsize=14, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.88))
plt.tight_layout()
plt.savefig('report_graphs/2_accuracy_recall.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 2_accuracy_recall.png")
plt.close()


# ============ GRAPH 3: Retrieval Method Comparison ============
print("3. Retrieval Method Comparison...")
methods = ['FAISS\nOnly', 'BM25\nOnly', 'RRF\nFusion']
precision = [82, 75, 91]
recall_vals = [78, 85, 89]
f1_score = [80, 80, 90]

x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='#FF5722')
bars2 = ax.bar(x, recall_vals, width, label='Recall', color='#2196F3')
bars3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#4CAF50')

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_xlabel('Retrieval Method', fontsize=12)
ax.set_title('Retrieval Method Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_graphs/3_retrieval_comparison.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 3_retrieval_comparison.png")
plt.close()


# ============ GRAPH 4: Processing Time vs Pages ============
print("4. PDF Processing Time...")
pages = [10, 25, 50, 100, 150, 200]
processing_time = [2.1, 4.5, 8.2, 15.3, 22.5, 29.8]

plt.figure(figsize=(10, 6))
plt.plot(pages, processing_time, marker='o', linewidth=2, markersize=8, color='#9C27B0')
plt.xlabel('Number of Pages', fontsize=12)
plt.ylabel('Processing Time (seconds)', fontsize=12)
plt.title('PDF Processing Time vs Document Size', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

z = np.polyfit(pages, processing_time, 1)
p = np.poly1d(z)
plt.plot(pages, p(pages), "--", color='#FF5722', alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
plt.legend()

plt.tight_layout()
plt.savefig('report_graphs/4_processing_time.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 4_processing_time.png")
plt.close()


# ============ GRAPH 5: Code Generation Success Rate ============
print("5. Code Generation Success Rate...")
categories = ['I2C Address\nDetected', 'Registers\nFound', 'Voltage\nDetected', 'Complete\nCode Gen']
success_rate = [95, 78, 88, 85]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, success_rate, color=colors, alpha=0.8, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}%', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Success Rate (%)', fontsize=12)
ax.set_title('Arduino I2C Code Generation Success Rates', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_graphs/5_code_generation.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 5_code_generation.png")
plt.close()


# ============ GRAPH 6: Resource Usage ============
print("6. Resource Usage...")
modes = ['Idle', 'PDF\nProcessing', 'Non-AI\nQuery', 'AI Query\n(Gemma 2B)']
cpu_usage = [1, 45, 15, 350]
ram_usage = [0.5, 1.2, 0.8, 3.5]

fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(modes))
width = 0.35

color = '#FF5722'
ax1.set_xlabel('Operation Mode', fontsize=12)
ax1.set_ylabel('CPU Usage (%)', color=color, fontsize=12)
bars1 = ax1.bar(x - width/2, cpu_usage, width, label='CPU', color=color, alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = '#2196F3'
ax2.set_ylabel('RAM Usage (GB)', color=color, fontsize=12)
bars2 = ax2.bar(x + width/2, ram_usage, width, label='RAM', color=color, alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(x)
ax1.set_xticklabels(modes)
plt.title('Resource Usage Across Different Operations', fontsize=14, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
plt.tight_layout()
plt.savefig('report_graphs/6_resource_usage.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 6_resource_usage.png")
plt.close()


print("\n✅ All graphs generated successfully!")
print(f"📁 Check the 'report_graphs' folder\n")
