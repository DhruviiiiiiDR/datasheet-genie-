import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
os.makedirs('report_graphs', exist_ok=True)

print("📊 Loading benchmark data...")

# Load your real data
try:
    with open('benchmark_results.json', 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded data from {data['pdf_name']}")
except FileNotFoundError:
    print("❌ benchmark_results.json not found!")
    print("Please run the benchmark in your Streamlit app first.")
    exit()

# Extract data
queries = data['queries']

# ============ GRAPH 1: Real Response Times ============
print("\n1. Generating Response Time graph...")

categories = [q['type'] for q in queries]
times = [q['time'] for q in queries]
colors = ['#4CAF50' if not q.get('llm_used', False) else '#2196F3' for q in queries]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, times, color=colors, alpha=0.8, edgecolor='black')

# Add value labels
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Response Time (seconds)', fontsize=12)
ax.set_xlabel('Query Type', fontsize=12)
ax.set_title(f'Real Response Times - {data["pdf_name"]}', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4CAF50', label='Non-AI Mode'),
    Patch(facecolor='#2196F3', label='AI Mode')
]
ax.legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('report_graphs/1_real_response_times.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 1_real_response_times.png")
plt.close()


# ============ GRAPH 2: Confidence Scores ============
print("2. Generating Confidence Scores graph...")

non_ai_queries = [q for q in queries if not q.get('llm_used', False)]
categories_conf = [q['type'] for q in non_ai_queries]
confidences = [q['confidence'] for q in non_ai_queries]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories_conf, confidences, color='#FF9800', alpha=0.8, edgecolor='black')

# Add value labels
for bar, conf in zip(bars, confidences):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{conf:.3f}', ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Confidence Score', fontsize=12)
ax.set_xlabel('Query Type', fontsize=12)
ax.set_title('Retrieval Confidence Scores', fontsize=14, fontweight='bold')
ax.axhline(y=0.42, color='r', linestyle='--', label='Threshold (0.42)')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('report_graphs/2_confidence_scores.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 2_confidence_scores.png")
plt.close()


# ============ GRAPH 3: System Stats ============
print("3. Generating System Stats graph...")

stats = {
    'PDF Pages': data['pages'],
    'Text Chunks': data['chunks'],
    'Queries Tested': len(queries),
    'Avg Response\nTime (s)': round(sum(times) / len(times), 2)
}

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(stats.keys(), stats.values(), 
              color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'], 
              alpha=0.8, edgecolor='black')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Count / Value', fontsize=12)
ax.set_title(f'System Performance Summary - {data["pdf_name"]}', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('report_graphs/3_system_stats.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved: 3_system_stats.png")
plt.close()


# ============ SUMMARY TABLE ============
print("\n📋 Creating summary table...")

print("\n" + "="*60)
print(f"BENCHMARK SUMMARY - {data['pdf_name']}")
print("="*60)
print(f"Timestamp: {data['timestamp']}")
print(f"PDF Pages: {data['pages']}")
print(f"Text Chunks: {data['chunks']}")
print(f"\nQuery Results:")
print("-"*60)
for q in queries:
    print(f"{q['type']:15} | {q['time']:6.2f}s | Conf: {q['confidence']:.3f}")
print("="*60)
print(f"Average Time: {sum(times)/len(times):.2f}s")
print(f"Fastest Query: {min(times):.2f}s")
print(f"Slowest Query: {max(times):.2f}s")
print("="*60)

print("\n✅ All graphs generated successfully!")
print("📁 Check the 'report_graphs' folder\n")
