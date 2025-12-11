import matplotlib.pyplot as plt
import numpy as np

# Data Setup
conditions = ['Baseline (Auto)', 'Single-HITL', 'Multi-Agent HITL']
colors = ["#e02514", "#47df19", "#1180db"]

# Metric Data Dictionary
# (Metric Name, Y-Label, Data List [Baseline, Single, Multi])
# Note: Converting rates to percentages where appropriate
data = {
    'Average Token Usage': ('Count', [833.12, 3961.27, 62325.88]),
    'Average Elapsed Time': ('Seconds', [9.05, 21.05, 184.55]),
    'Format Success Rate': ('Percentage (%)', [100.0, 100.0, 100.0]), # Derived from 0.0 error rate
    'Citation Validity Rate': ('Percentage (%)', [0.0, 0.0, 7.37]), # 0.07368 * 100
    'Verifiability': ('Score (1-5)', [3.85, 4.58, 4.76]),
    'Logical Soundness': ('Score (1-5)', [5.00, 4.99, 4.98]),
    'Novelty': ('Score (1-5)', [3.16, 3.31, 3.36]),
    'Relevance': ('Score (1-5)', [5.00, 5.00, 4.99]),
    'Clarity': ('Score (1-5)', [5.00, 4.97, 4.98])
}

# Create Figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 22))
# Main title larger than subtitles
fig.suptitle('Performance Comparison Across Workflows', fontsize=20, weight='bold', y=0.995)

axes_flat = axes.flatten()
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

for i, (metric, (ylabel, values)) in enumerate(data.items()):
    ax = axes_flat[i]
    bars = ax.bar(conditions, values, color=colors)
    
    ax.set_title(metric, fontsize=18, weight='bold')

    # Annotate subplot with panel label (e.g., (a))
    ax.text(
        0.02,
        1.08,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='bottom',
    )
    
    ax.set_ylabel(ylabel, fontsize=18)
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Only keep x-axis tick labels for subplots in the bottom row
    if i < 6:
        ax.set_xticklabels([])
    
    for bar in bars:
        height = bar.get_height()
        # Adjusted offset logic for visibility
        offset = height * 0.02 
        if height == 0:
            offset = max(values) * 0.02 if max(values) > 0 else 0.1
            if offset == 0: offset = 0.1
            
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + offset, 
                f'{height:.2f}', ha='center', va='bottom', fontsize=18, weight='bold')

    # Extend y-axis significantly to fit the large labels on top
    if ylabel == 'Percentage (%)':
        ax.set_ylim(top=110)
    elif ylabel == 'Score (1-5)':
        ax.set_ylim(top=5.5)
    else:
        ax.set_ylim(top=max(values) * 1.1 if max(values) > 0 else 1.0)
        # ax.set_yticks(np.arange(0, max(values) * 1.3, step=max(values) / 5 if max(values) >= 10 else 1))

# Layout adjustment with padding for the rotated labels and titles
plt.tight_layout(rect=[0, 0.02, 1, 0.99])
# plt.show()
plt.savefig('results.png')