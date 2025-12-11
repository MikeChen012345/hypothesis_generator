import matplotlib.pyplot as plt
import json

# Data Dictionary Construction
# Parsing the provided file content directly into dictionaries since I cannot read files from disk in this environment directly 
# but I can simulate reading them from the provided text in the prompt.
# Wait, I can read files using the standard open() if they are in the working directory. 
# The system prompt implies files are uploaded. I will assume they are available at the given filenames.

filenames = {
    '17B': {'Baseline': 'metrics_summary_baseline_17B.json', 'Single-HITL': 'metrics_summary_llm_17B.json'},
    '27B': {'Baseline': 'metrics_summary_baseline_27B.json', 'Single-HITL': 'metrics_summary_llm_27B.json'},
    '70B': {'Baseline': 'metrics_summary_baseline_70B.json', 'Single-HITL': 'metrics_summary_llm_70B.json'},
    '120B': {'Baseline': 'metrics_summary_baseline_120B.json', 'Single-HITL': 'metrics_summary_llm_120B.json'}
}

# Metric keys to plot
metrics = [
    "average_token_usage",
    "average_elapsed_time",
    "format_error_rate",
    "citation_validity_rate",
    "verifiability",
    "logical_soundness",
    "novelty",
    "relevance",
    "clarity"
]

ylabels = {
    "average_token_usage": "Count",
    "average_elapsed_time": "Seconds",
    "format_error_rate": "Percentage (%)",
    "citation_validity_rate": "Percentage (%)",
    "verifiability": "Score (1-5)",
    "logical_soundness": "Score (1-5)",
    "novelty": "Score (1-5)",
    "relevance": "Score (1-5)",
    "clarity": "Score (1-5)"
}

# Initialize storage
results = {
    'Baseline': {m: [] for m in metrics},
    'Single-HITL': {m: [] for m in metrics}
}
model_sizes = [17, 27, 70, 120]

# Load Data
for size in model_sizes:
    size_str = f"{size}B"
    
    # Load Baseline
    with open("results/" + filenames[size_str]['Baseline'], 'r') as f:
        data_base = json.load(f)
        for m in metrics:
            if m == "format_error_rate":
                # Convert error rate to success rate percentage
                results['Baseline'][m].append((1.0 - data_base[m]) * 100)
            else:
                results['Baseline'][m].append(data_base[m])
            
    # Load Single-HITL
    with open("results/" + filenames[size_str]['Single-HITL'], 'r') as f:
        data_hitl = json.load(f)
        for m in metrics:
            if m == "format_error_rate":
                results['Single-HITL'][m].append((1.0 - data_hitl[m]) * 100)
            else:
                results['Single-HITL'][m].append(data_hitl[m])

# Plotting
# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Influence of Backbone Model Parameter Size on Performance', fontsize=20, 
             weight='bold', y=0.95)

axes_flat = axes.flatten()
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

for i, metric in enumerate(metrics):
    ax = axes_flat[i]
    ylabel = ylabels[metric]

    ax.set_ylabel(ylabel, fontsize=20)
    if ylabel == 'Percentage (%)':
        ax.set_ylim(top=110)
    elif ylabel == 'Score (1-5)':
        ax.set_ylim(top=5.5)
    else:
        ax.set_ylim(top=max(results['Baseline'][metric] + results['Single-HITL'][metric]) * 1.1 if max(results['Baseline'][metric] + results['Single-HITL'][metric]) > 0 else 1.0)
    
    ax.text(
        0.02,
        1.08,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=16,
        fontweight='bold',
        va='bottom',
    )

    # Plot Baseline
    ax.plot(model_sizes, results['Baseline'][metric], marker='o', linestyle='--', label='Baseline', color='gray', markersize=10)
    
    # Plot Single-HITL
    ax.plot(model_sizes, results['Single-HITL'][metric], marker='o', linestyle='-', label='Single-HITL', color='skyblue', markersize=10)
    
    # Styling
    ax.set_title(metric.replace('_', ' ').title() if metric != "format_error_rate" else "Format Success Rate", fontsize=20, weight='bold')
    ax.set_xlabel('Parameter Size (B)', fontsize=20)
    ax.set_xticks(model_sizes)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Only add legend to the first plot to avoid clutter, or maybe a single legend outside?
    # Let's add it to the first one for now.
    if i == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('ablation.png')