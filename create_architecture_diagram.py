#!/usr/bin/env python3
"""
Create comprehensive architecture diagrams for Sundew Algorithms application.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

def create_system_architecture_diagram():
    """Create a comprehensive system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color scheme
    colors = {
        'core': '#2E8B57',       # SeaGreen
        'interface': '#4682B4',   # SteelBlue
        'data': '#DAA520',       # GoldenRod
        'benchmark': '#8B4513',  # SaddleBrown
        'runtime': '#9370DB',    # MediumPurple
        'config': '#DC143C',     # Crimson
        'tools': '#FF8C00'       # DarkOrange
    }

    # Title
    ax.text(5, 9.5, 'Sundew Algorithms - System Architecture',
            fontsize=20, fontweight='bold', ha='center')

    # Core Algorithm Layer
    core_box = FancyBboxPatch((0.5, 7), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['core'], alpha=0.7, edgecolor='black')
    ax.add_patch(core_box)
    ax.text(2, 7.75, 'Core Algorithm\n(SundewAlgorithm)',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Sub-components of core
    ax.text(0.7, 7.3, '• PI Controller\n• Energy Pressure\n• Gating Logic',
            fontsize=9, ha='left', va='center')

    # Configuration Layer
    config_box = FancyBboxPatch((4.5, 7), 2.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['config'], alpha=0.7, edgecolor='black')
    ax.add_patch(config_box)
    ax.text(5.75, 7.75, 'Configuration\n(SundewConfig)',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Runtime Pipeline Layer
    runtime_box = FancyBboxPatch((7.5, 7), 2, 1.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['runtime'], alpha=0.7, edgecolor='black')
    ax.add_patch(runtime_box)
    ax.text(8.5, 7.75, 'Pipeline\nRuntime',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Interface Layer
    interface_y = 5.5
    interfaces = [
        ('SignificanceModel', 1),
        ('ControlPolicy', 3),
        ('GatingStrategy', 5),
        ('EnergyModel', 7)
    ]

    for name, x in interfaces:
        interface_box = FancyBboxPatch((x-0.75, interface_y-0.3), 1.5, 0.6,
                                       boxstyle="round,pad=0.05",
                                       facecolor=colors['interface'], alpha=0.7, edgecolor='black')
        ax.add_patch(interface_box)
        ax.text(x, interface_y, name, fontsize=9, fontweight='bold',
                ha='center', va='center', color='white')

    # Data Layer
    data_y = 4
    datasets = [
        ('Breast Cancer', 1),
        ('Heart Disease', 2.5),
        ('IoT Sensors', 4),
        ('ECG/MIT-BIH', 5.5),
        ('Financial', 7),
        ('Network Security', 8.5)
    ]

    for name, x in datasets:
        data_box = FancyBboxPatch((x-0.5, data_y-0.25), 1, 0.5,
                                  boxstyle="round,pad=0.05",
                                  facecolor=colors['data'], alpha=0.7, edgecolor='black')
        ax.add_patch(data_box)
        ax.text(x, data_y, name, fontsize=8, ha='center', va='center', rotation=45)

    # Benchmarking Layer
    benchmark_box = FancyBboxPatch((0.5, 2.5), 4, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['benchmark'], alpha=0.7, edgecolor='black')
    ax.add_patch(benchmark_box)
    ax.text(2.5, 3, 'Benchmarking & Analysis',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    benchmark_items = ['Dataset Suite', 'Ablation Studies', 'Bootstrap Metrics', 'Adversarial Testing']
    for i, item in enumerate(benchmark_items):
        ax.text(0.7 + i*0.9, 2.7, item, fontsize=8, ha='center', va='center', rotation=0)

    # Tools Layer
    tools_box = FancyBboxPatch((5.5, 2.5), 4, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['tools'], alpha=0.7, edgecolor='black')
    ax.add_patch(tools_box)
    ax.text(7.5, 3, 'Tools & Utilities',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    tool_items = ['Power Capture', 'Runtime Monitor', 'Plotting', 'CLI Interface']
    for i, item in enumerate(tool_items):
        ax.text(5.7 + i*0.9, 2.7, item, fontsize=8, ha='center', va='center', rotation=0)

    # Results Layer
    results_box = FancyBboxPatch((1, 0.5), 8, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#696969', alpha=0.7, edgecolor='black')
    ax.add_patch(results_box)
    ax.text(5, 1, 'Results & Outputs\n(CSV, JSON, PNG plots, Bootstrap CIs, Layered Precision)',
            fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    # Add arrows to show data flow
    arrows = [
        # Core to interfaces
        ((2, 7), (2, 6)),
        ((5.75, 7), (4, 6)),
        ((8.5, 7), (6, 6)),
        # Interfaces to data
        ((2, 5.2), (2, 4.5)),
        ((4, 5.2), (4, 4.5)),
        ((6, 5.2), (6, 4.5)),
        # Data to benchmarking
        ((2.5, 3.8), (2.5, 3.5)),
        ((7.5, 3.8), (7.5, 3.5)),
        # Benchmarking to results
        ((2.5, 2.5), (2.5, 1.5)),
        ((7.5, 2.5), (7.5, 1.5))
    ]

    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", alpha=0.6)
        ax.add_patch(arrow)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['core'], alpha=0.7, label='Core Algorithm'),
        mpatches.Patch(color=colors['interface'], alpha=0.7, label='Interfaces'),
        mpatches.Patch(color=colors['data'], alpha=0.7, label='Data Sources'),
        mpatches.Patch(color=colors['benchmark'], alpha=0.7, label='Benchmarking'),
        mpatches.Patch(color=colors['tools'], alpha=0.7, label='Tools'),
        mpatches.Patch(color=colors['config'], alpha=0.7, label='Configuration'),
        mpatches.Patch(color=colors['runtime'], alpha=0.7, label='Runtime')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig('assets/sundew_system_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/sundew_system_architecture.svg', bbox_inches='tight')
    plt.close()

def create_algorithm_pipeline_diagram():
    """Create detailed algorithm pipeline flow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Sundew Algorithm - Processing Pipeline',
            fontsize=18, fontweight='bold', ha='center')

    # Pipeline stages
    stages = [
        {
            'name': 'Event Input',
            'pos': (1, 8),
            'size': (1.5, 0.8),
            'color': '#4CAF50',
            'details': 'Raw sensor data\nTime series events'
        },
        {
            'name': 'Significance\nCalculation',
            'pos': (1, 6.5),
            'size': (1.5, 0.8),
            'color': '#2196F3',
            'details': 'w_magnitude × magnitude +\nw_anomaly × anomaly +\nw_context × context +\nw_urgency × urgency'
        },
        {
            'name': 'PI Controller\n& Threshold',
            'pos': (4, 8),
            'size': (1.5, 0.8),
            'color': '#FF9800',
            'details': 'error = target_rate - actual_rate\nthreshold += Kp×error + Ki×∫error'
        },
        {
            'name': 'Energy\nAccounting',
            'pos': (4, 6.5),
            'size': (1.5, 0.8),
            'color': '#9C27B0',
            'details': 'Energy consumption\nRegeneration during dormancy\nCap-aware management'
        },
        {
            'name': 'Gating Decision',
            'pos': (7, 7.25),
            'size': (1.5, 0.8),
            'color': '#F44336',
            'details': 'P(activate) = sigmoid(\n(significance - threshold)\n/ temperature)'
        },
        {
            'name': 'Process Event',
            'pos': (8.5, 5.5),
            'size': (1.2, 0.6),
            'color': '#4CAF50',
            'details': 'Execute processing\nUpdate metrics'
        },
        {
            'name': 'Dormant Mode',
            'pos': (5.5, 5.5),
            'size': (1.2, 0.6),
            'color': '#607D8B',
            'details': 'Energy regeneration\nLow-power state'
        }
    ]

    # Draw stages
    for stage in stages:
        box = FancyBboxPatch(stage['pos'], stage['size'][0], stage['size'][1],
                            boxstyle="round,pad=0.1",
                            facecolor=stage['color'], alpha=0.8, edgecolor='black')
        ax.add_patch(box)

        # Stage name
        ax.text(stage['pos'][0] + stage['size'][0]/2,
                stage['pos'][1] + stage['size'][1]/2 + 0.15,
                stage['name'], fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')

        # Stage details
        ax.text(stage['pos'][0] + stage['size'][0]/2,
                stage['pos'][1] + stage['size'][1]/2 - 0.15,
                stage['details'], fontsize=8,
                ha='center', va='center', color='white')

    # Decision diamond for activation
    decision_points = np.array([[7.75, 6.5], [8.25, 6.8], [7.75, 7.1], [7.25, 6.8]])
    decision = plt.Polygon(decision_points, facecolor='#FFEB3B', alpha=0.8, edgecolor='black')
    ax.add_patch(decision)
    ax.text(7.75, 6.8, 'Activate?', fontsize=10, fontweight='bold', ha='center', va='center')

    # Flow arrows
    flows = [
        # Main pipeline flow
        ((1.75, 8), (1.75, 7.3)),  # Input to Significance
        ((1.75, 6.5), (3.8, 7.2)),  # Significance to Controller
        ((4.75, 8), (4.75, 7.3)),  # Controller to Energy
        ((5.5, 7.25), (7, 7.25)),  # Energy to Gating

        # Decision branches
        ((7.75, 6.5), (8.5, 6.1)),  # Yes to Process
        ((7.25, 6.8), (6.7, 6.1)),  # No to Dormant

        # Feedback loops
        ((8.5, 5.5), (2.5, 5), (2.5, 6.5), (2.5, 6.5)),  # Process feedback
        ((5.5, 5.5), (2.5, 5.2), (2.5, 6.5), (2.5, 6.5))   # Dormant feedback
    ]

    # Draw straight arrows
    straight_arrows = [
        ((1.75, 8), (1.75, 7.3)),
        ((1.75, 6.5), (3.8, 7.2)),
        ((4.75, 8), (4.75, 7.3)),
        ((5.5, 7.25), (7, 7.25)),
        ((7.75, 6.5), (8.5, 6.1)),
        ((7.25, 6.8), (6.7, 6.1))
    ]

    for start, end in straight_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", alpha=0.8)
        ax.add_patch(arrow)

    # Curved feedback arrows
    feedback_arrow1 = ConnectionPatch((8.5, 5.5), (1.75, 6.5), "data", "data",
                                     arrowstyle="->", shrinkA=5, shrinkB=5,
                                     mutation_scale=15, fc="red", alpha=0.6,
                                     connectionstyle="arc3,rad=0.5")
    ax.add_patch(feedback_arrow1)

    feedback_arrow2 = ConnectionPatch((5.5, 5.5), (4.75, 6.5), "data", "data",
                                     arrowstyle="->", shrinkA=5, shrinkB=5,
                                     mutation_scale=15, fc="blue", alpha=0.6,
                                     connectionstyle="arc3,rad=-0.3")
    ax.add_patch(feedback_arrow2)

    # Add labels
    ax.text(8.1, 6.3, 'YES', fontsize=10, fontweight='bold', color='green')
    ax.text(6.8, 6.6, 'NO', fontsize=10, fontweight='bold', color='red')
    ax.text(4, 5, 'Feedback Loop\n(Rate Control)', fontsize=9, ha='center', color='red')

    # Add metrics box
    metrics_box = FancyBboxPatch((0.5, 3), 9, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#E0E0E0', alpha=0.8, edgecolor='black')
    ax.add_patch(metrics_box)
    ax.text(5, 4, 'Key Metrics Tracked', fontsize=14, fontweight='bold', ha='center')

    metrics_text = """
• Activation Rate (EMA)           • Energy Savings %              • Processing Time
• Threshold History              • F1 Score & Recall             • Bootstrap Confidence Intervals
• Precision (with/without layers) • Probe Activations            • Controller Error History
    """
    ax.text(5, 3.4, metrics_text, fontsize=10, ha='center', va='center')

    # Add configuration note
    config_box = FancyBboxPatch((0.5, 1), 9, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFF3E0', alpha=0.8, edgecolor='black')
    ax.add_patch(config_box)
    ax.text(5, 2, 'Configuration Presets', fontsize=14, fontweight='bold', ha='center')

    config_text = """
• tuned_v2: Balanced performance        • custom_health_hd82: Heart disease optimized (82% savings)
• auto_tuned: Dataset adaptive          • custom_breast_probe: Breast cancer with probes (77% savings)
• aggressive: High activation           • energy_saver: Maximum efficiency (>90% savings)
    """
    ax.text(5, 1.4, config_text, fontsize=10, ha='center', va='center')

    plt.tight_layout()
    plt.savefig('assets/sundew_algorithm_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/sundew_algorithm_pipeline.svg', bbox_inches='tight')
    plt.close()

def create_data_flow_diagram():
    """Create data flow and results diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'Sundew Algorithms - Data Flow & Results Pipeline',
            fontsize=18, fontweight='bold', ha='center')

    # Input datasets
    datasets = [
        ('Breast Cancer\n(569 samples)', 1, 8),
        ('Heart Disease\n(303 samples)', 3, 8),
        ('IoT Sensors\n(5000 samples)', 5, 8),
        ('MIT-BIH ECG\n(109,446 samples)', 7, 8),
        ('Financial\n(1000 samples)', 9, 8),
        ('Network Security\n(2000 samples)', 11, 8)
    ]

    for name, x, y in datasets:
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor='#4CAF50', alpha=0.8, edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=9, fontweight='bold', ha='center', va='center', color='white')

    # Processing layer
    processing_box = FancyBboxPatch((2, 6), 8, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#2196F3', alpha=0.8, edgecolor='black')
    ax.add_patch(processing_box)
    ax.text(6, 6.5, 'Multi-Preset Processing Pipeline',
            fontsize=14, fontweight='bold', ha='center', va='center', color='white')

    # Presets
    presets = ['tuned_v2', 'auto_tuned', 'aggressive', 'conservative', 'custom_health_hd82', 'custom_breast_probe']
    for i, preset in enumerate(presets):
        ax.text(2.5 + i*1.2, 6.2, preset, fontsize=8, ha='center', va='center', color='white')

    # Analysis branches
    analyses = [
        ('Dataset Suite\nBenchmarking', 2, 4.5, '#FF9800'),
        ('Ablation\nStudies', 4, 4.5, '#9C27B0'),
        ('Bootstrap\nMetrics', 6, 4.5, '#F44336'),
        ('Layered\nClassifier', 8, 4.5, '#795548'),
        ('Adversarial\nTesting', 10, 4.5, '#607D8B')
    ]

    for name, x, y, color in analyses:
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color, alpha=0.8, edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold', ha='center', va='center', color='white')

    # Results outputs
    results = [
        ('CSV Results\n(dataset_suite_full.csv)', 1.5, 2.5, '#4CAF50'),
        ('JSON Telemetry\n(detailed logs)', 3.5, 2.5, '#2196F3'),
        ('PNG Plots\n(visualizations)', 5.5, 2.5, '#FF9800'),
        ('Statistical CIs\n(bootstrap)', 7.5, 2.5, '#9C27B0'),
        ('Performance Reports\n(markdown)', 9.5, 2.5, '#F44336'),
        ('Hardware Telemetry\n(power logs)', 11, 2.5, '#607D8B')
    ]

    for name, x, y, color in results:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=8, fontweight='bold', ha='center', va='center', color='white')

    # Storage locations
    storage_box = FancyBboxPatch((1, 0.5), 10, 1,
                                boxstyle="round,pad=0.1",
                                facecolor='#37474F', alpha=0.8, edgecolor='black')
    ax.add_patch(storage_box)
    ax.text(6, 1, 'Storage Locations', fontsize=12, fontweight='bold', ha='center', va='center', color='white')

    storage_text = "data/results/ • assets/ • results/plots/ • docs/ • examples/"
    ax.text(6, 0.7, storage_text, fontsize=10, ha='center', va='center', color='white')

    # Add flow arrows
    # Datasets to processing
    for _, x, y in datasets:
        arrow = ConnectionPatch((x, y-0.4), (6, 6), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="black", alpha=0.6)
        ax.add_patch(arrow)

    # Processing to analyses
    for _, x, y, _ in analyses:
        arrow = ConnectionPatch((6, 6), (x, y+0.4), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="black", alpha=0.6)
        ax.add_patch(arrow)

    # Analyses to results
    analysis_results_mapping = [(2, 1.5), (4, 3.5), (6, 7.5), (8, 5.5), (10, 9.5)]
    for (ax_pos, res_pos) in analysis_results_mapping:
        arrow = ConnectionPatch((ax_pos, 4.1), (res_pos, 2.8), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="black", alpha=0.6)
        ax.add_patch(arrow)

    # Results to storage
    for _, x, y, _ in results:
        arrow = ConnectionPatch((x, y-0.3), (6, 1.5), "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="black", alpha=0.6)
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig('assets/sundew_data_flow.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/sundew_data_flow.svg', bbox_inches='tight')
    plt.close()

def create_performance_summary_diagram():
    """Create a performance summary visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Sundew Algorithms - Performance Summary Dashboard', fontsize=20, fontweight='bold')

    # Dataset performance comparison
    datasets = ['Breast Cancer', 'Heart Disease', 'IoT Sensors', 'Network Security', 'Financial', 'MIT-BIH ECG']
    recalls = [0.118, 0.196, 0.500, 0.233, 0.164, 0.218]
    savings = [77.2, 82.0, 88.2, 89.2, 90.1, 88.8]

    colors = ['#E91E63', '#9C27B0', '#3F51B5', '#2196F3', '#00BCD4', '#4CAF50']

    # Recall vs Savings scatter
    ax1.scatter(recalls, savings, c=colors, s=150, alpha=0.8, edgecolors='black')
    ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy Savings (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Recall vs Energy Savings by Dataset', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add dataset labels
    for i, (recall, saving, dataset) in enumerate(zip(recalls, savings, datasets)):
        ax1.annotate(dataset, (recall, saving), xytext=(5, 5),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))

    # Preset comparison
    presets = ['tuned_v2', 'auto_tuned', 'aggressive', 'conservative', 'custom_health_hd82']
    avg_savings = [90.1, 87.8, 86.4, 92.3, 84.4]
    avg_recalls = [0.151, 0.197, 0.226, 0.081, 0.182]

    bars = ax2.bar(range(len(presets)), avg_savings, color=['#FF9800', '#2196F3', '#F44336', '#4CAF50', '#9C27B0'], alpha=0.8)
    ax2.set_xlabel('Presets', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Energy Savings (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Energy Savings by Preset', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(presets)))
    ax2.set_xticklabels(presets, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, avg_savings):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Layered precision improvement
    datasets_layer = ['Financial', 'Network Security', 'IoT Sensors', 'MIT-BIH ECG']
    baseline_precision = [0.220, 0.461, 0.670, 0.340]
    layered_precision = [1.000, 1.000, 1.000, 1.000]

    x = np.arange(len(datasets_layer))
    width = 0.35

    bars1 = ax3.bar(x - width/2, baseline_precision, width, label='Baseline', color='#FF5722', alpha=0.8)
    bars2 = ax3.bar(x + width/2, layered_precision, width, label='With Layered Classifier', color='#4CAF50', alpha=0.8)

    ax3.set_xlabel('Datasets', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision Improvement with Layered Classifier', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(datasets_layer, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.1)

    # Bootstrap confidence intervals
    ci_data = {
        'Breast Cancer': (0.385, 0.294, 0.475),
        'Heart Disease': (0.755, 0.680, 0.828),
        'IoT Sensors': (0.670, 0.574, 0.758),
        'Network Security': (0.461, 0.355, 0.562)
    }

    datasets_ci = list(ci_data.keys())
    means = [ci_data[d][0] for d in datasets_ci]
    lower_errors = [ci_data[d][0] - ci_data[d][1] for d in datasets_ci]
    upper_errors = [ci_data[d][2] - ci_data[d][0] for d in datasets_ci]

    ax4.errorbar(range(len(datasets_ci)), means,
                yerr=[lower_errors, upper_errors],
                fmt='o', capsize=5, capthick=2, markersize=8,
                color='#2196F3', ecolor='#FF9800', alpha=0.8)

    ax4.set_xlabel('Datasets', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax4.set_title('Bootstrap Confidence Intervals (95%)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(datasets_ci)))
    ax4.set_xticklabels(datasets_ci, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('assets/sundew_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/sundew_performance_dashboard.svg', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Creating Sundew Algorithms architecture diagrams...")

    # Create all diagrams
    create_system_architecture_diagram()
    print("System architecture diagram created")

    create_algorithm_pipeline_diagram()
    print("Algorithm pipeline diagram created")

    create_data_flow_diagram()
    print("Data flow diagram created")

    create_performance_summary_diagram()
    print("Performance summary dashboard created")

    print("\nAll diagrams saved to assets/ directory:")
    print("- sundew_system_architecture.png/svg")
    print("- sundew_algorithm_pipeline.png/svg")
    print("- sundew_data_flow.png/svg")
    print("- sundew_performance_dashboard.png/svg")
