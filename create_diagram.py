#!/usr/bin/env python3
"""
Create a comprehensive Sundew project diagram showing architecture and research pipeline.
"""

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def create_sundew_diagram():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 20))
    fig.suptitle('Sundew Algorithms: Architecture & Research Pipeline', fontsize=20, fontweight='bold')

    # Colors
    colors = {
        'input': '#E3F2FD',
        'core': '#FFF3E0',
        'control': '#E8F5E8',
        'output': '#FCE4EC',
        'research': '#F3E5F5',
        'tools': '#E0F2F1'
    }

    # ========== TOP DIAGRAM: CORE ALGORITHM ARCHITECTURE ==========
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 8)
    ax1.set_title('Core Algorithm Architecture', fontsize=16, fontweight='bold', pad=20)

    # Input stream
    input_box = FancyBboxPatch((0.5, 6), 2, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(1.5, 6.5, 'Input Stream\n{magnitude, anomaly,\ncontext, urgency}',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Feature processing
    feature_box = FancyBboxPatch((3.5, 6), 2, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['core'], edgecolor='black', linewidth=2)
    ax1.add_patch(feature_box)
    ax1.text(4.5, 6.5, 'Significance\nScoring\ns = Σwᵢ·fᵢ',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Gating decision
    gate_box = FancyBboxPatch((6.5, 6), 2, 1, boxstyle="round,pad=0.1",
                             facecolor=colors['core'], edgecolor='black', linewidth=2)
    ax1.add_patch(gate_box)
    ax1.text(7.5, 6.5, 'Temperature\nGating\nP = σ((s-θ)/T)',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Processing/Skip decision
    decision_box = FancyBboxPatch((9.5, 6), 2, 1, boxstyle="round,pad=0.1",
                                 facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax1.add_patch(decision_box)
    ax1.text(10.5, 6.5, 'Process/Skip\nDecision',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # PI Controller
    pi_box = FancyBboxPatch((2, 4), 3, 1, boxstyle="round,pad=0.1",
                           facecolor=colors['control'], edgecolor='black', linewidth=2)
    ax1.add_patch(pi_box)
    ax1.text(3.5, 4.5, 'PI Controller\nθ ← θ + Kₚ·err + Kᵢ·∫err',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Energy system
    energy_box = FancyBboxPatch((6, 4), 3, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['control'], edgecolor='black', linewidth=2)
    ax1.add_patch(energy_box)
    ax1.text(7.5, 4.5, 'Energy Account\nE ← E - cost + regen\nPressure: λ(1-E/Eₘₐₓ)',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Activation rate monitor
    monitor_box = FancyBboxPatch((2, 2), 3, 1, boxstyle="round,pad=0.1",
                                facecolor=colors['control'], edgecolor='black', linewidth=2)
    ax1.add_patch(monitor_box)
    ax1.text(3.5, 2.5, 'Activation Rate\nEMA Monitor\nρ ← α·current + (1-α)·ρ',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Target rate
    target_box = FancyBboxPatch((6, 2), 3, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['control'], edgecolor='black', linewidth=2)
    ax1.add_patch(target_box)
    ax1.text(7.5, 2.5, 'Target Rate\nρ* (configurable)\nError: err = ρ* - ρ',
             ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrows = [
        # Main flow
        ((2.5, 6.5), (3.5, 6.5)),
        ((5.5, 6.5), (6.5, 6.5)),
        ((8.5, 6.5), (9.5, 6.5)),
        # Feedback loops
        ((7.5, 6), (7.5, 5)),  # gating to energy
        ((7.5, 4), (7.5, 3)),  # energy to target
        ((6, 2.5), (5, 2.5)),  # target to monitor
        ((3.5, 3), (3.5, 4)),  # monitor to PI
        ((3.5, 5), (3.5, 6)),  # PI to significance (threshold feedback)
    ]

    for start, end in arrows:
        ax1.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    # ========== BOTTOM DIAGRAM: RESEARCH & DEVELOPMENT PIPELINE ==========
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 10)
    ax2.set_title('Research & Development Pipeline', fontsize=16, fontweight='bold', pad=20)

    # Development track
    dev_boxes = [
        ((1, 8), 'Data\nIngestion\n(MIT-BIH)', colors['input']),
        ((3.5, 8), 'Feature\nExtraction', colors['core']),
        ((6, 8), 'Algorithm\nDevelopment', colors['core']),
        ((8.5, 8), 'Parameter\nTuning', colors['control']),
        ((11, 8), 'Validation\n& Testing', colors['output'])
    ]

    for (x, y), text, color in dev_boxes:
        box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Research quality track
    research_boxes = [
        ((1, 6), 'Multi-seed\nRuns', colors['research']),
        ((3.5, 6), 'Cross-\nValidation', colors['research']),
        ((6, 6), 'Ablation\nStudies', colors['research']),
        ((8.5, 6), 'Statistical\nAnalysis', colors['research']),
        ((11, 6), 'Publication\nResults', colors['research'])
    ]

    for (x, y), text, color in research_boxes:
        box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Tools & infrastructure
    tools_boxes = [
        ((1, 4), 'UV Package\nManager', colors['tools']),
        ((3.5, 4), 'Pytest\nTesting', colors['tools']),
        ((6, 4), 'Ruff\nLinting', colors['tools']),
        ((8.5, 4), 'MyPy\nType Check', colors['tools']),
        ((11, 4), 'GitHub\nCI/CD', colors['tools'])
    ]

    for (x, y), text, color in tools_boxes:
        box = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(box)
        ax2.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Metrics & evaluation
    metrics_box = FancyBboxPatch((1, 2), 10, 1, boxstyle="round,pad=0.1",
                                facecolor='#FFFDE7', edgecolor='black', linewidth=2)
    ax2.add_patch(metrics_box)
    ax2.text(6, 2.5, 'Key Metrics: Activation Rate | Energy Savings | F1-Score | Precision/Recall | AUROC | Stability Index',
             ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw connecting arrows for pipeline flow
    pipeline_arrows = [
        # Development flow
        ((1.7, 8), (2.8, 8)),
        ((4.2, 8), (5.3, 8)),
        ((6.7, 8), (7.8, 8)),
        ((9.2, 8), (10.3, 8)),
        # Research flow
        ((1.7, 6), (2.8, 6)),
        ((4.2, 6), (5.3, 6)),
        ((6.7, 6), (7.8, 6)),
        ((9.2, 6), (10.3, 6)),
        # Tools flow
        ((1.7, 4), (2.8, 4)),
        ((4.2, 4), (5.3, 4)),
        ((6.7, 4), (7.8, 4)),
        ((9.2, 4), (10.3, 4)),
        # Vertical connections
        ((6, 7.5), (6, 6.5)),  # dev to research
        ((6, 5.5), (6, 4.5)),  # research to tools
        ((6, 3.5), (6, 3)),    # tools to metrics
    ]

    for start, end in pipeline_arrows:
        ax2.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='darkgreen'))

    # Add quality score indicator
    quality_box = FancyBboxPatch((12.5, 6), 1.2, 3, boxstyle="round,pad=0.1",
                                facecolor='#FFE0B2', edgecolor='red', linewidth=3)
    ax2.add_patch(quality_box)
    ax2.text(13.1, 7.5, 'Current:\n6.5-7/10\n\nTarget:\n8.5+/10',
             ha='center', va='center', fontsize=10, fontweight='bold', color='red')

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Processing'),
        mpatches.Patch(color=colors['core'], label='Core Algorithm'),
        mpatches.Patch(color=colors['control'], label='Control System'),
        mpatches.Patch(color=colors['output'], label='Output/Decision'),
        mpatches.Patch(color=colors['research'], label='Research Pipeline'),
        mpatches.Patch(color=colors['tools'], label='Development Tools')
    ]
    ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig('assets/sundew_comprehensive_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('assets/sundew_comprehensive_diagram.svg', bbox_inches='tight')
    print("Diagram saved as assets/sundew_comprehensive_diagram.png and .svg")

if __name__ == "__main__":
    create_sundew_diagram()
