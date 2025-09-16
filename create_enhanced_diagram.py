#!/usr/bin/env python3
"""
Create enhanced system architecture diagram for Sundew algorithms.
Shows the modular plugin architecture and component relationships.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Use non-interactive backend
plt.switch_backend('Agg')

def create_enhanced_architecture_diagram():
    """Create comprehensive enhanced system architecture diagram."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'Enhanced Sundew Algorithm Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(8, 11, 'Modular Research-Grade System with Pluggable Components',
            ha='center', va='center', fontsize=12, style='italic')

    # Define colors
    colors = {
        'input': '#E3F2FD',      # Light blue
        'significance': '#FFF3E0', # Light orange
        'gating': '#F3E5F5',      # Light purple
        'control': '#E8F5E8',     # Light green
        'energy': '#FFF8E1',      # Light yellow
        'monitoring': '#FCE4EC',  # Light pink
        'output': '#F1F8E9'       # Very light green
    }

    # Input Layer
    input_box = FancyBboxPatch(
        (0.5, 9), 3, 1.5, boxstyle="round,pad=0.1",
        facecolor=colors['input'], edgecolor='#1976D2', linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(2, 9.75, 'Input Features', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 9.4, 'magnitude, anomaly_score,\ncontext_relevance, urgency',
            ha='center', va='center', fontsize=9)

    # Significance Models Layer
    sig_linear = FancyBboxPatch(
        (5, 10), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['significance'], edgecolor='#F57C00', linewidth=1.5
    )
    ax.add_patch(sig_linear)
    ax.text(6.25, 10.4, 'Linear Significance', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(6.25, 10.15, 'Weighted sum', ha='center', va='center', fontsize=8)

    sig_neural = FancyBboxPatch(
        (5, 8.8), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['significance'], edgecolor='#F57C00', linewidth=1.5
    )
    ax.add_patch(sig_neural)
    ax.text(6.25, 9.2, 'Neural Significance', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(6.25, 8.95, 'NN + Temporal Attention', ha='center', va='center', fontsize=8)

    # Interface box
    sig_interface = FancyBboxPatch(
        (4.8, 8.6), 2.9, 2.2, boxstyle="round,pad=0.1",
        facecolor='white', edgecolor='#666', linewidth=2, linestyle='--'
    )
    ax.add_patch(sig_interface)
    ax.text(6.25, 8.2, 'SignificanceModel\nInterface', ha='center', va='center',
            fontsize=9, fontweight='bold', alpha=0.7)

    # Gating Strategies
    gating_temp = FancyBboxPatch(
        (8.5, 10), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['gating'], edgecolor='#7B1FA2', linewidth=1.5
    )
    ax.add_patch(gating_temp)
    ax.text(9.75, 10.4, 'Temperature Gating', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(9.75, 10.15, 'Sigmoid smoothing', ha='center', va='center', fontsize=8)

    gating_adaptive = FancyBboxPatch(
        (8.5, 8.8), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['gating'], edgecolor='#7B1FA2', linewidth=1.5
    )
    ax.add_patch(gating_adaptive)
    ax.text(9.75, 9.2, 'Adaptive Gating', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(9.75, 8.95, 'Context-aware', ha='center', va='center', fontsize=8)

    gating_interface = FancyBboxPatch(
        (8.3, 8.6), 2.9, 2.2, boxstyle="round,pad=0.1",
        facecolor='white', edgecolor='#666', linewidth=2, linestyle='--'
    )
    ax.add_patch(gating_interface)
    ax.text(9.75, 8.2, 'GatingStrategy\nInterface', ha='center', va='center',
            fontsize=9, fontweight='bold', alpha=0.7)

    # Control Policies
    control_pi = FancyBboxPatch(
        (12, 10), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['control'], edgecolor='#388E3C', linewidth=1.5
    )
    ax.add_patch(control_pi)
    ax.text(13.25, 10.4, 'PI Control', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(13.25, 10.15, 'Fast & stable', ha='center', va='center', fontsize=8)

    control_mpc = FancyBboxPatch(
        (12, 8.8), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['control'], edgecolor='#388E3C', linewidth=1.5
    )
    ax.add_patch(control_mpc)
    ax.text(13.25, 9.2, 'MPC Control', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(13.25, 8.95, 'Multi-objective opt.', ha='center', va='center', fontsize=8)

    control_interface = FancyBboxPatch(
        (11.8, 8.6), 2.9, 2.2, boxstyle="round,pad=0.1",
        facecolor='white', edgecolor='#666', linewidth=2, linestyle='--'
    )
    ax.add_patch(control_interface)
    ax.text(13.25, 8.2, 'ControlPolicy\nInterface', ha='center', va='center',
            fontsize=9, fontweight='bold', alpha=0.7)

    # Energy Models
    energy_simple = FancyBboxPatch(
        (5, 6.5), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['energy'], edgecolor='#F9A825', linewidth=1.5
    )
    ax.add_patch(energy_simple)
    ax.text(6.25, 6.9, 'Simple Energy', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(6.25, 6.65, 'Basic consumption', ha='center', va='center', fontsize=8)

    energy_realistic = FancyBboxPatch(
        (8.5, 6.5), 2.5, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['energy'], edgecolor='#F9A825', linewidth=1.5
    )
    ax.add_patch(energy_realistic)
    ax.text(9.75, 6.9, 'Realistic Energy', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(9.75, 6.65, 'Thermal + DVFS + HW', ha='center', va='center', fontsize=8)

    energy_interface = FancyBboxPatch(
        (4.8, 6.3), 6.4, 1.2, boxstyle="round,pad=0.1",
        facecolor='white', edgecolor='#666', linewidth=2, linestyle='--'
    )
    ax.add_patch(energy_interface)
    ax.text(8, 5.9, 'EnergyModel Interface', ha='center', va='center',
            fontsize=9, fontweight='bold', alpha=0.7)

    # Core Processing Engine
    core_engine = FancyBboxPatch(
        (6, 4), 4, 1.5, boxstyle="round,pad=0.1",
        facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=3
    )
    ax.add_patch(core_engine)
    ax.text(8, 4.9, 'Enhanced Sundew Core', ha='center', va='center',
            fontsize=14, fontweight='bold')
    ax.text(8, 4.5, 'Modular Processing Pipeline', ha='center', va='center', fontsize=11)
    ax.text(8, 4.2, 'Context → Significance → Gating → Control → Energy',
            ha='center', va='center', fontsize=9)

    # Monitoring System
    monitor_realtime = FancyBboxPatch(
        (12.5, 6), 3, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['monitoring'], edgecolor='#C2185B', linewidth=1.5
    )
    ax.add_patch(monitor_realtime)
    ax.text(14, 6.4, 'Real-time Monitor', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(14, 6.15, 'Alerts + Visualization', ha='center', va='center', fontsize=8)

    monitor_profiler = FancyBboxPatch(
        (12.5, 4.8), 3, 0.8, boxstyle="round,pad=0.05",
        facecolor=colors['monitoring'], edgecolor='#C2185B', linewidth=1.5
    )
    ax.add_patch(monitor_profiler)
    ax.text(14, 5.2, 'Performance Profiler', ha='center', va='center',
            fontsize=10, fontweight='bold')
    ax.text(14, 4.95, 'CPU + Memory + Energy', ha='center', va='center', fontsize=8)

    # Output Layer
    output_decision = FancyBboxPatch(
        (2, 2), 3, 1, boxstyle="round,pad=0.05",
        facecolor=colors['output'], edgecolor='#689F38', linewidth=1.5
    )
    ax.add_patch(output_decision)
    ax.text(3.5, 2.5, 'Processing Decision', ha='center', va='center',
            fontsize=11, fontweight='bold')

    output_metrics = FancyBboxPatch(
        (6, 2), 4, 1, boxstyle="round,pad=0.05",
        facecolor=colors['output'], edgecolor='#689F38', linewidth=1.5
    )
    ax.add_patch(output_metrics)
    ax.text(8, 2.5, 'Comprehensive Metrics & Analysis', ha='center', va='center',
            fontsize=11, fontweight='bold')

    output_deployment = FancyBboxPatch(
        (11, 2), 3, 1, boxstyle="round,pad=0.05",
        facecolor=colors['output'], edgecolor='#689F38', linewidth=1.5
    )
    ax.add_patch(output_deployment)
    ax.text(12.5, 2.5, 'Production Ready', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Performance Metrics Box
    perf_box = FancyBboxPatch(
        (0.5, 0.2), 6, 1.3, boxstyle="round,pad=0.1",
        facecolor='#FFF3E0', edgecolor='#FF8F00', linewidth=2
    )
    ax.add_patch(perf_box)
    ax.text(3.5, 1.2, 'Performance Achievements', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(
        3.5, 0.85, '• Enhanced Neural+PI: 8.0/10 research quality',
        ha='center', va='center', fontsize=9
    )
    ax.text(
        3.5, 0.65, '• 99.5% energy savings (vs 84% original)',
        ha='center', va='center', fontsize=9
    )
    ax.text(3.5, 0.45, '• Comprehensive stability analysis', ha='center', va='center', fontsize=9)

    # Architecture Features Box
    arch_box = FancyBboxPatch(
        (7.5, 0.2), 8, 1.3, boxstyle="round,pad=0.1",
        facecolor='#E8F5E8', edgecolor='#4CAF50', linewidth=2
    )
    ax.add_patch(arch_box)
    ax.text(11.5, 1.2, 'Enhanced Architecture Features', ha='center', va='center',
            fontsize=12, fontweight='bold')
    ax.text(11.5, 0.9, '• Pluggable component interfaces    • Production deployment tools',
            ha='center', va='center', fontsize=9)
    ax.text(11.5, 0.7, '• Neural networks with attention    • Real-time monitoring & alerts',
            ha='center', va='center', fontsize=9)
    ax.text(11.5, 0.5, '• MPC with stability analysis       • Multi-platform energy models',
            ha='center', va='center', fontsize=9)

    # Draw connections
    # Input to significance models
    ax.arrow(3.5, 9.5, 1.2, 0.3, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)
    ax.arrow(3.5, 9.5, 1.2, -0.3, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)

    # Significance to gating
    ax.arrow(7.5, 9.4, 0.8, 0.3, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)
    ax.arrow(7.5, 9.4, 0.8, -0.3, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)

    # Gating to control
    ax.arrow(11, 9.4, 0.8, 0.3, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)
    ax.arrow(11, 9.4, 0.8, -0.3, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)

    # To core engine
    ax.arrow(8, 8.2, 0, -2.5, head_width=0.15, head_length=0.15,
             fc='#3F51B5', ec='#3F51B5', linewidth=2)

    # Energy models to core
    ax.arrow(8, 6.3, 0, -0.6, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)

    # Core to monitoring
    ax.arrow(10, 4.7, 2.3, 0.8, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)
    ax.arrow(10, 4.7, 2.3, 0, head_width=0.1, head_length=0.1,
             fc='black', ec='black', alpha=0.7)

    # Core to outputs
    ax.arrow(6.5, 4, -3, -1.3, head_width=0.1, head_length=0.15,
             fc='#689F38', ec='#689F38', linewidth=2)
    ax.arrow(8, 4, 0, -0.8, head_width=0.15, head_length=0.15,
             fc='#689F38', ec='#689F38', linewidth=2)
    ax.arrow(9.5, 4, 1, -1.3, head_width=0.1, head_length=0.15,
             fc='#689F38', ec='#689F38', linewidth=2)

    plt.tight_layout()
    return fig

def main():
    """Create and save the enhanced architecture diagram."""
    print("Creating enhanced Sundew architecture diagram...")

    fig = create_enhanced_architecture_diagram()

    # Save the diagram
    output_path = "assets/enhanced_architecture_diagram.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"Enhanced architecture diagram saved to: {output_path}")

    # Also save as SVG for scalability
    svg_path = "assets/enhanced_architecture_diagram.svg"
    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"SVG version saved to: {svg_path}")

    plt.close(fig)
    print("Diagram creation complete!")

if __name__ == "__main__":
    main()
