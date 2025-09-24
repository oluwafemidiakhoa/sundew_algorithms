#!/usr/bin/env python3
"""
Minimal Sundew Algorithm Demo
Shows the core concept without unnecessary complexity
"""

import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class GatingResult:
    """Result of a gating decision"""
    activated: bool
    significance: float
    threshold: float
    energy_saved: float

class SimpleSundew:
    """Minimal implementation of the Sundew algorithm"""

    def __init__(self, target_rate: float = 0.2):
        """
        Args:
            target_rate: Target fraction of inputs to process (0.2 = 20%)
        """
        self.target_rate = target_rate
        self.threshold = 0.5
        self.activation_history = []

        # PI controller parameters
        self.kp = 0.01  # Proportional gain
        self.ki = 0.001  # Integral gain
        self.error_sum = 0

        # Hysteresis to prevent oscillation
        self.hysteresis_gap = 0.02
        self.was_active = False

    def compute_significance(self, value: float, context: dict) -> float:
        """
        Compute significance score for an input

        Args:
            value: Input magnitude
            context: Additional features (anomaly, urgency, etc.)
        """
        # Simple weighted sum of normalized features
        significance = 0.4 * abs(value) / 100  # Normalize assuming max 100
        significance += 0.3 * context.get('anomaly', 0)
        significance += 0.2 * context.get('urgency', 0)
        significance += 0.1 * context.get('trend', 0)

        return min(1.0, max(0.0, significance))

    def should_activate(self, significance: float) -> bool:
        """
        Decide whether to activate based on significance and threshold
        """
        # Apply hysteresis to prevent rapid switching
        if self.was_active:
            effective_threshold = self.threshold - self.hysteresis_gap
        else:
            effective_threshold = self.threshold + self.hysteresis_gap

        return significance > effective_threshold

    def update_threshold(self):
        """
        Adjust threshold to maintain target activation rate
        """
        # Calculate current activation rate
        if len(self.activation_history) < 10:
            return  # Need some history first

        recent_rate = sum(self.activation_history[-50:]) / len(self.activation_history[-50:])

        # PI controller
        error = self.target_rate - recent_rate
        self.error_sum += error

        # Update threshold
        self.threshold += self.kp * error + self.ki * self.error_sum

        # Keep threshold in valid range
        self.threshold = min(0.95, max(0.05, self.threshold))

    def process(self, value: float, context: dict) -> GatingResult:
        """
        Process one input through the gating algorithm
        """
        # Step 1: Compute significance
        significance = self.compute_significance(value, context)

        # Step 2: Make gating decision
        activate = self.should_activate(significance)

        # Step 3: Track activation
        self.activation_history.append(activate)
        self.was_active = activate

        # Step 4: Update threshold
        self.update_threshold()

        # Calculate energy saved (1.0 if dormant, 0.0 if activated)
        energy_saved = 0.0 if activate else 1.0

        return GatingResult(
            activated=activate,
            significance=significance,
            threshold=self.threshold,
            energy_saved=energy_saved
        )

def run_demo():
    """Run a simple demonstration"""

    print("Sundew Algorithm Demo")
    print("=" * 50)
    print(f"Target: Process only 20% of inputs")
    print(f"Method: Adaptive threshold with PI control")
    print()

    # Create algorithm
    algo = SimpleSundew(target_rate=0.2)

    # Generate synthetic data stream
    n_samples = 1000
    results = []

    for i in range(n_samples):
        # Create varying input patterns
        if i % 100 < 10:
            # Occasional high-importance events
            value = random.uniform(70, 100)
            context = {
                'anomaly': random.uniform(0.7, 1.0),
                'urgency': random.uniform(0.8, 1.0),
                'trend': random.uniform(0.5, 0.8)
            }
        else:
            # Normal background data
            value = random.uniform(10, 40)
            context = {
                'anomaly': random.uniform(0.0, 0.3),
                'urgency': random.uniform(0.0, 0.2),
                'trend': random.uniform(0.2, 0.5)
            }

        # Process through algorithm
        result = algo.process(value, context)
        results.append(result)

        # Print occasional updates
        if (i + 1) % 200 == 0:
            recent_activations = sum(r.activated for r in results[-100:])
            energy_saved = sum(r.energy_saved for r in results[-100:])
            print(f"Sample {i+1:4d}: Activated {recent_activations:3d}/100, "
                  f"Energy saved: {energy_saved:.1f}%, "
                  f"Threshold: {result.threshold:.3f}")

    # Final statistics
    print()
    print("Final Results:")
    print("-" * 50)
    total_activated = sum(r.activated for r in results)
    total_energy_saved = sum(r.energy_saved for r in results)

    print(f"Total samples: {n_samples}")
    print(f"Activated: {total_activated} ({100*total_activated/n_samples:.1f}%)")
    print(f"Energy saved: {100*total_energy_saved/n_samples:.1f}%")
    print(f"Final threshold: {algo.threshold:.3f}")

    # Plot results
    plot_results(results)

def plot_results(results: List[GatingResult]):
    """Create simple visualization of results"""

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))

    # Extract data
    significances = [r.significance for r in results]
    thresholds = [r.threshold for r in results]
    activations = [1 if r.activated else 0 for r in results]

    # Plot 1: Significance vs Threshold
    axes[0].plot(significances, 'b-', alpha=0.5, label='Significance')
    axes[0].plot(thresholds, 'r-', label='Threshold')
    axes[0].scatter([i for i, r in enumerate(results) if r.activated],
                   [r.significance for r in results if r.activated],
                   c='green', s=10, alpha=0.5, label='Activated')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Significance Scoring and Adaptive Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Activation pattern
    axes[1].fill_between(range(len(activations)), 0, activations,
                        alpha=0.5, color='green')
    axes[1].set_ylabel('Activated')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title('Activation Pattern (Green = Processing, White = Dormant)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Cumulative energy savings
    cumulative_savings = []
    total = 0
    for r in results:
        total += r.energy_saved
        cumulative_savings.append(100 * total / (len(cumulative_savings) + 1))

    axes[2].plot(cumulative_savings, 'g-')
    axes[2].axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% target')
    axes[2].set_xlabel('Sample')
    axes[2].set_ylabel('Energy Saved (%)')
    axes[2].set_title('Cumulative Energy Savings')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sundew_demo_results.png', dpi=150)
    print(f"\nVisualization saved to 'sundew_demo_results.png'")
    plt.show()

if __name__ == "__main__":
    run_demo()
