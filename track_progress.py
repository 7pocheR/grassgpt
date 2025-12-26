#!/usr/bin/env python3
"""
Track training progress across all phases from log files.

Handles:
- Different length sequences (unfinished jobs)
- Missing/incomplete log files
- Multiple runs of same phase (uses latest job ID)
- Phase variants (0.5, 1.2, etc.)
- Empty or corrupted logs
"""

import re
import os
import glob
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Configuration
LOGS_DIR = Path("logs")
OUTPUT_DIR = Path(".")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# Regex patterns for parsing
ITER_PATTERN = re.compile(r"iter (\d+): loss ([\d.]+), time")
STEP_PATTERN = re.compile(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)")

# Phase ordering for logical sorting
PHASE_ORDER = {
    'baseline': 0,
    'phase0.5': 0.5,
    'phase1': 1,
    'phase1.2': 1.2,
    'phase1.5': 1.5,
    'phase2': 2,
    'phase2.2': 2.2,
    'phase2.5': 2.5,
    'phase3': 3,
    'phase3.2': 3.2,
    'phase4': 4,
    'phase5': 5,
}

def parse_phase_from_filename(filename):
    """Extract phase name from log filename."""
    # Pattern: <jobid>_<phase>.out
    match = re.search(r'\d+_(.+)\.out$', filename)
    if match:
        return match.group(1)
    return None

def get_latest_log_for_phase(phase):
    """Get the latest log file for a given phase (highest job ID)."""
    pattern = f"*_{phase}.out"
    matching_files = list(LOGS_DIR.glob(pattern))

    if not matching_files:
        return None

    # Extract job IDs and find max
    job_ids = []
    for f in matching_files:
        match = re.match(r'(\d+)_', f.name)
        if match:
            job_ids.append((int(match.group(1)), f))

    if not job_ids:
        return None

    # Return file with highest job ID
    return max(job_ids, key=lambda x: x[0])[1]

def parse_log_file(filepath):
    """
    Parse a log file and extract training metrics.

    Returns:
        dict with keys:
            - 'iters': list of iteration numbers
            - 'train_losses': list of training losses
            - 'steps': list of checkpoint step numbers
            - 'val_losses': list of validation losses
            - 'step_train_losses': list of training losses at checkpoints
            - 'job_id': job ID
            - 'phase': phase name
    """
    if not filepath.exists():
        return None

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return None

    if not content.strip():
        return None

    # Parse iteration-level losses
    iters = []
    train_losses = []
    for match in ITER_PATTERN.finditer(content):
        iters.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))

    # Parse checkpoint-level metrics
    steps = []
    step_train_losses = []
    val_losses = []
    for match in STEP_PATTERN.finditer(content):
        steps.append(int(match.group(1)))
        step_train_losses.append(float(match.group(2)))
        val_losses.append(float(match.group(3)))

    # Extract job ID and phase
    job_id_match = re.match(r'(\d+)_', filepath.name)
    job_id = int(job_id_match.group(1)) if job_id_match else None

    phase = parse_phase_from_filename(filepath.name)

    return {
        'iters': iters,
        'train_losses': train_losses,
        'steps': steps,
        'step_train_losses': step_train_losses,
        'val_losses': val_losses,
        'job_id': job_id,
        'phase': phase,
        'filepath': filepath,
    }

def collect_all_phases():
    """Collect data for all phases, using latest log for each."""
    phases_data = {}

    # Get all unique phases
    all_logs = list(LOGS_DIR.glob("*.out"))
    unique_phases = set()
    for log in all_logs:
        phase = parse_phase_from_filename(log.name)
        if phase:
            unique_phases.add(phase)

    # Parse latest log for each phase
    for phase in unique_phases:
        latest_log = get_latest_log_for_phase(phase)
        if latest_log:
            data = parse_log_file(latest_log)
            if data and (data['iters'] or data['steps']):  # Has some data
                phases_data[phase] = data

    return phases_data

def get_grassmann_ratio(phase):
    """Get Grassmann parameter ratio for each phase."""
    # Based on the parameter counts we saw earlier
    ratios = {
        'baseline': '0%',
        'phase0.5': '25%',  # c_attn only
        'phase1': '28%',    # c_attn + attn.c_proj
        'phase1.2': '28%',  # c_attn + attn.c_proj
        'phase1.5': '30%',  # mlp.c_fc only
        'phase2': '34%',    # mlp.c_fc + attn.c_proj
        'phase2.2': '34%',  # mlp.c_fc + attn.c_proj
        'phase2.5': '33%',  # c_attn + mlp.c_fc
        'phase3': '62%',    # c_attn + mlp.c_fc + attn.c_proj
        'phase3.2': '62%',  # c_attn + mlp.c_fc + attn.c_proj
        'phase4': '90%',    # Most layers
        'phase5': '100%',   # All layers
    }
    return ratios.get(phase, 'N/A')

def create_progress_table(phases_data):
    """Create a markdown table with current progress."""
    lines = []
    lines.append("# Training Progress Summary")
    lines.append("")
    lines.append(f"**Total phases tracked:** {len(phases_data)}")
    lines.append("")
    lines.append("| Phase | Grass% | Best Val | Best Train | Gen Gap | Step @ Best | Current Iter | Status |")
    lines.append("|-------|--------|----------|------------|---------|-------------|--------------|--------|")

    # Sort by best validation loss (ascending)
    sorted_phases = sorted(
        phases_data.keys(),
        key=lambda x: (
            min(phases_data[x]['val_losses']) if phases_data[x]['val_losses'] else 999,
            PHASE_ORDER.get(x, 999)
        )
    )

    for phase in sorted_phases:
        data = phases_data[phase]

        current_iter = data['iters'][-1] if data['iters'] else 0
        grass_ratio = get_grassmann_ratio(phase)

        # Find best (minimum) validation loss
        if data['val_losses'] and data['step_train_losses']:
            best_val_idx = np.argmin(data['val_losses'])
            best_val = data['val_losses'][best_val_idx]
            best_train = data['step_train_losses'][best_val_idx]
            best_step = data['steps'][best_val_idx]
            gen_gap = best_val - best_train

            best_val_str = f"{best_val:.4f}"
            best_train_str = f"{best_train:.4f}"
            gen_gap_str = f"{gen_gap:.4f}"
            best_step_str = str(best_step)
        else:
            best_val_str = 'N/A'
            best_train_str = 'N/A'
            gen_gap_str = 'N/A'
            best_step_str = 'N/A'

        # Determine status
        total_steps = len(data['val_losses'])
        if current_iter >= 15000:
            status = '✓ Complete'
        elif current_iter > 0:
            progress_pct = int(100 * current_iter / 15000)
            status = f'Running {progress_pct}%'
        else:
            status = 'No data'

        lines.append(
            f"| {phase} | {grass_ratio} | {best_val_str} | {best_train_str} | "
            f"{gen_gap_str} | {best_step_str} | {current_iter} | {status} |"
        )

    lines.append("")
    lines.append("## Phase Descriptions")
    lines.append("")
    lines.append("- **baseline**: AdamW only (no Grassmann)")
    lines.append("- **phase0.5**: c_attn (Grassmann), attn.c_proj (AdamW)")
    lines.append("- **phase1**: c_attn + attn.c_proj (both Grassmann G(0,-1,r) or G(1,0,r))")
    lines.append("- **phase1.2**: c_attn + attn.c_proj (G(1,0,r)) + **no skip**")
    lines.append("- **phase1.5**: mlp.c_fc (Grassmann), attn.c_proj (AdamW)")
    lines.append("- **phase2**: mlp.c_fc + attn.c_proj (both Grassmann)")
    lines.append("- **phase2.2**: mlp.c_fc + attn.c_proj (G(1,0,r)) + **no skip**")
    lines.append("- **phase2.5**: c_attn + mlp.c_fc (Grassmann), attn.c_proj (AdamW)")
    lines.append("- **phase3**: c_attn + mlp.c_fc + attn.c_proj (all Grassmann)")
    lines.append("- **phase3.2**: c_attn + mlp.c_fc + attn.c_proj (G(1,0,r)) + **no skip**")
    lines.append("- **phase4**: Full MLP (c_fc + c_proj) + attn layers")
    lines.append("- **phase5**: All layers G(1,0,r), no skip connections")
    lines.append("")

    return "\n".join(lines)

def plot_validation_loss(phases_data):
    """Plot validation loss vs steps for all phases, marking best point for each."""
    plt.figure(figsize=(14, 7))

    # Sort phases for consistent colors
    sorted_phases = sorted(
        phases_data.keys(),
        key=lambda x: PHASE_ORDER.get(x, 999)
    )

    for phase in sorted_phases:
        data = phases_data[phase]
        if data['steps'] and data['val_losses']:
            # Plot all points
            line = plt.plot(
                data['steps'],
                data['val_losses'],
                marker='o',
                markersize=3,
                label=phase,
                alpha=0.7,
                linewidth=1.5
            )[0]

            # Mark best point with star in same color
            best_idx = np.argmin(data['val_losses'])
            best_step = data['steps'][best_idx]
            best_val = data['val_losses'][best_idx]
            plt.plot(
                best_step,
                best_val,
                marker='*',
                markersize=15,
                color=line.get_color(),
                markeredgecolor='black',
                markeredgewidth=0.5,
                zorder=10
            )

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('Validation Loss Across Phases (★ = Best)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'validation_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / 'validation_loss_comparison.png'}")

def plot_train_loss(phases_data):
    """Plot training loss vs iterations for all phases."""
    plt.figure(figsize=(12, 6))

    sorted_phases = sorted(
        phases_data.keys(),
        key=lambda x: PHASE_ORDER.get(x, 999)
    )

    for phase in sorted_phases:
        data = phases_data[phase]
        if data['iters'] and data['train_losses']:
            # Subsample for readability (plot every 10th point if >1000 points)
            iters = data['iters']
            losses = data['train_losses']

            if len(iters) > 1000:
                iters = iters[::10]
                losses = losses[::10]

            plt.plot(
                iters,
                losses,
                label=phase,
                alpha=0.6,
                linewidth=1
            )

    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Across Phases')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'training_loss_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / 'training_loss_comparison.png'}")

def plot_generalization_gap(phases_data):
    """Plot generalization gap (val - train) at best val point for each phase."""
    plt.figure(figsize=(12, 6))

    phases = []
    gaps = []
    colors = []

    # Sort by best val loss
    sorted_phases = sorted(
        phases_data.keys(),
        key=lambda x: (
            min(phases_data[x]['val_losses']) if phases_data[x]['val_losses'] else 999,
            PHASE_ORDER.get(x, 999)
        )
    )

    for phase in sorted_phases:
        data = phases_data[phase]
        if data['val_losses'] and data['step_train_losses']:
            best_val_idx = np.argmin(data['val_losses'])
            best_val = data['val_losses'][best_val_idx]
            best_train = data['step_train_losses'][best_val_idx]
            gen_gap = best_val - best_train

            phases.append(phase)
            gaps.append(gen_gap)

            # Color by Grassmann ratio
            grass_pct = get_grassmann_ratio(phase)
            if grass_pct == '0%':
                colors.append('gray')
            elif '.2' in phase:
                colors.append('red')  # No-skip variants
            elif '.5' in phase:
                colors.append('green')  # Skip attn.c_proj variants
            else:
                colors.append('blue')  # Standard phases

    # Create bar plot
    bars = plt.bar(range(len(phases)), gaps, color=colors, alpha=0.7, edgecolor='black')

    plt.xticks(range(len(phases)), phases, rotation=45, ha='right')
    plt.ylabel('Generalization Gap (Val - Train)', fontsize=12)
    plt.title('Generalization Gap at Best Val Loss Point', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Baseline (AdamW)'),
        Patch(facecolor='green', alpha=0.7, label='.5 variants (skip attn.c_proj)'),
        Patch(facecolor='blue', alpha=0.7, label='Standard Grassmann'),
        Patch(facecolor='red', alpha=0.7, label='.2 variants (no skip)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'generalization_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOT_DIR / 'generalization_gap.png'}")

def main():
    print("Collecting training data from logs...")
    phases_data = collect_all_phases()

    if not phases_data:
        print("No valid log data found!")
        return

    print(f"Found data for {len(phases_data)} phases")

    # Create markdown table
    print("\nGenerating progress table...")
    table_md = create_progress_table(phases_data)
    output_path = OUTPUT_DIR / "TRAINING_PROGRESS.md"
    with open(output_path, 'w') as f:
        f.write(table_md)
    print(f"Saved: {output_path}")

    # Create plots
    print("\nGenerating plots...")
    plot_validation_loss(phases_data)
    plot_train_loss(phases_data)
    plot_generalization_gap(phases_data)

    print("\n✓ Progress tracking complete!")
    print(f"  - Summary: {output_path}")
    print(f"  - Plots: {PLOT_DIR}/")

if __name__ == "__main__":
    main()
