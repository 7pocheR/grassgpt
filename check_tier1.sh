#!/bin/bash
# Quick check of Tier 1 experiment status

echo "=== TIER 1 EXPERIMENT STATUS ==="
echo ""

# Check if jobs are running
echo "Job Status:"
squeue -u $USER -j 610968,610969,610970 -o "%.10i %.30j %.8T %.10M %.8D %R" 2>/dev/null | head -10
echo ""

# Run comparison script
python track_tier1_progress.py
