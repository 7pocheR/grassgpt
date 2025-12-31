#!/bin/bash
# Quick monitoring script for parallel Grassmann experiments

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Grassmann Manifold Training - Job Monitoring Dashboard       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Job status
echo "=== Job Queue Status ==="
squeue -u $USER -o "%.10i %.20j %.2t %.10M %.6D %R"
echo ""

# Function to display job stats
show_job_stats() {
    local job_id=$1
    local job_name=$2
    local log_file="logs/${job_id}_*.out"

    echo "┌─ Job $job_id: $job_name"

    # Parameter count
    local params=$(grep "number of parameters" $log_file 2>/dev/null | head -1)
    if [ -n "$params" ]; then
        echo "│  $params"
    fi

    # Latest training steps
    local latest=$(grep "^step [0-9]*:" $log_file 2>/dev/null | tail -3)
    if [ -n "$latest" ]; then
        echo "$latest" | while read line; do
            echo "│  $line"
        done
    else
        echo "│  No training steps yet"
    fi

    # Generalization gap at latest step
    local last_step=$(grep "^step [0-9]*:" $log_file 2>/dev/null | tail -1)
    if [ -n "$last_step" ]; then
        local train_loss=$(echo "$last_step" | grep -oP "train loss \K[0-9.]+")
        local val_loss=$(echo "$last_step" | grep -oP "val loss \K[0-9.]+")
        local gap=$(awk "BEGIN {printf \"%.4f\", $val_loss - $train_loss}")
        echo "│  Generalization gap: $gap (val - train)"
    fi

    echo "└─"
    echo ""
}

# Show stats for each job
echo "=== Training Progress ==="
show_job_stats "608059" "24L, r=384, hybrid (baseline)"
show_job_stats "608061" "36L, r=256, hybrid (optimal)"
show_job_stats "609282" "36L, r=256, full block"

# Comparison with previous run
echo "=== Comparison with Job 605987 (elementwise gating) ==="
echo "At iter 1000:"
echo "  605987 (elementwise): train=4.83, val=4.84, gap=0.008"
echo "  608059 (hybrid):      train=4.58, val=4.60, gap=0.014 → 5% better loss, worse gap"
echo ""
echo "At iter 3000:"
echo "  605987 (elementwise): train=3.53, val=3.53, gap=0.0003"
echo "  608059 (hybrid):      train=3.58, val=3.60, gap=0.0215 → 1.5% worse loss, 7× worse gap"
echo ""

echo "Use: watch -n 60 ./monitor_jobs.sh  # Auto-refresh every 60 seconds"
