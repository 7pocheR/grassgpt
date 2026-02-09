#!/bin/bash
# Chain continuation jobs for Job 608059 (24L, r=384, hybrid baseline)
# This script submits continuation jobs that automatically start when previous job completes

echo "=== Setting up continuation chain for Job 608059 (24L hybrid) ==="
echo ""

# Submit first continuation (depends on original job 608059)
echo "Submitting continuation 1 (depends on 608059)..."
JOB1=$(sbatch --dependency=afterok:608059 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB1"

# Submit second continuation (depends on first continuation)
echo "Submitting continuation 2 (depends on $JOB1)..."
JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB2"

# Submit third continuation (depends on second continuation)
echo "Submitting continuation 3 (depends on $JOB2)..."
JOB3=$(sbatch --dependency=afterok:$JOB2 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB3"

# Submit fourth continuation (depends on third continuation)
echo "Submitting continuation 4 (depends on $JOB3)..."
JOB4=$(sbatch --dependency=afterok:$JOB3 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB4"

# Submit fifth continuation (depends on fourth continuation)
echo "Submitting continuation 5 (depends on $JOB4)..."
JOB5=$(sbatch --dependency=afterok:$JOB4 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB5"

# Submit sixth continuation (depends on fifth continuation)
echo "Submitting continuation 6 (depends on $JOB5)..."
JOB6=$(sbatch --dependency=afterok:$JOB5 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB6"

# Submit seventh continuation (depends on sixth continuation)
echo "Submitting continuation 7 (depends on $JOB6)..."
JOB7=$(sbatch --dependency=afterok:$JOB6 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB7"

# Submit eighth continuation (depends on seventh continuation)
echo "Submitting continuation 8 (depends on $JOB7)..."
JOB8=$(sbatch --dependency=afterok:$JOB7 --parsable submit_gpt2m_grassmann_hybridgate_r384_continue.sh)
echo "  Job ID: $JOB8"

echo ""
echo "=== Continuation chain submitted ==="
echo "Training will automatically continue for up to 9 more 12-hour sessions (108 hours total)"
echo "Original job 608059 → $JOB1 → $JOB2 → $JOB3 → $JOB4 → $JOB5 → $JOB6 → $JOB7 → $JOB8"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check dependencies: scontrol show job <JOB_ID>"
