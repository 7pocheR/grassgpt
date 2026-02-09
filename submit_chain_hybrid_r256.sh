#!/bin/bash
# Chain continuation jobs for Job 608061 (36L, r=256, hybrid optimal)
# This script submits continuation jobs that automatically start when previous job completes

echo "=== Setting up continuation chain for Job 608061 (36L hybrid) ==="
echo ""

# Target: 168k iterations at ~1800 iters/12h → ~14 continuation jobs needed
# Submitting 15 continuations to ensure completion

# Submit first continuation (depends on original job 608061)
echo "Submitting continuation 1 (depends on 608061)..."
JOB1=$(sbatch --dependency=afterok:608061 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh)
echo "  Job ID: $JOB1"

# Submit remaining continuations in chain
for i in {2..15}; do
    PREV_JOB=$JOB1
    if [ $i -eq 2 ]; then PREV_JOB=$JOB1; fi
    if [ $i -eq 3 ]; then PREV_JOB=$JOB2; fi
    if [ $i -eq 4 ]; then PREV_JOB=$JOB3; fi
    if [ $i -eq 5 ]; then PREV_JOB=$JOB4; fi
    if [ $i -eq 6 ]; then PREV_JOB=$JOB5; fi
    if [ $i -eq 7 ]; then PREV_JOB=$JOB6; fi
    if [ $i -eq 8 ]; then PREV_JOB=$JOB7; fi
    if [ $i -eq 9 ]; then PREV_JOB=$JOB8; fi
    if [ $i -eq 10 ]; then PREV_JOB=$JOB9; fi
    if [ $i -eq 11 ]; then PREV_JOB=$JOB10; fi
    if [ $i -eq 12 ]; then PREV_JOB=$JOB11; fi
    if [ $i -eq 13 ]; then PREV_JOB=$JOB12; fi
    if [ $i -eq 14 ]; then PREV_JOB=$JOB13; fi
    if [ $i -eq 15 ]; then PREV_JOB=$JOB14; fi

    echo "Submitting continuation $i (depends on $PREV_JOB)..."
    case $i in
        2) JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB2" ;;
        3) JOB3=$(sbatch --dependency=afterok:$JOB2 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB3" ;;
        4) JOB4=$(sbatch --dependency=afterok:$JOB3 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB4" ;;
        5) JOB5=$(sbatch --dependency=afterok:$JOB4 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB5" ;;
        6) JOB6=$(sbatch --dependency=afterok:$JOB5 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB6" ;;
        7) JOB7=$(sbatch --dependency=afterok:$JOB6 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB7" ;;
        8) JOB8=$(sbatch --dependency=afterok:$JOB7 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB8" ;;
        9) JOB9=$(sbatch --dependency=afterok:$JOB8 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB9" ;;
        10) JOB10=$(sbatch --dependency=afterok:$JOB9 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB10" ;;
        11) JOB11=$(sbatch --dependency=afterok:$JOB10 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB11" ;;
        12) JOB12=$(sbatch --dependency=afterok:$JOB11 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB12" ;;
        13) JOB13=$(sbatch --dependency=afterok:$JOB12 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB13" ;;
        14) JOB14=$(sbatch --dependency=afterok:$JOB13 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB14" ;;
        15) JOB15=$(sbatch --dependency=afterok:$JOB14 --parsable submit_gpt2_large_grassmann_hybridgate_r256_continue.sh); echo "  Job ID: $JOB15" ;;
    esac
done

echo ""
echo "=== Continuation chain submitted ==="
echo "Training will automatically continue for up to 15 more 12-hour sessions (180 hours total)"
echo "Target: 168k iterations (100 tokens/param)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check dependencies: scontrol show job <JOB_ID>"
