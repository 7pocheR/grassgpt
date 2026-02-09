#!/bin/bash
# Chain continuation jobs for Job 609282 (36L, r=256, full block decomposition)
# This script submits continuation jobs that automatically start when previous job completes

echo "=== Setting up continuation chain for Job 609282 (36L full block) ==="
echo ""

# Target: 168k iterations at ~900 iters/12h (2× slower) → ~30 continuation jobs needed
# Submitting 30 continuations to ensure completion

echo "Submitting continuation 1 (depends on 609282)..."
JOB1=$(sbatch --dependency=afterok:609282 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh)
echo "  Job ID: $JOB1"

# Submit remaining continuations in chain
for i in {2..30}; do
    case $i in
        2) JOB2=$(sbatch --dependency=afterok:$JOB1 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 2 (depends on $JOB1)... Job ID: $JOB2" ;;
        3) JOB3=$(sbatch --dependency=afterok:$JOB2 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 3 (depends on $JOB2)... Job ID: $JOB3" ;;
        4) JOB4=$(sbatch --dependency=afterok:$JOB3 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 4 (depends on $JOB3)... Job ID: $JOB4" ;;
        5) JOB5=$(sbatch --dependency=afterok:$JOB4 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 5 (depends on $JOB4)... Job ID: $JOB5" ;;
        6) JOB6=$(sbatch --dependency=afterok:$JOB5 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 6 (depends on $JOB5)... Job ID: $JOB6" ;;
        7) JOB7=$(sbatch --dependency=afterok:$JOB6 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 7 (depends on $JOB6)... Job ID: $JOB7" ;;
        8) JOB8=$(sbatch --dependency=afterok:$JOB7 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 8 (depends on $JOB7)... Job ID: $JOB8" ;;
        9) JOB9=$(sbatch --dependency=afterok:$JOB8 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 9 (depends on $JOB8)... Job ID: $JOB9" ;;
        10) JOB10=$(sbatch --dependency=afterok:$JOB9 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 10 (depends on $JOB9)... Job ID: $JOB10" ;;
        11) JOB11=$(sbatch --dependency=afterok:$JOB10 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 11 (depends on $JOB10)... Job ID: $JOB11" ;;
        12) JOB12=$(sbatch --dependency=afterok:$JOB11 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 12 (depends on $JOB11)... Job ID: $JOB12" ;;
        13) JOB13=$(sbatch --dependency=afterok:$JOB12 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 13 (depends on $JOB12)... Job ID: $JOB13" ;;
        14) JOB14=$(sbatch --dependency=afterok:$JOB13 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 14 (depends on $JOB13)... Job ID: $JOB14" ;;
        15) JOB15=$(sbatch --dependency=afterok:$JOB14 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 15 (depends on $JOB14)... Job ID: $JOB15" ;;
        16) JOB16=$(sbatch --dependency=afterok:$JOB15 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 16 (depends on $JOB15)... Job ID: $JOB16" ;;
        17) JOB17=$(sbatch --dependency=afterok:$JOB16 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 17 (depends on $JOB16)... Job ID: $JOB17" ;;
        18) JOB18=$(sbatch --dependency=afterok:$JOB17 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 18 (depends on $JOB17)... Job ID: $JOB18" ;;
        19) JOB19=$(sbatch --dependency=afterok:$JOB18 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 19 (depends on $JOB18)... Job ID: $JOB19" ;;
        20) JOB20=$(sbatch --dependency=afterok:$JOB19 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 20 (depends on $JOB19)... Job ID: $JOB20" ;;
        21) JOB21=$(sbatch --dependency=afterok:$JOB20 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 21 (depends on $JOB20)... Job ID: $JOB21" ;;
        22) JOB22=$(sbatch --dependency=afterok:$JOB21 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 22 (depends on $JOB21)... Job ID: $JOB22" ;;
        23) JOB23=$(sbatch --dependency=afterok:$JOB22 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 23 (depends on $JOB22)... Job ID: $JOB23" ;;
        24) JOB24=$(sbatch --dependency=afterok:$JOB23 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 24 (depends on $JOB23)... Job ID: $JOB24" ;;
        25) JOB25=$(sbatch --dependency=afterok:$JOB24 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 25 (depends on $JOB24)... Job ID: $JOB25" ;;
        26) JOB26=$(sbatch --dependency=afterok:$JOB25 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 26 (depends on $JOB25)... Job ID: $JOB26" ;;
        27) JOB27=$(sbatch --dependency=afterok:$JOB26 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 27 (depends on $JOB26)... Job ID: $JOB27" ;;
        28) JOB28=$(sbatch --dependency=afterok:$JOB27 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 28 (depends on $JOB27)... Job ID: $JOB28" ;;
        29) JOB29=$(sbatch --dependency=afterok:$JOB28 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 29 (depends on $JOB28)... Job ID: $JOB29" ;;
        30) JOB30=$(sbatch --dependency=afterok:$JOB29 --parsable submit_gpt2_large_grassmann_fullblock_r256_continue.sh); echo "Submitting continuation 30 (depends on $JOB29)... Job ID: $JOB30" ;;
    esac
done

echo ""
echo "=== Continuation chain submitted ==="
echo "Training will automatically continue for up to 30 more 12-hour sessions (360 hours total)"
echo "Target: 168k iterations (100 tokens/param)"
echo "Note: Full block decomp is 2× slower, so more continuations needed"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check dependencies: scontrol show job <JOB_ID>"
