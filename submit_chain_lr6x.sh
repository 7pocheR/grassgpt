#!/bin/bash
# Chain lr6x training across multiple 12h sessions
# Current job: 568931 (running, starting from scratch)

# Submit cycle 2 (resume from cycle 1 checkpoint)
JOB2=$(sbatch --dependency=afterany:568931 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 2: $JOB2 (after 568931)"

# Submit cycles 3-8 (all resume)
JOB3=$(sbatch --dependency=afterany:$JOB2 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 3: $JOB3 (after $JOB2)"

JOB4=$(sbatch --dependency=afterany:$JOB3 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 4: $JOB4 (after $JOB3)"

JOB5=$(sbatch --dependency=afterany:$JOB4 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 5: $JOB5 (after $JOB4)"

JOB6=$(sbatch --dependency=afterany:$JOB5 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 6: $JOB6 (after $JOB5)"

JOB7=$(sbatch --dependency=afterany:$JOB6 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 7: $JOB7 (after $JOB6)"

JOB8=$(sbatch --dependency=afterany:$JOB7 --parsable submit_openwebtext_lr6x_resume.sh)
echo "Submitted lr6x cycle 8: $JOB8 (after $JOB7)"

echo ""
echo "lr6x training chain:"
echo "  568931 (running, scratch) -> $JOB2 (resume) -> ... -> $JOB8 (resume)"
echo "  Total: 8 cycles × 12h = 96h of training"
