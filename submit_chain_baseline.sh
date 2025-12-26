#!/bin/bash
# Chain baseline training across multiple 12h sessions

# Current job: 568926 (running)

# Submit cycle 2 (depends on current job)
JOB2=$(sbatch --dependency=afterany:568926 --parsable submit_openwebtext_baseline_resume.sh)
echo "Submitted baseline cycle 2: $JOB2 (after 568926)"

# Submit cycle 3 (depends on cycle 2)
JOB3=$(sbatch --dependency=afterany:$JOB2 --parsable submit_openwebtext_baseline_resume.sh)
echo "Submitted baseline cycle 3: $JOB3 (after $JOB2)"

# Submit cycle 4 (depends on cycle 3)
JOB4=$(sbatch --dependency=afterany:$JOB3 --parsable submit_openwebtext_baseline_resume.sh)
echo "Submitted baseline cycle 4: $JOB4 (after $JOB3)"

# Submit cycle 5 (depends on cycle 4)
JOB5=$(sbatch --dependency=afterany:$JOB4 --parsable submit_openwebtext_baseline_resume.sh)
echo "Submitted baseline cycle 5: $JOB5 (after $JOB4)"

echo ""
echo "Baseline training chain:"
echo "  568926 (running) -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5"
echo "  Estimated coverage: ~60-70k iters (60-70% of 100k goal)"
