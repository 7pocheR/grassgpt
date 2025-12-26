#!/bin/bash
# Chain stable_rank_x2 training across multiple 12h sessions

# Current job: 568936 (running)

# Submit cycle 2 (depends on current job)
JOB2=$(sbatch --dependency=afterany:568936 --parsable submit_openwebtext_grassmann_resume.sh)
echo "Submitted stable_rank_x2 cycle 2: $JOB2 (after 568936)"

# Submit cycle 3
JOB3=$(sbatch --dependency=afterany:$JOB2 --parsable submit_openwebtext_grassmann_resume.sh)
echo "Submitted stable_rank_x2 cycle 3: $JOB3 (after $JOB2)"

# Submit cycle 4
JOB4=$(sbatch --dependency=afterany:$JOB3 --parsable submit_openwebtext_grassmann_resume.sh)
echo "Submitted stable_rank_x2 cycle 4: $JOB4 (after $JOB3)"

# Submit cycle 5
JOB5=$(sbatch --dependency=afterany:$JOB4 --parsable submit_openwebtext_grassmann_resume.sh)
echo "Submitted stable_rank_x2 cycle 5: $JOB5 (after $JOB4)"

# Submit cycle 6
JOB6=$(sbatch --dependency=afterany:$JOB5 --parsable submit_openwebtext_grassmann_resume.sh)
echo "Submitted stable_rank_x2 cycle 6: $JOB6 (after $JOB5)"

echo ""
echo "stable_rank_x2 training chain:"
echo "  568936 (running) -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5 -> $JOB6"
echo "  Estimated coverage: ~60-70k iters (slower due to Grassmann)"
