#!/bin/bash
# Chain rank99 training across multiple 12h sessions

# Current job: 568937 (running)

# Submit cycle 2
JOB2=$(sbatch --dependency=afterany:568937 --parsable submit_openwebtext_rank99_resume.sh)
echo "Submitted rank99 cycle 2: $JOB2 (after 568937)"

# Submit cycle 3
JOB3=$(sbatch --dependency=afterany:$JOB2 --parsable submit_openwebtext_rank99_resume.sh)
echo "Submitted rank99 cycle 3: $JOB3 (after $JOB2)"

# Submit cycle 4
JOB4=$(sbatch --dependency=afterany:$JOB3 --parsable submit_openwebtext_rank99_resume.sh)
echo "Submitted rank99 cycle 4: $JOB4 (after $JOB3)"

# Submit cycle 5
JOB5=$(sbatch --dependency=afterany:$JOB4 --parsable submit_openwebtext_rank99_resume.sh)
echo "Submitted rank99 cycle 5: $JOB5 (after $JOB4)"

# Submit cycle 6
JOB6=$(sbatch --dependency=afterany:$JOB5 --parsable submit_openwebtext_rank99_resume.sh)
echo "Submitted rank99 cycle 6: $JOB6 (after $JOB5)"

echo ""
echo "rank99 training chain:"
echo "  568937 (running) -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5 -> $JOB6"
