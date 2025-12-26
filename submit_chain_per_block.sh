#!/bin/bash
# Chain per_block training across multiple 12h sessions

# Current job: 568938 (running)

# Submit cycle 2
JOB2=$(sbatch --dependency=afterany:568938 --parsable submit_openwebtext_per_block_resume.sh)
echo "Submitted per_block cycle 2: $JOB2 (after 568938)"

# Submit cycle 3
JOB3=$(sbatch --dependency=afterany:$JOB2 --parsable submit_openwebtext_per_block_resume.sh)
echo "Submitted per_block cycle 3: $JOB3 (after $JOB2)"

# Submit cycle 4
JOB4=$(sbatch --dependency=afterany:$JOB3 --parsable submit_openwebtext_per_block_resume.sh)
echo "Submitted per_block cycle 4: $JOB4 (after $JOB3)"

# Submit cycle 5
JOB5=$(sbatch --dependency=afterany:$JOB4 --parsable submit_openwebtext_per_block_resume.sh)
echo "Submitted per_block cycle 5: $JOB5 (after $JOB4)"

# Submit cycle 6
JOB6=$(sbatch --dependency=afterany:$JOB5 --parsable submit_openwebtext_per_block_resume.sh)
echo "Submitted per_block cycle 6: $JOB6 (after $JOB5)"

echo ""
echo "per_block training chain:"
echo "  568938 (running) -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5 -> $JOB6"
