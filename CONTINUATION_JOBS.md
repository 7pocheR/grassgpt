# Continuation Jobs - Automatic Training Resumption

This directory contains scripts for automatically chaining training jobs to handle SLURM's 12-hour time limit.

## Quick Start

To set up automatic continuation for the three parallel experiments:

```bash
# Job 608059 (24L, r=384, hybrid baseline)
./submit_chain_baseline.sh

# Job 608061 (36L, r=256, hybrid optimal)
./submit_chain_hybrid_r256.sh

# Job 609282 (36L, r=256, full block decomposition)
./submit_chain_fullblock_r256.sh
```

## How It Works

### 1. Continuation Scripts

Each job has a corresponding continuation script that:
- Uses `--init_from=resume` to load from checkpoint
- Maintains all hyperparameters from original config
- Automatically starts when previous job completes

**Available continuation scripts:**
- `submit_gpt2m_grassmann_hybridgate_r384_continue.sh` → Job 608059
- `submit_gpt2_large_grassmann_hybridgate_r256_continue.sh` → Job 608061
- `submit_gpt2_large_grassmann_fullblock_r256_continue.sh` → Job 609282

### 2. Dependency Chains

Chain scripts use SLURM's `--dependency=afterok:JOBID` to create automatic sequences:

```bash
# Example: Job 608059 → JOB1 → JOB2 → JOB3 → ...
sbatch --dependency=afterok:608059 submit_gpt2m_grassmann_hybridgate_r384_continue.sh
```

**Chain structure:**
- **Job 608059 (24L hybrid):** 8 continuations → 108 hours total
- **Job 608061 (36L hybrid):** 15 continuations → 180 hours total
- **Job 609282 (36L full block):** 30 continuations → 360 hours total

### 3. Checkpoint Mechanism

Training automatically saves checkpoints to:
- Job 608059: `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2m-grass-hybridgate-r384/ckpt.pt`
- Job 608061: `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-hybridgate-r256/ckpt.pt`
- Job 609282: `/net/scratch2/junyuren/nanoGPT-manifold/out-gpt2large-grass-fullblock-r256/ckpt.pt`

Checkpoints contain:
- Model weights (Grassmann + AdamW parameters)
- Optimizer state (momentum, variance)
- Training state (iter_num, best_val_loss)
- Config (model_args)

## Monitoring

```bash
# Check all jobs (including pending dependencies)
squeue -u $USER

# Check specific job dependencies
scontrol show job <JOB_ID>

# Monitor training progress
tail -f logs/<JOB_ID>_*.out

# Use dashboard for comparison
watch -n 60 ./monitor_jobs.sh
```

## Training Targets

| Job | Model | Rank | Target Iters | Est. Time | Iters/12h | Continuations |
|-----|-------|------|--------------|-----------|-----------|---------------|
| 608059 | 24L hybrid | r=384 | 100,000 | ~60h | ~11,000 | 8 |
| 608061 | 36L hybrid | r=256 | 168,000 | ~126h | ~1,800 | 15 |
| 609282 | 36L full block | r=256 | 168,000 | ~252h | ~900 | 30 |

**Note:** Full block decomposition is 2× slower due to memory constraints (batch_size=16 vs 32).

## Manual Submission

If you prefer to submit continuations manually:

```bash
# Wait for original job to finish, then:
sbatch submit_gpt2m_grassmann_hybridgate_r384_continue.sh

# Or submit with dependency immediately:
sbatch --dependency=afterok:608059 submit_gpt2m_grassmann_hybridgate_r384_continue.sh
```

## Troubleshooting

### Job fails to resume
- Check checkpoint exists: `ls -lh /net/scratch2/junyuren/nanoGPT-manifold/out-*/ckpt.pt`
- Check error logs: `tail -f logs/<JOB_ID>_*.err`
- Verify config matches: continuation uses same config file as original

### Dependency not triggering
- Check original job status: `squeue -u $USER`
- Check dependency status: `scontrol show job <DEPENDENT_JOB_ID>`
- Dependencies only trigger on `afterok` (successful completion)

### Training stops before max_iters
- Check `max_iters` in config matches target
- Monitor iter_num in checkpoint: `python -c "import torch; print(torch.load('path/to/ckpt.pt')['iter_num'])"`

## Safety Notes

- **Checkpoint directory:** Always use `/net/scratch2/` (home has strict quota)
- **SLURM limits:** General partition has 12h max, plan accordingly
- **Dependencies:** Use `afterok` (not `afterany`) to avoid chaining failed jobs
- **Monitoring:** Check logs regularly to catch issues early

## Example: Submitting All Chains

```bash
# Submit all three continuation chains at once
./submit_chain_baseline.sh &
./submit_chain_hybrid_r256.sh &
./submit_chain_fullblock_r256.sh &
wait

# Monitor
squeue -u $USER
```

This will set up 8 + 15 + 30 = 53 pending jobs that will automatically execute as their dependencies complete.
