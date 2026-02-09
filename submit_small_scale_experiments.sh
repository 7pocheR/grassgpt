#!/bin/bash
# Submit all small-scale experiments in sequence
# Each job is independent, can run in parallel

# Baseline
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_base
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_small_baseline.out
#SBATCH --error=logs/%j_small_baseline.err

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_small_baseline.py
EOF

# Grassmann r=128
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_r128
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_small_mlp_r128.out
#SBATCH --error=logs/%j_small_mlp_r128.err

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_small_mlp_grass_r128.py
EOF

# Grassmann r=192
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_r192
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_small_mlp_r192.out
#SBATCH --error=logs/%j_small_mlp_r192.err

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_small_mlp_grass_r192.py
EOF

# Grassmann r=256
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=sm_r256
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h100:4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_small_mlp_r256.out
#SBATCH --error=logs/%j_small_mlp_r256.err

source /home/junyuren/.conda/envs/manifold_muon/bin/activate
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2_small_mlp_grass_r256.py
EOF

echo "Submitted 4 small-scale experiments"
