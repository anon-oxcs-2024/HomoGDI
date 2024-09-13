py="/slurm-storage/linbao/thesis_code_1/GraphGPS/configs/GPS/debug/wandb_test.py"

for i in {1..10}; do
    cat <<EOT > temp_submit.job
#!/bin/bash
#SBATCH -N 1      # nodes requested
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --output=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.out   # standard output
#SBATCH --error=/slurm-storage/linbao/thesis_code_1/outputs/%x_%j.err    # standard error

srun -u /slurm-storage/linbao/.conda/envs/GPS-envV2/bin/python "$py"
EOT
    sleep 1s 
    sbatch temp_submit.job
    #cat temp_submit.job
done