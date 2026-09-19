#!/bin/bash -l
#SBATCH --job-name=stylehgan
#SBATCH --ntasks=5
#SBATCH --error=error_%j.txt
#SBATCH --partition=lsc
#SBATCH --output=gen_%j.log
#SBATCH --gres=gpu:2
CUDA_VISIBLE_DEVICES=4,5
echo "Nodo: $(hostname)"
echo "GPUs visibles: $CUDA_VISIBLE_DEVICES"
echo "Ejecutando en el nodo: $(hostname)"
conda activate /usuarios03/hector.anaya1/Experiments/stylegan-v/env
python src/infra/launch.py hydra.run.dir=. exp_suffix=ltc_experiment_sky_2 env=local dataset=sky_timelapse  dataset.resolution=256 num_gpus=2 model.generator.motion.gen_strategy=ltc model.generator.time_enc.min_period_len=256

