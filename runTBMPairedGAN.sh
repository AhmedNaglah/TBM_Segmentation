#!/bin/sh 
#SBATCH --account=pinaki.sarder 
#SBATCH --job-name=GANSeg
#SBATCH --output=GANSeg%j.log
#SBATCH --ntasks=1
#SBATCH --mem=120gb
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-default

date;hostname;pwd

module load conda

conda activate wsi

conda run python ./runTBMPairedGAN.py \
    --svsBase "/blue/pinaki.sarder/nlucarelli/kpmp_new/" \
    --fid "65fc4d4ed2f45e99a916b24c" \
    --outputdir "/orange/pinaki.sarder/ahmed.naglah/data/GANGAN3" \
    --username ${DSA_USERNAME} \
    --password ${DSA_PW} \
    --apiUrl ${DSA_URL} \
    --patchSize 512 \
    --layerName "tubules" \
    --name "GANSeg2" \
    --checkpoint_path "/orange/pinaki.sarder/ahmed.naglah/data/kpmpCycleGAN/output_kpmpCycleGAN3/training_checkpoints/ckpt-1"