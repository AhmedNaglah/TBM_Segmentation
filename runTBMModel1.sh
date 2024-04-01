#!/bin/sh 
#SBATCH --account=pinaki.sarder 
#SBATCH --job-name=tbmSegmentation
#SBATCH --output=tbmSegmentation%j.log
#SBATCH --ntasks=1
#SBATCH --mem=40gb
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-default

date;hostname;pwd

module load conda

conda activate wsi

conda run python "/orange/pinaki.sarder/ahmed.naglah/projects/TBM_Segmentation/runTBMModel1.py" \
    --svsBase "/blue/pinaki.sarder/nlucarelli/kpmp_new/" \
    --fid "65fc511dd2f45e99a916b258" \
    --outputdir "/orange/pinaki.sarder/ahmed.naglah/data/tbmSegPaired" \
    --username ${DSA_USERNAME} \
    --password ${DSA_PW} \
    --apiUrl ${DSA_URL} \
    --patchSize 512 \
    --layerName "tubules" \
    --name "tbmSegPaired"