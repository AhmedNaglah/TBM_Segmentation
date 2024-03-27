#!/bin/sh 
#SBATCH --account=pinaki.sarder 
#SBATCH --job-name=tbmSegmentation
#SBATCH --output=tbmSegmentation%j.log
#SBATCH --ntasks=1
#SBATCH --mem=120gb
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=hpg-default

date;hostname;pwd

module load conda

conda activate wsi

conda run python "/orange/pinaki.sarder/ahmed.naglah/projects/TBM_Segmentation/runTBMPaired.py" \
    --svsBase "/blue/pinaki.sarder/nlucarelli/kpmp_new/" \
    --fid "66017fc2d2f45e99a91dbc17" \
    --outputdir "/orange/pinaki.sarder/ahmed.naglah/data/tbmSegPaired" \
    --username ${DSA_USERNAME} \
    --password ${DSA_PW} \
    --apiUrl ${DSA_URL} \
    --patchSize 512 \
    --layerName "tubules" \
    --name "tbmSegPaired"