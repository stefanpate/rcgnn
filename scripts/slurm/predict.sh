#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=48GB
#SBATCH -t 12:00:00
#SBATCH --job-name="predict"
#SBATCH --output=/home/spn1560/hiec/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/hiec/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu
#SBATCH --array=0-41

# Args
script=/home/spn1560/hiec/scripts/predict.py
run_id=(
    cfb7fdbeb07a4967affe7f4485957236
    8a016e73833f4a9f8e3127ab05a8f448
    e3738fe315f64c928c44fe4f8ac6de24
    ec68f688ce774c09a5811c38a38b1f8f
    0c19be31db744a0e8a0c87905f46c81a
    6c60b4f3d0d74c4a9c71ab58d8ca79df
    825c6f409237497991112112d2432f48
    d651d3368fc8427383ab68266359966f
    707754f2458b467d871b76e7cf93b8df
    4b7d87b9ffba476e82cadf81c28dd3f8
    ac3565b3fe984552922766a57944a54a
    103f73d69ff640c99bb8645a2e9146c6
    9f917717289f4e9a89564564a38361f3
    28abcb5c5573447fa20f212703d5d597
    8cff9872dc0441e5a7d1561d75e41149
    7725706057394618bc1c2fad58614627
    0917682b20b545a59ed98c2d423c56d1
    e109a0a7d3564f99bb6688796405dbf1
    6bb1753ed0b14a6fb6eddb0d0b3679a5
    e54a25eafc65463a95dee11b4a46e4cd
    f0d279b0ffed4544bc23a81327e449d0
    000fc5f378954a27bf934ccd58241a57
    bafce811361c4d19afe7919fb4666fb2
    f1cf506a0c0f48b49bbf491f51db24f7
    c4fdab61494a4affb7b48b3f18b0a314
    a91ff6acc5a848f18a179c42f5127a85
    36a93d8131bd460599ef98740f8d322d
    120f090f4b3e471a965979401b3271ae
    34cda1e3f64a4b18a98894ed0e86be86
    e627a08cec3c47e18b4fd9f1ade6b9fe
    7e836b6654ac4b9aba81fe692b059c5d
    ec2310e5872f47de8b9c81d1edce26b9
    04cc886dade64a68a84b151ddd92cef2
    55c062d229394c4f82828814300c0f2b
    56c0235555af4e4c98dd6293c824cc2f
    1d6c62f5f2d44b3fa82851639fb44ead
    30ed98a93bb149f6b7b6c08779e95915
    35d782fa639e4035b08d5c74a4f002f9
    874ef317e1b7403c9be3665c3c301d90
    af29e2f62cf34bf2824dc8c72e086da5
    da9ed201898a451a8a822cf1a30ec30b
    d1a3cc3f1d36488e8222efda0d595914
)
# Commands
ulimit -c 0
module purge
module load gcc/9.2.0
eval "$(conda shell.bash hook)"
conda activate /home/spn1560/.conda/envs/hiec2
python $script run_id=${run_id[$SLURM_ARRAY_TASK_ID]}