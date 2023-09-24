# Hierarchical EC prediction

## Installation. 

## Directories
1. configs 
2. data 
3. scripts
4. src (packages and modules)
5. artifacts (stores results)

## Dev Notes
1. Whenever you push to remote, please: 1) export conda environment and requirements.txt, 2) black and vulture
2. Never push data/ to github (too big), instead use DVC to store it
3. Use dagshub and MLFlow to keep track of experimental results and parameters
4. Use tensorboard 
5. Use the git hash when saving images

    A. git add .

    B. git commit -m "finished changing XYZ"

    C. python run_script_to_generate_figure.py

    D. git add .

    E. git commit -m "ran new code and generated new figures changing XYZ"

## Scripts (in order)
1. download_proteins.py contains the script to get proteins from Uniprot
2. esm_embeddings.py uses ESM encoder to generate embeddings for each protein
3. process.py splits data into train and test sets. Also creates pairs of proteins for self-supervised learning (pending decision whether to do it a priori or during train)
4. train.py trains model
5. evaluate.py evaluates model on external sources