# Enzyme-reaction catalyic link prediction

## Installation
- Install required packages listed in environment.yml
- Run pip install -e /path/to/root_dir

## Directories
1. artifacts
2. data
3. notebooks
4. scripts
5. src

## Dev Notes


## Scripts
1. aggregate_gs.py - Compiles grid search results for matrix factorization into a single dataframe
2. batch_fit.py - Writes and submits slurm jobs to train models given cross validation / data configurations and hyperparameters to do a grid search over
3. batch_resume.py - Loads previously trained models and resumes training
4. gnn_min_multiclass_cv.py - Fits GNN on reaction operator multiclass classification task
5. mf_fit.py - Fits matrix factorization models
6. seq2esm.py - Generates ESM-1b embedddings given amino acid sequences in a fasta file format
7. two_channel_fit.py - Fits GNN models taking two inputs, i.e., protein-reaction pairs

## Src
atom_mapping.py - Functions to get all-atom-mapping of reactions starting with reaction center
cross_validation.py
   - Dataclasses for batch script parameters & cross validation configurations
   - BatchGridSearch class to handle batch submission of slurm jobs during cross validation and/or grid search
data.py - Datapoint and dataset objects and some supporting functions
featurizer.py - Reaction featurizers; customized chemprop classes
mf.py - Matrix factorization classes
model.py - Customized chemprop MPNN and several more models, subclassed lightning modules
nn.py - Pytorch model components, message passing, prediction heads, aggregation
utils.py - Basic data handling utilities