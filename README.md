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
0. scrapy to get all valid ec_numbers (optional), else just use data/ec.txt

   >  scrapy shell 
   >  fetch ('https://www.brenda-enzymes.org/all_enzymes.php')
   >  response.xpath('//div[@class="equal"]/div[@class="row"]/div[1]/text()').getall() # save to data/ec.txt

1. download_proteins.py contains the script to get proteins from Uniprot (Eric)

   Version 1. Use data/ec.txt to guide collection from uniprot restapi

      There are 8423 unique EC numbers, 71314 proteins

   Version 2. Get all 40M uniprot accession_ids (todo later)


   Format: [{ec_number: [{accession_id1: sequence1}, {accession_id2: sequence2}]]

   References
   A. https://www.uniprot.org/help/return_fields

1b. make_fasta.py contains the script to convert json to fasta file 
   
   Format: 
   """> accession_id | ec_number
   AAVAWEAGKPLSIEEIEVAPPK
   """"

2. esm_embeddings.py uses ESM encoder to generate embeddings for each protein (Eric)
3. process.py splits data into train and test sets. Also creates pairs of proteins for self-supervised learning (pending decision whether to do it a priori or during train)
4. train.py trains model (Stefan handles this)
5. evaluate.py evaluates model on external sources (Stefan handles this)



#  Stefan does
1. git clone <url>
2. git checkout -b <new_branch> # stefan makes a new branch... (training branch; diff from master branch)
3. git add . git commit -m "asdfasdf" -> new_branch
4. git add . git commit -m "asdfasdfzxcvzxcv" -> new_branch
5. git add . git commit -m "zxoiweprwe"
5. Submit a pull request (emailing me your best essay)
5. The other person will review it (pull it and review it), and if it is good, then we incorporate it into main branch


model (new branch: model_updates) -> DL 3 layers (commit) -> 4 layers (commit) -> 5 layers (commit)
loss funciton (new branch: loss_function_updates) -> DL 4 layers + CE -> DL 4 layers + BCE -> ...

model_loss_function (new branch: loss_funciton model updates)
