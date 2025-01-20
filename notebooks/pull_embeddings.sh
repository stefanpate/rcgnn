#!/bin/bash

pids=$(awk '{print $1}' test_embeddings.txt)

for pid in $pids
do
    rsync -nv spn1560@quest.northwestern.edu:/projects/p30041/spn1560/hiec/data/sprhea/esm/$pid.pt /home/stef/quest_data/hiec/data/sprhea/esm
done