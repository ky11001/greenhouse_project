#!/bin/bash

python create_dataset.py -dir "medium_ws" -name "medium" -ntrj 800

python multigpu_train.py -dir "medium_ws" -name e2e_medium_ws -f_name medium -act_mode abs_delta -lr 0.000095 -bs 12 -hd 256 -exp ws -save_every 10000 -record_every 300 -eval_every 25000 --num_clusters 10 --clusters_to_mask 3
