#!/bin/sh

python create_dataset.py -dir "small_ws" -name "small_100" -ntrj 100
python multigpu_train.py -dir small_ws -name e2e_small_ws_3 -f_name small_100 -act_mode abs_delta -lr 0.000095 -bs 13 -hd 256 -exp ws -save_every 10000 -record_every 300 -eval_every 20000 --num_clusters 12 --clusters_to_mask 3
