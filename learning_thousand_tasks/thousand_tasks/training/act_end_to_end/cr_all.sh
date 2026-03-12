#!/bin/bash

# python create_dataset.py -dir "small_ws" -name "small" -ntrj 100
python create_dataset.py -dir "medium_ws" -name "medium" -ntrj 100
python create_dataset.py -dir "large_ws" -name "large" -ntrj 100
