#!/bin/bash
# this script is for downloading the netflix dataset and extracting the tar file 
# after install delete tar files
# Usage: ./download.sh

wget https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz
tar -xvf nf_prize_dataset.tar.gz
tar -xvf download/training_set.tar -C ./download/
rm nf_prize_dataset.tar.gz ./download/*.tar
