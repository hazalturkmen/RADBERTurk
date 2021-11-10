#!/bin/bash
cd ~/PycharmProjects/TurkRADBERT/pytorch/src/pretraining

eval "$(conda shell.bash hook)"
conda deactivate
conda activate pretraining


split -l 100 /home/hazal/nlp_dataset/brain_CT/example/tr_medical_300.txt  tr-cased-shards-
mkdir cased_shards
mv tr-* cased_shards

cd bert
export NUM_PROC=5



find ../cased_shards -type f | xargs -I% -P $NUM_PROC -n 1 \
python create_pretraining_data.py --input_file % --output_file %.tfrecord \
--vocab_file ../bert-base-turkish-cased-tf/vocab.txt --do_lower_case=False -max_seq_length=512 \
--max_predictions_per_seq=75 --masked_lm_prob=0.15 --random_seed=12345 \
--dupe_factor=5


# shellcheck disable=SC2103
cd ..
mkdir cased_tfrecords/
mv cased_shards/*.tfrecord cased_tfrecords/

