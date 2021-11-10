#!/bin/bash
cd ~/PycharmProjects/TurkRADBERT/pytorch/src/pretraining

eval "$(conda shell.bash hook)"
conda deactivate
conda activate pretraining

cd bert

python run_pretraining.py --input_file=gs://pretraining/cased_tfrecords/*.tfrecord \
--output_dir=gs://pretraining/bert-base-turkish-cased --bert_config_file=config.json  --init_checkpoint=bert-base-turkish-cased-tf/model.ckpt.data-00000-of-00001 \
--max_seq_length=512 --max_predictions_per_seq=75 --do_train=True \
--train_batch_size=8 --num_train_steps=1000000 --num_warmup_steps=10000 --learning_rate=1e-4 \
--save_checkpoints_steps=100000 --keep_checkpoint_max=20

