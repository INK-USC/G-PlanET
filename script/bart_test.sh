python ./main/bart_simpletransformers/train.py --model facebook/bart-base \
    --train_data_path data/train_data/train_table_seq_iter.jsonl \
    --eval_data_path data/valid_data/valid_unseen_table_seq_iter.jsonl \
    --max_seq_length 1024 --max_length 256 --train_batch_size 4 --eval_batch_size 4 --learning_rate 3e-5 --num_train_epochs 1 \
    --output ./model