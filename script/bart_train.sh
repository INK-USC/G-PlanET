python ./main/bart_simpletransformers/test.py --model $MODEL_PATH \
    --test_data_path $DATA_PATH \
    --max_seq_length 32 --max_length 256 --train_batch_size 4 --eval_batch_size 4 --learning_rate 3e-5 --num_train_epochs 5 --num_beams 4 \
    --output $OUTPUT_NAME \