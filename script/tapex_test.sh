export TRAINED_TAPEX_MODEL_DIR=tapex_simple

python robosense/methods/table_transformers/train_tapex.py \
--do_predict \
--test_file data/test_data/tests_unseen_table_seq_tapex.jsonl \
--output_dir tapex_simple_predict \
--resume_from_checkpoint $TRAINED_TAPEX_MODEL_DIR \
--model_name_or_path microsoft/tapex-large \
--per_device_eval_batch_size 2 \
--predict_with_generate \
--num_beams 5 \
--val_max_target_length 256