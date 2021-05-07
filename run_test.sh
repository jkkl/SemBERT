CUDA_VISIBLE_DEVICES=-1 python run_intention_classifier.py \
--data_dir data/intention2_v2/ \
--task_desc intention2_v2_32_moresmalltag_2spect \
--task_name intention \
--train_batch_size 32 \
--max_seq_length 32 \
--bert_model bert-base-chinese \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--do_test \
--do_lower_case \
--max_num_aspect 2 \
--output_dir saved_models/intention2_v2_32_moresmalltag_2spect 
