python run_classifier.py \
--bert_model bert-base-uncased-local
--learning_rate 2e-5
--num_train_epochs 2
--do_train --do_eval
--do_lower_case
--max_num_aspect 3
--output_dir glue/snli_model_dir
--warmup_proportion 0.1
--no_cuda
