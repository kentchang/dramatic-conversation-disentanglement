# dramatic-conversation-disentanglement

Code to support Kent K. Chang, Danica Chen, David Bamman, “Dramatic Conversation Disentanglement.” ACL 2023 Findings.

## Train

See 

accelerate launch --config_file accelerate_config.yaml train_baseline.py \
  --encoder_name DialogueLineEncoder \
  --epochs 6 \
  --batch_size 4 \
  --use_tqdm True \
  --train_file train_221206.tsv \
  --dev_file dev_101722.tsv \
  --model_name bert-base-cased \
  --num_negative_examples 5

accelerate launch --config_file accelerate_config.yaml train_linear.py \
  --encoder_name LogisticRegression \
  --epochs 6 \
  --batch_size 4 \
  --train_file train_221206.tsv \
  --dev_file dev_101722.tsv \
  --model_name bert-base-cased \
  --num_negative_examples 5

accelerate launch --config_file accelerate_config.yaml train_multitask.py \
  --encoder_name MultitaskDialogueEncoder \
  --epochs 10 \
  --batch_size 4 \
  --train_file train_221206.tsv \
  --dev_file dev_101722.tsv \
  --model_name bert-base-cased \
  --cluster_weight 0.5 \
  --num_negative_examples 5