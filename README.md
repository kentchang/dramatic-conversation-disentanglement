# dramatic-conversation-disentanglement

Code to support Kent K. Chang, Danica Chen, David Bamman, “Dramatic Conversation Disentanglement.” ACL 2023 Findings. 

N.B. the code is not thoroughly cleaned. Please contact kentkchang [at] berkeley.edu if you have any questions.

## Setup

```
pip install -r requirements.txt
```

## Train

1. BERT baseline 

```
accelerate launch --config_file accelerate_config.yaml train_baseline.py \
  --encoder_name DialogueLineEncoder \
  --epochs 6 \
  --batch_size 4 \
  --use_tqdm True \
  --train_file train.tsv \
  --dev_file dev.tsv \
  --model_name bert-base-cased \
  --num_negative_examples 5
```

2. 6-way classifier, based on [Structural Characterization for Dialogue Disentanglement](https://aclanthology.org/2022.acl-long.23/) (Ma et al., ACL 2022) -- original implementation [here](https://github.com/xbmxb/StructureCharacterization4DD)

```
accelerate launch --config_file accelerate_config.yaml train_4DD.py \
  --epochs 10 \
  --batch_size 2 \
  --train_file train.tsv \
  --dev_file dev.tsv \
  --max_previous_utterance 6 \
  --model_name bert-base-cased \
  --model_output output \
  --log_output log \
  --use_tqdm True
```

3. Featurized

```
accelerate launch --config_file accelerate_config.yaml train_linear.py \
  --encoder_name LogisticRegression \
  --epochs 6 \
  --batch_size 4 \
  --train_file train.tsv \
  --dev_file dev.tsv \
  --model_name bert-base-cased \
  --num_negative_examples 5
```

4. Multi-task

```
accelerate launch --config_file accelerate_config.yaml train_multitask.py \
  --encoder_name MultitaskDialogueEncoder \
  --epochs 10 \
  --batch_size 4 \
  --train_file train.tsv \
  --dev_file dev.tsv \
  --model_name bert-base-cased \
  --cluster_weight 0.5 \
  --num_negative_examples 5
```

## Inference

Download the trained model (6-way classifier, `4DD`) [here](https://yosemite.ischool.berkeley.edu/kentkchang/dcd_pytorch_model-01062023-221722-epoch4.bin) and put `dcd_pytorch_model-01062023-221722-epoch4.bin` in `trained_models`. 

```
accelerate launch --config_file accelerate_config_2proc.yaml run_test-take5-inference_4DD.py \
  --batch_size 12 \
  --max_previous_utterance 6 \
  --model_path  dcd_pytorch_model-01062023-221722-epoch4.bin \
  --model_folder trained_models \
  --model_name bert-base-cased \
  --test_folder data/ \
  --preds_output_folder output/
```