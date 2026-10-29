import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from datasets import Dataset
from transformers import T5Tokenizer, T5EncoderModel, TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig, TaskType
FILE_PATH = ""  # Training Excel file.
MODEL_CHECKPOINT = ""  # Local ProtT5 model directory.
CHECKPOINT_ROOT = ""  # Training checkpoint directory.
MODEL_OUTPUT_DIR = ""  # Directory for the five trained fold models.
OOF_OUTPUT_PATH = ""  # Output CSV file for out-of-fold predictions.

MAX_LEN = 51
BATCH_SIZE = 8
ACCUM_STEPS = 4
LEARNING_RATE = 0.0001
EPOCHS = 30
NUM_FOLDS = 5

def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['Label'])
    df['Label'] = df['Label'].astype(int)
    sequences = df['Sequence'].astype(str).str.upper().tolist()
    labels = df['Label'].tolist()
    return (sequences, labels, df)

class ProtT5ForSequenceClassification(nn.Module):

    def __init__(self, model_path, num_labels=1):
        super().__init__()
        self.base_model = T5EncoderModel.from_pretrained(model_path)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['q', 'v', 'k', 'o'])
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.lstm_hidden = 128
        self.bilstm = nn.LSTM(input_size=1024, hidden_size=self.lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(self.lstm_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Linear(1024 + self.lstm_hidden * 2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_labels))
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        center_idx = last_hidden_state.shape[1] // 2
        center_repr = last_hidden_state[:, center_idx, :]
        lstm_out, _ = self.bilstm(last_hidden_state)
        attn_weights = self.attention(lstm_out)
        mask = attention_mask.unsqueeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1000000000.0)
        attn_weights = torch.softmax(attn_weights, dim=1)
        global_repr = torch.sum(lstm_out * attn_weights, dim=1)
        fused_repr = torch.cat([center_repr, global_repr], dim=1)
        logits = self.classifier(fused_repr)
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
        output = (logits,)
        if loss is not None:
            output = (loss,) + output
        return output

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = 1 / (1 + np.exp(-logits))
    auc = roc_auc_score(labels, probs)
    return {'auc': auc}

def configure_paths() -> None:
    names = ('FILE_PATH', 'MODEL_CHECKPOINT', 'CHECKPOINT_ROOT', 'MODEL_OUTPUT_DIR', 'OOF_OUTPUT_PATH',)
    missing = [name for name in names if not str(globals()[name]).strip()]
    if missing:
        raise ValueError("Set the following paths before running: " + ", ".join(missing))
    for name in names:
        globals()[name] = Path(globals()[name]).expanduser()

def main():
    configure_paths()
    if not os.path.exists(CHECKPOINT_ROOT):
        os.makedirs(CHECKPOINT_ROOT)
    sequences, labels, df_raw = load_data(FILE_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT, do_lower_case=False, legacy=False)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    all_predictions = np.zeros(len(df_raw))
    fold_aucs = []
    for fold, (train_index, test_index) in enumerate(skf.split(sequences, labels), start=1):
        model_name = f'ProtT5_branch_fold_{fold}'
        print(f'Fold {fold}/{NUM_FOLDS}')
        train_seqs = [sequences[i] for i in train_index]
        test_seqs = [sequences[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]

        def preprocess_function(examples):
            seqs_cleaned = [s.replace('U', 'X').replace('Z', 'X').replace('O', 'X') for s in examples['sequence']]
            seqs_spaced = [' '.join(list(seq)) for seq in seqs_cleaned]
            encodings = tokenizer(seqs_spaced, max_length=MAX_LEN, padding='max_length', truncation=True)
            encodings['labels'] = examples['label']
            return encodings
        train_ds = Dataset.from_dict({'sequence': train_seqs, 'label': train_labels})
        test_ds = Dataset.from_dict({'sequence': test_seqs, 'label': test_labels})
        train_ds = train_ds.map(preprocess_function, batched=True)
        test_ds = test_ds.map(preprocess_function, batched=True)
        train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        model = ProtT5ForSequenceClassification(MODEL_CHECKPOINT)
        fold_output_dir = os.path.join(CHECKPOINT_ROOT, model_name)
        args = TrainingArguments(output_dir=fold_output_dir, overwrite_output_dir=False, eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, learning_rate=LEARNING_RATE, per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE * 2, gradient_accumulation_steps=ACCUM_STEPS, num_train_epochs=EPOCHS, weight_decay=0.01, lr_scheduler_type='cosine', load_best_model_at_end=True, metric_for_best_model='auc', greater_is_better=True, warmup_ratio=0.1, logging_steps=50, remove_unused_columns=False, run_name=model_name, report_to='none', save_safetensors=False)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=6)])
        last_checkpoint = get_last_checkpoint(fold_output_dir)
        if last_checkpoint:
            print(f'Resuming from checkpoint: {last_checkpoint}')
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
        eval_metrics = trainer.evaluate()
        print(f"Fold {fold} Best AUC: {eval_metrics['eval_auc']:.4f}")
        fold_aucs.append(eval_metrics['eval_auc'])
        preds_output = trainer.predict(test_ds)
        logits = preds_output.predictions[0] if isinstance(preds_output.predictions, tuple) else preds_output.predictions
        probs = 1 / (1 + np.exp(-logits))
        all_predictions[test_index] = probs.flatten()
        save_path = MODEL_OUTPUT_DIR / model_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_path))
        torch.save(model.classifier.state_dict(), save_path / 'classifier_head.pth')
        del model, trainer
        torch.cuda.empty_cache()
    OOF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({'label': df_raw['Label'].values, 'score': all_predictions})
    output_file = OOF_OUTPUT_PATH
    results_df.to_csv(output_file, index=False)
    print(f'Mean AUC: {np.mean(fold_aucs):.4f}')
    print(f'OOF predictions saved to: {output_file}')
if __name__ == '__main__':
    main()
