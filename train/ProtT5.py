import pandas as pd
import numpy as np
import torch
import os
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef, recall_score, f1_score
from datasets import Dataset
from transformers import T5Tokenizer, T5EncoderModel, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_callback import PrinterCallback
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_model, LoraConfig, TaskType
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE_PATH = 'final_train_data.xlsx'
MODEL_CHECKPOINT = "protT5_local"
CHECKPOINT_ROOT = "./protT5_checkpoints_fusion"
MAX_LEN = 51
BATCH_SIZE = 8
ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
EPOCHS = 30
NUM_FOLDS = 5
class EpochLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            epoch = logs.get("epoch", state.epoch)
            loss = logs.get("loss", 0.0)
            grad = logs.get("grad_norm", 0.0)
            lr = logs.get("learning_rate", 0.0)
            print(f"Epoch: {epoch:.0f} | Loss: {loss:.4f} | Grad: {grad:.4f} | LR: {lr:.2e}")
def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['Label']).reset_index(drop=True)
    df['Label'] = df['Label'].astype(int)
    sequences = df['Sequence'].astype(str).str.upper().tolist()
    labels = df['Label'].tolist()
    return sequences, labels, df
class ProtT5ForSequenceClassification(nn.Module):
    def __init__(self, model_path, num_labels=1):
        super().__init__()
        print(f"Loading ProtT5 base: {model_path}")
        self.base_model = T5EncoderModel.from_pretrained(model_path)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q", "v", "k", "o"])
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.base_model.print_trainable_parameters()
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
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
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
    return {"auc": auc}
def main():
    if not os.path.exists(CHECKPOINT_ROOT):
        os.makedirs(CHECKPOINT_ROOT)
    sequences, labels, df_raw = load_data(FILE_PATH)
    uniprot_ids = df_raw['UniProt_ID'].tolist()
    tokenizer = T5Tokenizer.from_pretrained(MODEL_CHECKPOINT, do_lower_case=False, legacy=False)
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    all_predictions = np.zeros(len(df_raw))
    fold_aucs = []
    for fold, (train_index, test_index) in enumerate(skf.split(sequences, labels)):
        print(f"\n{'=' * 20} Starting Fold {fold + 1}/{NUM_FOLDS} {'=' * 20}")
        train_seqs = [sequences[i] for i in train_index]
        test_seqs = [sequences[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]
        def preprocess_function(examples):
            seqs_cleaned = [s.replace('U', 'X').replace('Z', 'X').replace('O', 'X') for s in examples["sequence"]]
            seqs_spaced = [" ".join(list(seq)) for seq in seqs_cleaned]
            encodings = tokenizer(seqs_spaced, max_length=MAX_LEN, padding="max_length", truncation=True)
            encodings["labels"] = examples["label"]
            return encodings
        train_ds = Dataset.from_dict({"sequence": train_seqs, "label": train_labels})
        test_ds = Dataset.from_dict({"sequence": test_seqs, "label": test_labels})
        train_ds = train_ds.map(preprocess_function, batched=True)
        test_ds = test_ds.map(preprocess_function, batched=True)
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        model = ProtT5ForSequenceClassification(MODEL_CHECKPOINT)
        fold_output_dir = os.path.join(CHECKPOINT_ROOT, f"fold_{fold}")
        args = TrainingArguments(output_dir=fold_output_dir, overwrite_output_dir=False, eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch", save_total_limit=2, learning_rate=LEARNING_RATE, per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE * 2, gradient_accumulation_steps=ACCUM_STEPS, num_train_epochs=EPOCHS, weight_decay=0.01, lr_scheduler_type="cosine", load_best_model_at_end=True, metric_for_best_model="auc", greater_is_better=True, warmup_ratio=0.1, remove_unused_columns=False, run_name=f"protT5_fusion_fold_{fold}", report_to="none", save_safetensors=False)
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=6)])
        trainer.remove_callback(PrinterCallback)
        trainer.add_callback(EpochLogCallback)
        last_checkpoint = get_last_checkpoint(fold_output_dir)
        if last_checkpoint:
            print(f"Found checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
        preds_output = trainer.predict(test_ds)
        logits = preds_output.predictions[0] if isinstance(preds_output.predictions, tuple) else preds_output.predictions
        probs = 1 / (1 + np.exp(-logits))
        probs = probs.flatten()
        all_predictions[test_index] = probs
        binary_preds = (probs >= 0.5).astype(int)
        fold_auc = roc_auc_score(test_labels, probs)
        fold_aupr = average_precision_score(test_labels, probs)
        fold_acc = accuracy_score(test_labels, binary_preds)
        fold_mcc = matthews_corrcoef(test_labels, binary_preds)
        fold_sen = recall_score(test_labels, binary_preds)
        fold_f1 = f1_score(test_labels, binary_preds)
        print(f"\nFold {fold + 1} Training and Evaluation Completed")
        print(f"AUC:  {fold_auc:.4f} | AUPR: {fold_aupr:.4f} | ACC: {fold_acc:.4f}")
        print(f"MCC:  {fold_mcc:.4f} | SEN:  {fold_sen:.4f} | F1:  {fold_f1:.4f}")
        fold_aucs.append(fold_auc)
        save_path = f"./trained_models/ProtT5_Fusion/fold_{fold}_auc{fold_auc:.4f}"
        trainer.save_model(save_path)
        torch.save(model.classifier.state_dict(), os.path.join(save_path, "classifier_head.pth"))
        del model, trainer
        torch.cuda.empty_cache()
    os.makedirs("./scores", exist_ok=True)
    results_df = pd.DataFrame({"UniProt_ID": uniprot_ids, "Label": df_raw['Label'].values, "Predict_Prob": all_predictions})
    results_df.to_csv(f"./scores/ProtT5_Fusion_results.csv", index=False)
    print(f"\n{'=' * 30}")
    print(f"Best AUC for each fold: {[round(x, 4) for x in fold_aucs]}")
    print(f"Average AUC: {np.mean(fold_aucs):.4f}")
    print(f"Results saved to: ./scores/ProtT5_Fusion_results.csv")
    print(f"{'=' * 30}")
if __name__ == "__main__":
    main()