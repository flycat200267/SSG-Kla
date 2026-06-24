import pandas as pd
import numpy as np
import torch
import os
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, matthews_corrcoef, recall_score, f1_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import get_last_checkpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = 'train_data.xlsx'
model_checkpoint = "esm2_t30_150M_UR50D"
CHECKPOINT_ROOT = "./esm2_BiGating_checkpoints"
df = pd.read_excel(file_path)
df = df.dropna(subset=['Label']).reset_index(drop=True)
df['Label'] = df['Label'].astype(int)
sequences = list(df['Sequence'])
labels = list(df['Label'])
uniprot_ids = list(df['UniProt_ID'])
class EpochLogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            epoch = logs.get("epoch", state.epoch)
            loss = logs.get("loss", 0.0)
            grad = logs.get("grad_norm", 0.0)
            lr = logs.get("learning_rate", 0.0)
            print(f"Epoch: {epoch:.0f} | Loss: {loss:.4f} | Grad: {grad:.4f} | LR: {lr:.2e}")
class BiGatingFusionHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(BiGatingFusionHead, self).__init__()
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=5, padding=2)
        self.mhsa = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.2)
        self.attn_pool = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.local_to_global_gate = nn.Sequential(nn.Linear(256, hidden_size), nn.Sigmoid())
        self.global_to_local_gate = nn.Sequential(nn.Linear(hidden_size, 256), nn.Sigmoid())
        fusion_dim = hidden_size + 256 + hidden_size
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_labels))
    def forward(self, features):
        center_idx = features.shape[1] // 2
        center_repr = features[:, center_idx, :]
        x_conv = features.transpose(1, 2)
        conv3_pool = torch.max(F.relu(self.conv3(x_conv)), dim=2)[0]
        conv5_pool = torch.max(F.relu(self.conv5(x_conv)), dim=2)[0]
        local_motif = torch.cat([conv3_pool, conv5_pool], dim=1)
        mhsa_out, _ = self.mhsa(features, features, features)
        attn_weights = torch.softmax(self.attn_pool(mhsa_out), dim=1)
        global_repr = torch.sum(mhsa_out * attn_weights, dim=1)
        g_gate = self.local_to_global_gate(local_motif)
        gated_global = global_repr * g_gate
        l_gate = self.global_to_local_gate(global_repr)
        gated_local = local_motif * l_gate
        fused_repr = torch.cat([center_repr, gated_local, gated_global], dim=1)
        logits = self.classifier(fused_repr)
        return logits
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    predictions = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    predictions = predictions[:, 1]
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = 0.5
    return {"auc": auc}
def main():
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    num_sf = 5
    skf = StratifiedKFold(n_splits=num_sf, shuffle=True, random_state=42)
    all_predictions = np.zeros(len(df))
    fold_best_aucs = []
    for fold, (train_index, test_index) in enumerate(skf.split(sequences, labels)):
        print(f"\n{'=' * 20} Starting Fold {fold + 1}/{num_sf} (ESM2 + Bi-directional Gating) {'=' * 20}")
        train_sequences = [sequences[i] for i in train_index]
        test_sequences = [sequences[i] for i in test_index]
        train_labels = [labels[i] for i in train_index]
        test_labels = [labels[i] for i in test_index]
        train_tokenized = tokenizer(train_sequences)
        test_tokenized = tokenizer(test_sequences)
        train_dataset = Dataset.from_dict(train_tokenized)
        test_dataset = Dataset.from_dict(test_tokenized)
        train_dataset = train_dataset.add_column("labels", train_labels)
        test_dataset = test_dataset.add_column("labels", test_labels)
        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, classifier_dropout=0.4)
        for param in model.parameters():
            param.requires_grad = True
        hidden_size = model.config.hidden_size
        model.classifier = BiGatingFusionHead(hidden_size, num_labels)
        model = model.to(DEVICE)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ESM2 + Bi-Gating Architecture | Parameters: {trainable_params / 1e6:.2f}M")
        fold_output_dir = os.path.join(CHECKPOINT_ROOT, f"fold_{fold}")
        args = TrainingArguments(output_dir=fold_output_dir, overwrite_output_dir=False, eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch", save_total_limit=1, learning_rate=4e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=30, weight_decay=0.01, label_smoothing_factor=0.1, load_best_model_at_end=True, metric_for_best_model="auc", greater_is_better=True, lr_scheduler_type="cosine", warmup_ratio=0.1, run_name=f"esm2_bigating_fold_{fold}", report_to="none", disable_tqdm=True)
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset, processing_class=tokenizer, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=6, early_stopping_threshold=0.0001)])
        if trainer.callback_handler.has_callback(PrinterCallback):
            trainer.remove_callback(PrinterCallback)
        if trainer.callback_handler.has_callback(ProgressCallback):
            trainer.remove_callback(ProgressCallback)
        trainer.add_callback(EpochLogCallback)
        last_checkpoint = get_last_checkpoint(fold_output_dir)
        if last_checkpoint:
            print(f"Resuming from {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()
        best_auc = trainer.state.best_metric
        if best_auc is None:
            eval_metrics = trainer.evaluate()
            best_auc = eval_metrics['eval_auc']
        predictions = trainer.predict(test_dataset)
        logits = predictions.predictions
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        fold_predictions = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        fold_predictions = fold_predictions[:, 1]
        all_predictions[test_index] = fold_predictions
        binary_preds = (fold_predictions >= 0.5).astype(int)
        fold_auc = roc_auc_score(test_labels, fold_predictions)
        fold_aupr = average_precision_score(test_labels, fold_predictions)
        fold_acc = accuracy_score(test_labels, binary_preds)
        fold_mcc = matthews_corrcoef(test_labels, binary_preds)
        fold_sen = recall_score(test_labels, binary_preds)
        fold_f1 = f1_score(test_labels, binary_preds)
        print(f"\nFold {fold + 1} Training and Evaluation Completed!")
        print(f"AUC:  {fold_auc:.4f} | AUPR: {fold_aupr:.4f} | ACC: {fold_acc:.4f}")
        print(f"MCC:  {fold_mcc:.4f} | SEN:  {fold_sen:.4f} | F1:  {fold_f1:.4f}")
        fold_best_aucs.append(fold_auc)
        save_path = f"./trained_models/ESM2_BiGating/fold_{fold}_auc{best_auc:.4f}"
        trainer.save_model(save_path)
        del model, trainer
        torch.cuda.empty_cache()
    os.makedirs("./scores", exist_ok=True)
    results_df = pd.DataFrame({"UniProt_ID": uniprot_ids, "Label": df['Label'].values, "Predict_Prob": all_predictions})
    output_file = "./scores/ESM2_BiGating_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 30}")
    print(f"Best AUC for each fold: {[round(x, 4) for x in fold_best_aucs]}")
    print(f"Average AUC: {np.mean(fold_best_aucs):.4f}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 30}")
if __name__ == "__main__":
    main()