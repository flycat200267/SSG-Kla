import os
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmModel, EsmConfig, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_PATH = "train_data.xlsx"
SAPROT_MODEL_PATH = "Saprot_Local"
NPZ_PATH = "afdb_51mer_coords.npz"
POCKET_NPZ_PATH = "afdb_spatial_pocket.npz"
SAVE_DIR = "./checkpoints_modify"
os.makedirs(SAVE_DIR, exist_ok=True)
MAX_LEN = 128
BATCH_SIZE = 16
ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
SEED = 42
NUM_FOLDS = 5
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
    def forward(self, x, mask):
        attn_weights = self.attention(x).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
class RBFExpansion(nn.Module):
    def __init__(self, K=16, d_min=0.0, d_max=20.0):
        super().__init__()
        self.K = K
        self.means = nn.Parameter(torch.linspace(d_min, d_max, K))
        self.stds = nn.Parameter(torch.ones(K) * ((d_max - d_min) / K))
    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return torch.exp(-((dist - self.means) / self.stds) ** 2)
class EquivariantGraphAttentionLayer(nn.Module):
    def __init__(self, node_dim, pocket_dim, hidden_dim=256, rbf_dim=16):
        super().__init__()
        self.node_compress = nn.Linear(node_dim + pocket_dim, hidden_dim)
        self.rbf = RBFExpansion(K=rbf_dim, d_min=0.0, d_max=20.0)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.attn_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim))
    def forward(self, h_seq, pos, pocket_feats, mask):
        B, L, _ = h_seq.shape
        h_combined = torch.cat([h_seq, pocket_feats], dim=-1)
        h_comp = self.node_compress(h_combined)
        if self.training:
            noise = torch.randn_like(pos) * 0.1 * mask.unsqueeze(-1)
            pos = pos + noise
        dist_sq = torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1)
        dist = torch.sqrt(dist_sq + 1e-6)
        rbf_feat = self.rbf(dist)
        h_i = h_comp.unsqueeze(2).expand(B, L, L, -1)
        h_j = h_comp.unsqueeze(1).expand(B, L, L, -1)
        edge_inputs = torch.cat([h_i, h_j, rbf_feat], dim=-1)
        m_ij = self.edge_mlp(edge_inputs)
        alpha_ij = self.attn_mlp(m_ij)
        m_ij = m_ij * alpha_ij
        pocket_mask = (dist < 10.0).float().unsqueeze(-1)
        m_ij = m_ij * pocket_mask
        mask_mat = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        m_ij = m_ij * mask_mat
        m_i = torch.sum(m_ij, dim=2)
        h_updated = self.node_update(torch.cat([h_comp, m_i], dim=-1))
        return h_seq + h_updated
class SaProtEGATCrossModalClassifier(nn.Module):
    def __init__(self, saprot_model_path, pocket_dim):
        super().__init__()
        self.saprot = EsmModel.from_pretrained(saprot_model_path, torch_dtype=torch.bfloat16)
        self.saprot.gradient_checkpointing_enable()
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=128, lora_dropout=0.1, target_modules=["query", "key", "value", "dense", "dense_h_to_4h", "dense_4h_to_h"])
        self.saprot = get_peft_model(self.saprot, peft_config)
        if hasattr(self.saprot, "enable_input_require_grads"):
            self.saprot.enable_input_require_grads()
        self.hidden_dim = self.saprot.config.hidden_size
        self.egat = EquivariantGraphAttentionLayer(node_dim=self.hidden_dim, pocket_dim=pocket_dim, hidden_dim=256, rbf_dim=16)
        self.seq_pooler = AttentionPooling(self.hidden_dim)
        self.geo_pooler = AttentionPooling(self.hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, input_ids, attention_mask, coords, pocket_feats, valid_mask, labels=None):
        outputs = self.saprot(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        seq_out = outputs.last_hidden_state
        geo_out = self.egat(seq_out, coords.to(seq_out.dtype), pocket_feats.to(seq_out.dtype), valid_mask)
        k_seq_feat = self.seq_pooler(seq_out, valid_mask)
        k_geo_feat = self.geo_pooler(geo_out, valid_mask)
        fused_feat = torch.cat([k_seq_feat, k_geo_feat], dim=-1)
        logits = self.classifier(fused_feat)
        if labels is not None:
            loss = self.bce_loss(logits.view(-1).float(), labels.view(-1).float())
            return loss, logits
        return logits
class EGATDataset(Dataset):
    def __init__(self, df, tokenizer, coords_array, pocket_array, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.coords_array = coords_array
        self.pocket_array = pocket_array
        self.max_len = max_len
        self.pocket_dim = pocket_array.shape[-1]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        row = self.df.iloc[index]
        seq = str(row['Sequence']).strip().replace("U", "X").replace("Z", "X").replace("O", "X")
        label = torch.tensor(row['Label'], dtype=torch.float)
        foldseek_seq = str(row['foldseek_seq']).strip() if pd.notna(row['foldseek_seq']) else seq
        min_len = min(len(seq), len(foldseek_seq))
        seq, foldseek_seq = seq[:min_len], foldseek_seq[:min_len]
        interleaved = "".join([f"{aa}{st}" for zip_res in zip(seq, foldseek_seq) for aa, st in [zip_res]])
        encoding = self.tokenizer(interleaved, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=True)
        coords_aa = torch.tensor(self.coords_array[index], dtype=torch.float)[:min_len]
        coords_expanded = coords_aa.repeat_interleave(2, dim=0)
        pocket_aa = torch.tensor(self.pocket_array[index], dtype=torch.float)[:min_len]
        pocket_expanded = pocket_aa.repeat_interleave(2, dim=0)
        coords_final = torch.zeros((self.max_len, 3))
        pocket_final = torch.zeros((self.max_len, self.pocket_dim))
        valid_mask = torch.zeros(self.max_len)
        valid_expand_len = min(coords_expanded.shape[0], pocket_expanded.shape[0], self.max_len - 2)
        if valid_expand_len > 0:
            coords_final[1: 1 + valid_expand_len, :] = coords_expanded[:valid_expand_len, :]
            pocket_final[1: 1 + valid_expand_len, :] = pocket_expanded[:valid_expand_len, :]
            is_valid = (coords_expanded[:valid_expand_len, :].abs().sum(dim=-1) > 1e-5).float()
            valid_mask[1: 1 + valid_expand_len] = is_valid
        return {'input_ids': encoding['input_ids'].squeeze(), 'attention_mask': encoding['attention_mask'].squeeze(), 'coords': coords_final, 'pocket_feats': pocket_final, 'valid_mask': valid_mask, 'label': label}
def train_one_epoch_crossmodal(model, dataloader, optimizer, scheduler, epoch_idx):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1} Train", leave=False)
    for step, batch in enumerate(pbar):
        input_ids, attn_mask, coords = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['coords'].to(DEVICE)
        pocket_feats, valid_mask, labels = batch['pocket_feats'].to(DEVICE), batch['valid_mask'].to(DEVICE), batch['label'].to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, _ = model(input_ids, attn_mask, coords, pocket_feats, valid_mask, labels)
            loss = loss / ACCUM_STEPS
        loss.backward()
        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        current_loss = loss.item() * ACCUM_STEPS
        total_loss += current_loss
        pbar.set_postfix({'loss': f"{current_loss:.4f}"})
    return total_loss / len(dataloader)
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    probs_all, labels_all = [], []
    for batch in dataloader:
        input_ids, attn_mask, coords = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['coords'].to(DEVICE)
        pocket_feats, valid_mask, labels = batch['pocket_feats'].to(DEVICE), batch['valid_mask'].to(DEVICE), batch['label'].to(DEVICE)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, attn_mask, coords, pocket_feats, valid_mask)
        probs = torch.sigmoid(logits).float().cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()
        probs_all.extend(probs)
        labels_all.extend(labels)
    preds_binary = (np.array(probs_all) >= 0.5).astype(int)
    acc = accuracy_score(labels_all, preds_binary)
    mcc = matthews_corrcoef(labels_all, preds_binary)
    try:
        auc = roc_auc_score(labels_all, probs_all)
        ap = average_precision_score(labels_all, probs_all)
    except:
        auc, ap = 0.5, 0.5
    tn, fp, fn, tp = confusion_matrix(labels_all, preds_binary).ravel()
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, auc, mcc, ap, sen, spec, probs_all, labels_all
def main():
    torch.cuda.empty_cache()
    print(f"Starting SSG-Kla Distance 10.0 / Noise 0.1")
    tokenizer = EsmTokenizer.from_pretrained(SAPROT_MODEL_PATH)
    df = pd.read_excel(DATA_PATH).dropna(subset=['Label'])
    df['Label'] = df['Label'].astype(int)
    labels = df['Label'].tolist()
    all_coords = np.load(NPZ_PATH)['coords']
    pocket_raw_data = np.load(POCKET_NPZ_PATH)
    pocket_key = list(pocket_raw_data.keys())[0]
    all_pockets = pocket_raw_data[pocket_key]
    pocket_feature_dim = all_pockets.shape[-1]
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    all_predictions = np.zeros(len(df))
    fold_best_aucs = []
    for fold, (train_index, val_index) in enumerate(skf.split(df['Sequence'], labels)):
        print(f"\n{'=' * 20} Starting Fold {fold + 1}/{NUM_FOLDS} {'=' * 20}")
        train_ds = EGATDataset(df.iloc[train_index], tokenizer, all_coords[train_index], all_pockets[train_index], MAX_LEN)
        val_ds = EGATDataset(df.iloc[val_index], tokenizer, all_coords[val_index], all_pockets[val_index], MAX_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        model = SaProtEGATCrossModalClassifier(SAPROT_MODEL_PATH, pocket_dim=pocket_feature_dim).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.15)
        total_steps = len(train_loader) * NUM_EPOCHS // ACCUM_STEPS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
        best_auc = 0.0
        best_model_path = os.path.join(SAVE_DIR, f"saprot_modify_fold_{fold}_best.pth")
        patience, patience_counter = 8, 0
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch_crossmodal(model, train_loader, optimizer, scheduler, epoch)
            acc, auc, mcc, ap, sen, spec, _, _ = evaluate(model, val_loader)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {train_loss:.4f} | AUC: {auc:.4f} | ACC: {acc:.4f} | MCC: {mcc:.4f} | AP: {ap:.4f} | SEN: {sen:.4f} | SPEC: {spec:.4f}")
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping.")
                    break
        print(f"Fold {fold + 1} Best AUC: {best_auc:.4f}")
        fold_best_aucs.append(best_auc)
        model.load_state_dict(torch.load(best_model_path))
        _, _, _, _, _, _, fold_probs, _ = evaluate(model, val_loader)
        all_predictions[val_index] = fold_probs
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
    print(f"\n{'=' * 30}")
    print(f"Average AUC: {np.mean(fold_best_aucs):.4f}")
    os.makedirs("./scores", exist_ok=True)
    results_df = pd.DataFrame({"label": labels, "score": all_predictions})
    output_csv = "./scores/SaProt_modify_results.csv"
    results_df.to_csv(output_csv, index=False)
if __name__ == "__main__":
    main()