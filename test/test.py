import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5Tokenizer, T5EncoderModel, EsmTokenizer, \
    EsmModel
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef, confusion_matrix, average_precision_score, \
    f1_score
from scipy.special import expit, logit
import warnings

warnings.filterwarnings('ignore')

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_CSV_PATH = "test_data.xlsx"
TEST_NPZ_PATH = "afdb_51mer_coords.npz"
TEST_POCKET_NPZ_PATH = "afdb_spatial_pocket.npz"
OUT_DIR = "./test_scores"
os.makedirs(OUT_DIR, exist_ok=True)
NUM_FOLDS = 5

ESM2_BASE = "esm2_t30_150M_UR50D"
ESM2_CKPT_DIR = "trained_models/ESM2_BiGating"

T5_BASE = "protT5_local"
T5_CKPT_DIR = "ProtT5_Fusion"

SAPROT_BASE = "Saprot_Local"
SAPROT_CKPT_DIR = "checkpoints_modify"

THETA_ESM = 0.561
THETA_T5 = 0.287
THETA_SAPROT = 0.152
LAMBDA_PENALTY = 0.7959


def calculate_entropy(probs):
    eps = 1e-9
    entropy = - (probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
    return entropy / np.log(2)


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
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_labels)
        )

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
        return self.classifier(fused_repr)


def run_esm2(peplist):
    print("\n[1/3] Starting ESM2 5-fold inference...")
    all_fold_probs = []
    tokenizer = AutoTokenizer.from_pretrained(ESM2_BASE)

    for fold in range(NUM_FOLDS):
        pattern = os.path.join(ESM2_CKPT_DIR, f"fold_{fold}_auc*")
        matched = glob.glob(pattern)
        if not matched:
            continue
        model_dir = matched[0]
        model = AutoModelForSequenceClassification.from_pretrained(
            ESM2_BASE, num_labels=2, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, classifier_dropout=0.4
        )
        model.classifier = BiGatingFusionHead(model.config.hidden_size, 2)
        if os.path.exists(f"{model_dir}/pytorch_model.bin"):
            model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location=DEVICE))
        else:
            model.load_state_dict(load_file(f"{model_dir}/model.safetensors"))
        model.to(DEVICE).eval()

        fold_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(peplist), 32), desc=f"ESM2 Fold {fold + 1}"):
                batch = peplist[i: i + 32]
                encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
                logits = model(**encoded).logits
                exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])
                probs = (exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True))[:, 1]
                fold_probs.extend(probs.cpu().numpy())
        all_fold_probs.append(fold_probs)
        del model
        torch.cuda.empty_cache()
    return np.mean(all_fold_probs, axis=0)
class ProtT5ForSequenceClassification(nn.Module):
    def __init__(self, model_path, num_labels=1):
        super().__init__()
        self.base_model = T5EncoderModel.from_pretrained(model_path)
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1,
                                 target_modules=["q", "v", "k", "o"])
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.lstm_hidden = 128
        self.bilstm = nn.LSTM(input_size=1024, hidden_size=self.lstm_hidden, num_layers=1, batch_first=True,
                              bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(self.lstm_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Linear(1024 + self.lstm_hidden * 2, 512), nn.LayerNorm(512), nn.ReLU(),
                                        nn.Dropout(0.3), nn.Linear(512, num_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        center_idx = last_hidden_state.shape[1] // 2
        center_repr = last_hidden_state[:, center_idx, :]
        lstm_out, _ = self.bilstm(last_hidden_state)
        attn_weights = self.attention(lstm_out)
        attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)
        global_repr = torch.sum(lstm_out * attn_weights, dim=1)
        fused_repr = torch.cat([center_repr, global_repr], dim=1)
        return self.classifier(fused_repr)

def run_prott5(peplist):
    print("\n[2/3] Starting ProtT5 5-fold inference...")
    all_fold_probs = []
    tokenizer = T5Tokenizer.from_pretrained(T5_BASE, do_lower_case=False, legacy=False)
    peplist = [" ".join(list(s.replace('U', 'X').replace('Z', 'X').replace('O', 'X'))) for s in peplist]

    for fold in range(NUM_FOLDS):
        pattern = os.path.join(T5_CKPT_DIR, f"fold_{fold}_auc*")
        matched = glob.glob(pattern)
        if not matched: continue
        model_dir = matched[0]
        model = ProtT5ForSequenceClassification(T5_BASE, 1)
        model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location=DEVICE))
        model.to(DEVICE).eval()

        fold_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(peplist), 16), desc=f"T5 Fold {fold + 1}"):
                batch = peplist[i: i + 16]
                encoded = tokenizer(batch, max_length=51, padding="max_length", truncation=True,
                                    return_tensors="pt").to(DEVICE)
                logits = model(encoded.input_ids, encoded.attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_probs.extend(probs)
        all_fold_probs.append(fold_probs)
        del model
        torch.cuda.empty_cache()
    return np.mean(all_fold_probs, axis=0)

class RBFExpansion(nn.Module):
    def __init__(self, K=16, d_min=0.0, d_max=20.0):
        super().__init__()
        self.means = nn.Parameter(torch.linspace(d_min, d_max, K))
        self.stds = nn.Parameter(torch.ones(K) * ((d_max - d_min) / K))

    def forward(self, dist):
        return torch.exp(-((dist.unsqueeze(-1) - self.means) / self.stds) ** 2)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x, mask):
        attn_weights = self.attention(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return torch.bmm(F.softmax(attn_weights, dim=-1).unsqueeze(1), x).squeeze(1)

class EquivariantGraphAttentionLayer(nn.Module):
    def __init__(self, node_dim, pocket_dim, hidden_dim=256, rbf_dim=16):
        super().__init__()
        self.node_compress = nn.Linear(node_dim + pocket_dim, hidden_dim)
        self.rbf = RBFExpansion(K=rbf_dim, d_min=0.0, d_max=20.0)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.2),
                                      nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.attn_mlp = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.node_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, node_dim))

    def forward(self, h_seq, pos, pocket_feats, mask):
        B, L, _ = h_seq.shape
        h_comp = self.node_compress(torch.cat([h_seq, pocket_feats], dim=-1))
        dist = torch.sqrt(torch.sum((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2, dim=-1) + 1e-6)
        rbf_feat = self.rbf(dist)
        edge_inputs = torch.cat(
            [h_comp.unsqueeze(2).expand(B, L, L, -1), h_comp.unsqueeze(1).expand(B, L, L, -1), rbf_feat], dim=-1)
        m_ij = self.edge_mlp(edge_inputs) * self.attn_mlp(self.edge_mlp(edge_inputs))
        m_ij = m_ij * (dist < 10.0).float().unsqueeze(-1) * (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(-1)
        return h_seq + self.node_update(torch.cat([h_comp, torch.sum(m_ij, dim=2)], dim=-1))

class SaProtEGATCrossModalClassifier(nn.Module):
    def __init__(self, saprot_model_path, pocket_dim):
        super().__init__()
        self.saprot = get_peft_model(EsmModel.from_pretrained(saprot_model_path, torch_dtype=torch.bfloat16),
                                     LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=128,
                                                target_modules=["query", "key", "value", "dense", "dense_h_to_4h",
                                                                "dense_4h_to_h"]))
        self.hidden_dim = self.saprot.config.hidden_size
        self.egat = EquivariantGraphAttentionLayer(self.hidden_dim, pocket_dim)
        self.seq_pooler, self.geo_pooler = AttentionPooling(self.hidden_dim), AttentionPooling(self.hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(),
                                        nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, coords, pocket_feats, valid_mask):
        seq_out = self.saprot(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        geo_out = self.egat(seq_out, coords.to(seq_out.dtype), pocket_feats.to(seq_out.dtype), valid_mask)
        fused_feat = torch.cat([self.seq_pooler(seq_out, valid_mask), self.geo_pooler(geo_out, valid_mask)], dim=-1)
        return self.classifier(fused_feat)

def run_saprot(df):
    print("\n[3/3] Starting SaProt+EGAT 5-fold inference...")
    all_coords = np.load(TEST_NPZ_PATH)['coords']
    pocket_raw_data = np.load(TEST_POCKET_NPZ_PATH)
    all_pockets = pocket_raw_data[list(pocket_raw_data.keys())[0]]
    pocket_feature_dim = all_pockets.shape[-1]
    tokenizer = EsmTokenizer.from_pretrained(SAPROT_BASE)
    all_fold_probs = []
    for fold in range(NUM_FOLDS):
        pattern = os.path.join(SAPROT_CKPT_DIR, f"saprot_modify_fold_{fold}_best.pth")
        matched = glob.glob(pattern)
        if not matched: continue
        ckpt_path = matched[0]
        model = SaProtEGATCrossModalClassifier(SAPROT_BASE, pocket_feature_dim)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE), strict=True)
        model.to(DEVICE).eval()
        fold_probs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(df), 16), desc=f"SaProt Fold {fold + 1}"):
                batch_df = df.iloc[i:i + 16]
                input_ids_list, mask_list, coords_list, pocket_list, valid_list = [], [], [], [], []
                for idx, (_, row) in enumerate(batch_df.iterrows()):
                    seq = str(row['Sequence']).strip().replace("U", "X").replace("Z", "X").replace("O", "X")
                    foldseek_seq = str(row['foldseek_seq']).strip() if pd.notna(row['foldseek_seq']) else seq
                    min_len = min(len(seq), len(foldseek_seq))
                    seq, foldseek_seq = seq[:min_len], foldseek_seq[:min_len]
                    interleaved = "".join([f"{aa}{st}" for aa, st in zip(seq, foldseek_seq)])
                    encoded = tokenizer(interleaved, max_length=128, padding='max_length', truncation=True,
                                        return_tensors='pt')
                    c_expand = torch.tensor(all_coords[i:i + 16][idx], dtype=torch.float)[:min_len].repeat_interleave(2,dim=0)
                    p_expand = torch.tensor(all_pockets[i:i + 16][idx], dtype=torch.float)[:min_len].repeat_interleave(2, dim=0)
                    coords_final, pocket_final, valid_mask = torch.zeros((128, 3)), torch.zeros((128, pocket_feature_dim)), torch.zeros(128)
                    v_len = min(c_expand.shape[0], p_expand.shape[0], 126)
                    if v_len > 0:
                        coords_final[1: 1 + v_len, :] = c_expand[:v_len, :]
                        pocket_final[1: 1 + v_len, :] = p_expand[:v_len, :]
                        valid_mask[1: 1 + v_len] = (c_expand[:v_len, :].abs().sum(dim=-1) > 1e-5).float()
                    input_ids_list.append(encoded['input_ids'].squeeze())
                    mask_list.append(encoded['attention_mask'].squeeze())
                    coords_list.append(coords_final)
                    pocket_list.append(pocket_final)
                    valid_list.append(valid_mask)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(torch.stack(input_ids_list).to(DEVICE), torch.stack(mask_list).to(DEVICE),
                                   torch.stack(coords_list).to(DEVICE), torch.stack(pocket_list).to(DEVICE),
                                   torch.stack(valid_list).to(DEVICE))
                fold_probs.extend(torch.sigmoid(logits).float().cpu().numpy().flatten())
        all_fold_probs.append(fold_probs)
        del model
        torch.cuda.empty_cache()
    return np.mean(all_fold_probs, axis=0) if len(all_fold_probs) > 0 else np.zeros(len(df))

def main():
    print("Reading test set data...")
    df = pd.read_excel(TEST_CSV_PATH)
    peplist = df['Sequence'].tolist()
    labels = df['Label'].values if 'Label' in df.columns else np.zeros(len(peplist))
    s_esm = run_esm2(peplist)
    s_t5 = run_prott5(peplist)
    s_saprot = run_saprot(df)
    pd.DataFrame({"label": labels, "score": s_esm}).to_csv(os.path.join(OUT_DIR, "Test_ESM2_results.csv"), index=False)
    pd.DataFrame({"label": labels, "score": s_t5}).to.csv(os.path.join(OUT_DIR, "Test_ProtT5_results.csv"), index=False)
    pd.DataFrame({"label": labels, "score": s_saprot}).to.csv(os.path.join(OUT_DIR, "Test_SaProt_modify_results.csv"),
                                                              index=False)
    print("\n" + "=" * 60)
    print("Executing dynamic uncertainty fusion based on information entropy...")
    eps = 1e-6
    s_esm = np.clip(s_esm, eps, 1 - eps)
    s_t5 = np.clip(s_t5, eps, 1 - eps)
    s_saprot = np.clip(s_saprot, eps, 1 - eps)
    l_esm, l_t5, l_saprot = logit(s_esm), logit(s_t5), logit(s_saprot)
    u_esm, u_t5, u_saprot = calculate_entropy(s_esm), calculate_entropy(s_t5), calculate_entropy(s_saprot)
    w_esm = THETA_ESM * np.exp(-LAMBDA_PENALTY * u_esm)
    w_t5 = THETA_T5 * np.exp(-LAMBDA_PENALTY * u_t5)
    w_saprot = THETA_SAPROT * np.exp(-LAMBDA_PENALTY * u_saprot)
    w_total = w_esm + w_t5 + w_saprot
    w_esm, w_t5, w_saprot = w_esm / w_total, w_t5 / w_total, w_saprot / w_total
    final_test_prob = expit(w_esm * l_esm + w_t5 * l_t5 + w_saprot * l_saprot)
    df_final = pd.DataFrame({'label': labels, 'score': final_test_prob})
    df_final.to_csv(os.path.join(OUT_DIR, "Test_results.csv"), index=False)
    auc_esm = roc_auc_score(labels, s_esm)
    auc_t5 = roc_auc_score(labels, s_t5)
    auc_saprot = roc_auc_score(labels, s_saprot)
    test_auc = roc_auc_score(labels, final_test_prob)
    preds_binary = (final_test_prob >= 0.5).astype(int)
    test_acc = accuracy_score(labels, preds_binary)
    test_mcc = matthews_corrcoef(labels, preds_binary)
    test_ap = average_precision_score(labels, final_test_prob)
    test_f1 = f1_score(labels, preds_binary)
    _, _, _, tp = confusion_matrix(labels, preds_binary).ravel()
    fn = np.sum((labels == 1) & (preds_binary == 0))
    test_sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    print("\n" + "=" * 60)
    print("Uncertainty-Aware Test Results")
    print("=" * 60)
    print("Single-modal AUC:")
    print(f"   ESM2: {auc_esm:.5f} | ProtT5: {auc_t5:.5f} | SaProt: {auc_saprot:.5f}")
    print("-" * 60)
    print("Dynamic fusion performance:")
    print(f"   AUC : {test_auc:.5f}")
    print(f"   ACC : {test_acc:.5f}")
    print(f"   MCC : {test_mcc:.5f}")
    print(f"   SEN : {test_sen:.5f}")
    print(f"   F1  : {test_f1:.5f}")
    print(f"   AP  : {test_ap:.5f}")
    print("=" * 60)
    print(f"Prediction pipeline completed. Final results saved to: {os.path.join(OUT_DIR, 'Test_results.csv')}")


if __name__ == "__main__":
    main()