from __future__ import annotations
import contextlib
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, average_precision_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EsmModel, EsmTokenizer, T5EncoderModel, T5Tokenizer
try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None
PREDICT_DATA_PATH = ""  # Prediction Excel or CSV file.
STRUCTURE_NPZ_PATH = ""  # Aligned structural feature NPZ file.
OUTPUT_PATH = ""  # Output prediction CSV file.
ESM2_MODEL_PATH = ""  # Local ESM-2 model directory.
PROTT5_MODEL_PATH = ""  # Local ProtT5 model directory.
SAPROT_MODEL_PATH = ""  # Local SaProt model directory.
ESM2_CHECKPOINT_DIR = ""  # Parent of ESM2_branch_fold_1 through ESM2_branch_fold_5.
PROTT5_CHECKPOINT_DIR = ""  # Parent of ProtT5_branch_fold_1 through ProtT5_branch_fold_5.
SAPROT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "trained_models" / "Saprot_DAGAT"  # Parent of SaProt_branch_fold_1.pth through SaProt_branch_fold_5.pth.

DEVICE_NAME = os.getenv('SSG_KLA_DEVICE', 'auto')
ESM2_BATCH_SIZE = 16
PROTT5_BATCH_SIZE = 8
SAPROT_BATCH_SIZE = 8
NUM_WORKERS = 0
THETA_ESM2 = None
THETA_PROTT5 = None
THETA_SAPROT = None
LAMBDA = None
WINDOW_SIZE = 51
CENTER_INDEX = 25
MAX_SAPROT_LEN = 64
AA_VOCAB_SIZE = 21
AA_UNKNOWN_ID = 20
AA_EMBED_DIM = 16
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_ID = {aa: i for i, aa in enumerate(AA_ALPHABET)}
SS_VOCAB_SIZE = 9
SS_UNKNOWN_ID = 8
SS_EMBED_DIM = 8
PHYS_DIM = AA_EMBED_DIM + SS_EMBED_DIM + 1
GRAPH_HIDDEN_DIM = 256
RBF_DIM = 16
GRAPH_CUTOFF_ANGSTROM = 10.0
COORD_NOISE_STD_ANGSTROM = 0.1
EPS = 1e-06
SEED = 42

def set_seed(seed: int=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)

def autocast_context(device: torch.device):
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

class BiGatingFusionHead(nn.Module):

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.conv3 = nn.Conv1d(hidden_size, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(hidden_size, 128, kernel_size=5, padding=2)
        self.mhsa = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True, dropout=0.2)
        self.attn_pool = nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
        self.local_to_global_gate = nn.Sequential(nn.Linear(256, hidden_size), nn.Sigmoid())
        self.global_to_local_gate = nn.Sequential(nn.Linear(hidden_size, 256), nn.Sigmoid())
        fusion_dim = hidden_size + 256 + hidden_size
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4), nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, num_labels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
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

def find_fold_dirs(root: Path, branch: str) -> list[Path]:
    if branch not in ('ESM2', 'ProtT5'):
        raise ValueError(f'Unsupported directory-based branch: {branch}')
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Checkpoint root does not exist: {root}')
    paths = [root / f'{branch}_branch_fold_{fold}' for fold in range(1, 6)]
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f'Missing {branch} fold model directories:\n  - ' + '\n  - '.join(missing))
    return paths

def load_state_dict_from_hf_dir(model_dir: Path) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    safetensor_path = model_dir / 'model.safetensors'
    pytorch_path = model_dir / 'pytorch_model.bin'
    if safetensor_path.exists():
        if load_safetensors is None:
            raise ImportError('Checkpoint is model.safetensors but safetensors is unavailable.')
        return load_safetensors(str(safetensor_path), device='cpu')
    if pytorch_path.exists():
        return torch.load(pytorch_path, map_location='cpu')
    raw = list(model_dir.glob('*.pth')) + list(model_dir.glob('*.pt'))
    if raw:
        return torch.load(raw[0], map_location='cpu')
    raise FileNotFoundError(f'No model.safetensors, pytorch_model.bin, .pth or .pt found in {model_dir}')

def load_esm_fold(fold_dir: Path, base_model_path: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(base_model_path), num_labels=2, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, classifier_dropout=0.4)
    model.classifier = BiGatingFusionHead(model.config.hidden_size, 2)
    state = load_state_dict_from_hf_dir(fold_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_unexpected = {'esm.embeddings.position_embeddings.weight'}
    real_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if missing or real_unexpected:
        raise RuntimeError(f'ESM-2 checkpoint mismatch in {fold_dir}\nMissing keys: {missing}\nUnexpected keys: {real_unexpected}\nAll unexpected keys before filtering: {unexpected}')
    model.to(device).eval()
    return (tokenizer, model)

@torch.inference_mode()
def predict_esm2(sequences: list[str], base_model_path: Path, checkpoint_root: Path, device: torch.device, batch_size: int) -> np.ndarray:
    fold_dirs = find_fold_dirs(checkpoint_root, 'ESM2')
    fold_probs = []
    for fold_dir in fold_dirs:
        tokenizer, model = load_esm_fold(fold_dir, base_model_path, device)
        probs = []
        for start in range(0, len(sequences), batch_size):
            batch_seqs = sequences[start:start + batch_size]
            enc = tokenizer(batch_seqs, padding=True, truncation=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())
        fold_probs.append(np.asarray(probs, dtype=np.float64))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(fold_probs, axis=0), axis=0)

class ProtT5ForSequenceClassification(nn.Module):

    def __init__(self, model_path: Path, num_labels: int=1):
        super().__init__()
        self.base_model = T5EncoderModel.from_pretrained(str(model_path))
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['q', 'v', 'k', 'o'])
        self.base_model = get_peft_model(self.base_model, peft_config)
        self.lstm_hidden = 128
        self.bilstm = nn.LSTM(input_size=1024, hidden_size=self.lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(self.lstm_hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Linear(1024 + self.lstm_hidden * 2, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_labels))

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_dtype = self.bilstm.weight_ih_l0.dtype
        last_hidden_state = last_hidden_state.to(dtype=lstm_dtype)
        center_idx = last_hidden_state.shape[1] // 2
        center_repr = last_hidden_state[:, center_idx, :]
        lstm_out, _ = self.bilstm(last_hidden_state)
        attn_weights = self.attention(lstm_out)
        mask = attention_mask.unsqueeze(-1)
        attn_weights = attn_weights.masked_fill(mask == 0, -1000000000.0)
        attn_weights = torch.softmax(attn_weights, dim=1)
        global_repr = torch.sum(lstm_out * attn_weights, dim=1)
        fused_repr = torch.cat([center_repr, global_repr], dim=1)
        return self.classifier(fused_repr)

def load_prott5_fold(fold_dir: Path, base_model_path: Path, device: torch.device):
    tokenizer = T5Tokenizer.from_pretrained(str(base_model_path), do_lower_case=False, legacy=False)
    model = ProtT5ForSequenceClassification(base_model_path)
    state = load_state_dict_from_hf_dir(fold_dir)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'ProtT5 checkpoint mismatch in {fold_dir}\nMissing keys: {missing}\nUnexpected keys: {unexpected}')
    model.to(device).eval()
    return (tokenizer, model)

def clean_prott5_sequence(seq: str) -> str:
    return seq.upper().replace('U', 'X').replace('Z', 'X').replace('O', 'X')

@torch.inference_mode()
def predict_prott5(sequences: list[str], base_model_path: Path, checkpoint_root: Path, device: torch.device, batch_size: int) -> np.ndarray:
    fold_dirs = find_fold_dirs(checkpoint_root, 'ProtT5')
    fold_probs = []
    for fold_dir in fold_dirs:
        tokenizer, model = load_prott5_fold(fold_dir, base_model_path, device)
        probs = []
        for start in range(0, len(sequences), batch_size):
            batch_seqs = [' '.join(list(clean_prott5_sequence(s))) for s in sequences[start:start + batch_size]]
            enc = tokenizer(batch_seqs, max_length=WINDOW_SIZE, padding='max_length', truncation=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.sigmoid(logits.float()).reshape(-1)
            probs.extend(p.cpu().numpy().tolist())
        fold_probs.append(np.asarray(probs, dtype=np.float64))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(fold_probs, axis=0), axis=0)

class AttentionPooling(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bool = mask.bool()
        scores = self.attention(x).squeeze(-1).float()
        scores = scores.masked_fill(~mask_bool, -1000000000.0)
        weights = F.softmax(scores, dim=-1) * mask_bool.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-08)
        return torch.bmm(weights.to(x.dtype).unsqueeze(1), x).squeeze(1)

class RBFExpansion(nn.Module):

    def __init__(self, channels: int=16, d_min: float=0.0, d_max: float=20.0):
        super().__init__()
        self.means = nn.Parameter(torch.linspace(d_min, d_max, channels))
        initial_std = (d_max - d_min) / channels
        self.log_stds = nn.Parameter(torch.full((channels,), math.log(initial_std)))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        distances = distances.unsqueeze(-1)
        stds = self.log_stds.exp().clamp_min(0.0001)
        return torch.exp(-((distances - self.means) / stds) ** 2)

class DistanceAwareGraphAttention(nn.Module):

    def __init__(self, node_dim: int, physical_dim: int=PHYS_DIM, hidden_dim: int=GRAPH_HIDDEN_DIM, rbf_dim: int=RBF_DIM, cutoff: float=GRAPH_CUTOFF_ANGSTROM, noise_std: float=COORD_NOISE_STD_ANGSTROM):
        super().__init__()
        self.cutoff = cutoff
        self.noise_std = noise_std
        self.node_compress = nn.Linear(node_dim + physical_dim, hidden_dim)
        self.rbf = RBFExpansion(channels=rbf_dim, d_min=0.0, d_max=20.0)
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2 + rbf_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.node_update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, node_dim))

    def forward(self, h_seq: torch.Tensor, coords: torch.Tensor, physical_features: torch.Tensor, structure_mask: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = h_seq.shape
        node_input = torch.cat([h_seq, physical_features], dim=-1)
        h_comp = self.node_compress(node_input)
        graph_coords = coords
        if self.training:
            noise = torch.randn_like(graph_coords) * self.noise_std
            graph_coords = graph_coords + noise * structure_mask.unsqueeze(-1)
        displacement = graph_coords.unsqueeze(2) - graph_coords.unsqueeze(1)
        distances = torch.sqrt(torch.sum(displacement.square(), dim=-1) + 1e-06)
        rbf_features = self.rbf(distances)
        h_i = h_comp.unsqueeze(2).expand(batch_size, length, length, -1)
        h_j = h_comp.unsqueeze(1).expand(batch_size, length, length, -1)
        edge_input = torch.cat([h_i, h_j, rbf_features], dim=-1)
        messages = self.edge_mlp(edge_input)
        valid_pair_mask = structure_mask.bool().unsqueeze(2) & structure_mask.bool().unsqueeze(1)
        cutoff_mask = distances < self.cutoff
        self_mask = torch.eye(length, device=h_seq.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = valid_pair_mask & cutoff_mask & ~self_mask
        attention_logits = self.attention_score(messages).squeeze(-1).float()
        attention_logits = attention_logits.masked_fill(~edge_mask, -1000000000.0)
        alpha = F.softmax(attention_logits, dim=2) * edge_mask.float()
        alpha = alpha / alpha.sum(dim=2, keepdim=True).clamp_min(1e-08)
        aggregated = (messages * alpha.to(messages.dtype).unsqueeze(-1)).sum(dim=2)
        delta_h = self.node_update(torch.cat([h_comp, aggregated], dim=-1))
        delta_h = delta_h * structure_mask.unsqueeze(-1)
        return h_seq + delta_h

class SaProtAlignedDAGATClassifier(nn.Module):

    def __init__(self, saprot_model_path: Path):
        super().__init__()
        load_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.saprot = EsmModel.from_pretrained(str(saprot_model_path), torch_dtype=load_dtype)
        lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=128, lora_dropout=0.1, target_modules=['query', 'key', 'value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'])
        self.saprot = get_peft_model(self.saprot, lora_config)
        self.hidden_dim = int(self.saprot.config.hidden_size)
        self.aa_embedding = nn.Embedding(AA_VOCAB_SIZE, AA_EMBED_DIM, padding_idx=AA_UNKNOWN_ID)
        self.ss_embedding = nn.Embedding(SS_VOCAB_SIZE, SS_EMBED_DIM, padding_idx=SS_UNKNOWN_ID)
        self.dagat = DistanceAwareGraphAttention(node_dim=self.hidden_dim, physical_dim=PHYS_DIM)
        self.seq_pooler = AttentionPooling(self.hidden_dim)
        self.geo_pooler = AttentionPooling(self.hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask, coords, aa_ids, ss_ids, rsa_vals, dssp_mask, sequence_mask, structure_mask):
        outputs = self.saprot(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        h_seq = outputs.last_hidden_state
        aa_features = self.aa_embedding(aa_ids)
        ss_features = self.ss_embedding(ss_ids)
        rsa_features = rsa_vals.to(ss_features.dtype) * dssp_mask.unsqueeze(-1).to(ss_features.dtype)
        physical_features = torch.cat([aa_features, ss_features, rsa_features], dim=-1)
        physical_features = physical_features * structure_mask.unsqueeze(-1).to(physical_features.dtype)
        h_geo = self.dagat(h_seq, coords.to(h_seq.dtype), physical_features.to(h_seq.dtype), structure_mask.to(h_seq.dtype))
        k_seq = self.seq_pooler(h_seq, sequence_mask)
        has_structure = structure_mask.sum(dim=1, keepdim=True) > 0
        geo_pool_mask = torch.where(has_structure, structure_mask, sequence_mask)
        k_geo = self.geo_pooler(h_geo, geo_pool_mask)
        k_geo = torch.where(has_structure, k_geo, k_seq)
        fused = torch.cat([k_seq, k_geo], dim=-1)
        return self.classifier(fused)

class PredictionSaProtDataset(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer: EsmTokenizer, coords: np.ndarray, aa_ids: np.ndarray, ss_ids: np.ndarray, rsa_vals: np.ndarray, coordinate_mask: np.ndarray, dssp_mask: np.ndarray, max_len: int=MAX_SAPROT_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.coords = coords
        self.aa_ids = aa_ids
        self.ss_ids = ss_ids
        self.rsa_vals = rsa_vals
        self.coordinate_mask = coordinate_mask
        self.dssp_mask = dssp_mask
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    @staticmethod
    def clean_sequence(value: object) -> str:
        return str(value).strip().upper().replace('U', 'X').replace('Z', 'X').replace('O', 'X')

    def prepare_sequences(self, row: pd.Series):
        sequence = self.clean_sequence(row['Sequence'])
        foldseek = str(row['foldseek_seq']).strip()
        if len(sequence) != WINDOW_SIZE:
            raise ValueError(f'Sequence length is {len(sequence)}, expected {WINDOW_SIZE}')
        if len(foldseek) != WINDOW_SIZE:
            raise ValueError(f'foldseek_seq length is {len(foldseek)}, expected {WINDOW_SIZE}')
        if sequence[CENTER_INDEX] != 'K':
            raise ValueError('Sequence position 26 is not lysine')
        interleaved = ' '.join((aa + structural_token for aa, structural_token in zip(sequence, foldseek)))
        return (sequence, interleaved)

    def tokenize(self, interleaved: str):
        return self.tokenizer(interleaved, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=True, return_special_tokens_mask=True)

    @staticmethod
    def non_special_positions(encoding) -> torch.Tensor:
        attention_mask = encoding['attention_mask'].squeeze(0).bool()
        active_positions = torch.nonzero(attention_mask, as_tuple=False).flatten()
        if active_positions.numel() < 2:
            raise RuntimeError('Tokenizer output lacks CLS/EOS positions')
        return active_positions[1:-1]

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        sequence, interleaved = self.prepare_sequences(row)
        encoding = self.tokenize(interleaved)
        token_positions = self.non_special_positions(encoding)
        if int(token_positions.numel()) != WINDOW_SIZE:
            raise RuntimeError(f'Sample {index}: expected {WINDOW_SIZE} aligned SaProt tokens, got {int(token_positions.numel())}.')
        residue_coords = torch.as_tensor(self.coords[index], dtype=torch.float32)
        residue_aa = torch.as_tensor(self.aa_ids[index], dtype=torch.long).clamp(0, AA_UNKNOWN_ID)
        residue_ss = torch.as_tensor(self.ss_ids[index], dtype=torch.long).clamp(0, SS_UNKNOWN_ID)
        residue_rsa = torch.as_tensor(self.rsa_vals[index], dtype=torch.float32).clamp(0.0, 1.0)
        residue_valid = torch.as_tensor(self.coordinate_mask[index], dtype=torch.float32)
        residue_dssp_valid = torch.as_tensor(self.dssp_mask[index], dtype=torch.float32)
        residue_ss = torch.where(residue_dssp_valid.bool(), residue_ss, torch.full_like(residue_ss, SS_UNKNOWN_ID))
        coords = torch.zeros((self.max_len, 3), dtype=torch.float32)
        aa_ids = torch.full((self.max_len,), AA_UNKNOWN_ID, dtype=torch.long)
        ss_ids = torch.full((self.max_len,), SS_UNKNOWN_ID, dtype=torch.long)
        rsa_vals = torch.zeros((self.max_len, 1), dtype=torch.float32)
        dssp_mask = torch.zeros(self.max_len, dtype=torch.float32)
        sequence_mask = torch.zeros(self.max_len, dtype=torch.float32)
        structure_mask = torch.zeros(self.max_len, dtype=torch.float32)
        coords[token_positions] = residue_coords
        aa_ids[token_positions] = residue_aa
        ss_ids[token_positions] = residue_ss
        rsa_vals[token_positions] = residue_rsa
        dssp_mask[token_positions] = residue_dssp_valid
        sequence_mask[token_positions] = 1.0
        structure_mask[token_positions] = residue_valid
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'coords': coords, 'aa_ids': aa_ids, 'ss_ids': ss_ids, 'rsa_vals': rsa_vals, 'dssp_mask': dssp_mask, 'sequence_mask': sequence_mask, 'structure_mask': structure_mask}

def load_prediction_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError('Prediction table must be .xlsx, .xls, or .csv')
    aliases = {'UniProt_ID': 'UniProt ID', 'uniprot_id': 'UniProt ID', 'UniProtID': 'UniProt ID', 'position': 'Position', 'sequence': 'Sequence', 'Foldseek_seq': 'foldseek_seq', 'Foldseek_Seq': 'foldseek_seq'}
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    required = {'UniProt ID', 'Position', 'Sequence', 'foldseek_seq'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Prediction table missing required columns: {sorted(missing)}')
    df = df.copy().reset_index(drop=True)
    df['UniProt ID'] = df['UniProt ID'].astype(str).str.strip()
    df['Position'] = pd.to_numeric(df['Position'], errors='raise').astype(int)
    df['Sequence'] = df['Sequence'].astype(str).str.strip().str.upper()
    lengths = df['Sequence'].str.len()
    if not (lengths == WINDOW_SIZE).all():
        bad = np.flatnonzero((lengths != WINDOW_SIZE).to_numpy())[:5]
        raise ValueError(f'All sequences must have length 51. Bad row indices: {bad.tolist()}')
    centers = df['Sequence'].str[CENTER_INDEX]
    if not (centers == 'K').all():
        bad = np.flatnonzero((centers != 'K').to_numpy())[:5]
        raise ValueError(f'All sequences must be lysine-centered at position 26. Bad row indices: {bad.tolist()}')
    foldseek_clean = df['foldseek_seq'].astype(str).str.strip()
    fallback = foldseek_clean.str.len() != WINDOW_SIZE
    df['_foldseek_fallback'] = fallback.astype(int)
    df['foldseek_seq'] = foldseek_clean
    df.loc[fallback, 'foldseek_seq'] = '#' * WINDOW_SIZE
    return df

def load_prediction_npz(path: Path, df: pd.DataFrame) -> dict[str, np.ndarray]:
    required_keys = {'coords', 'aa_ids', 'ss_ids', 'rsa_vals', 'valid_mask', 'dssp_mask', 'residue_ids', 'uniprot_ids', 'target_positions', 'center_index', 'window_size'}
    with np.load(path, allow_pickle=False) as data:
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f'Structural NPZ missing keys: {sorted(missing)}')
        arrays = {k: data[k] for k in required_keys}
    n = len(df)
    expected_shapes = {'coords': (n, WINDOW_SIZE, 3), 'aa_ids': (n, WINDOW_SIZE), 'ss_ids': (n, WINDOW_SIZE), 'rsa_vals': (n, WINDOW_SIZE, 1), 'valid_mask': (n, WINDOW_SIZE), 'dssp_mask': (n, WINDOW_SIZE), 'residue_ids': (n, WINDOW_SIZE), 'uniprot_ids': (n,), 'target_positions': (n,)}
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f'NPZ {key} shape={arrays[key].shape}, expected={expected}')
    if int(arrays['center_index']) != CENTER_INDEX:
        raise ValueError(f"NPZ center_index={int(arrays['center_index'])}; expected 25")
    if int(arrays['window_size']) != WINDOW_SIZE:
        raise ValueError(f"NPZ window_size={int(arrays['window_size'])}; expected 51")
    excel_uids = df['UniProt ID'].astype(str).str.strip().str.upper().to_numpy()
    npz_uids = np.char.upper(np.char.strip(arrays['uniprot_ids'].astype(str)))
    excel_pos = df['Position'].astype(np.int32).to_numpy()
    npz_pos = arrays['target_positions'].astype(np.int32)
    aligned = (excel_uids == npz_uids) & (excel_pos == npz_pos)
    if not aligned.all():
        first = int(np.flatnonzero(~aligned)[0])
        raise ValueError(f'Prediction table and structural NPZ are not in identical order. First mismatch row {first}: table={excel_uids[first]}|{excel_pos[first]}, NPZ={npz_uids[first]}|{npz_pos[first]}')
    if not np.isfinite(arrays['coords']).all():
        raise ValueError('coords contains NaN/Inf')
    if not np.isfinite(arrays['rsa_vals']).all():
        raise ValueError('rsa_vals contains NaN/Inf')
    return arrays

def find_saprot_checkpoints(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'Checkpoint root does not exist: {root}')
    paths = [root / f'SaProt_branch_fold_{fold}.pth' for fold in range(1, 6)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing SaProt fold checkpoints:\n  - ' + '\n  - '.join(missing))
    return paths

@torch.inference_mode()
def predict_saprot(df: pd.DataFrame, arrays: dict[str, np.ndarray], base_model_path: Path, checkpoint_root: Path, device: torch.device, batch_size: int, num_workers: int) -> np.ndarray:
    tokenizer = EsmTokenizer.from_pretrained(str(base_model_path))
    dataset = PredictionSaProtDataset(df=df, tokenizer=tokenizer, coords=arrays['coords'], aa_ids=arrays['aa_ids'], ss_ids=arrays['ss_ids'], rsa_vals=arrays['rsa_vals'], coordinate_mask=arrays['valid_mask'], dssp_mask=arrays['dssp_mask'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    checkpoint_paths = find_saprot_checkpoints(checkpoint_root)
    fold_probs = []
    for ckpt in checkpoint_paths:
        model = SaProtAlignedDAGATClassifier(base_model_path).to(device)
        state = torch.load(ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_unexpected = [key for key in unexpected if key.startswith('saprot.') and key.endswith('embeddings.position_embeddings.weight')]
        real_unexpected = [key for key in unexpected if key not in allowed_unexpected]
        if missing or real_unexpected:
            raise RuntimeError(f'SaProt checkpoint mismatch in {ckpt}\nMissing keys: {missing}\nUnexpected keys: {real_unexpected}\nAll unexpected keys before filtering: {unexpected}')
        model.eval()
        probs = []
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast_context(device):
                logits = model(**batch)
            p = torch.sigmoid(logits.float()).reshape(-1)
            probs.extend(p.cpu().numpy().tolist())
        fold_probs.append(np.asarray(probs, dtype=np.float64))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return np.mean(np.stack(fold_probs, axis=0), axis=0)

def binary_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0 - EPS)
    entropy = -(p * np.log(p + 1e-09) + (1.0 - p) * np.log(1.0 - p + 1e-09))
    return entropy / np.log(2.0)

def get_fusion_parameters() -> tuple[np.ndarray, float]:
    raw_theta = (THETA_ESM2, THETA_PROTT5, THETA_SAPROT)
    if any(value is None for value in raw_theta) or LAMBDA is None:
        raise ValueError('Set THETA_ESM2, THETA_PROTT5, THETA_SAPROT, and LAMBDA before inference.')
    fusion_theta = np.asarray(raw_theta, dtype=np.float64)
    lambda_value = float(LAMBDA)
    if not np.isfinite(fusion_theta).all() or not np.isfinite(lambda_value):
        raise ValueError('Fusion parameters must be finite numeric values.')
    if np.any(fusion_theta <= 0.0) or fusion_theta.sum() <= 0.0:
        raise ValueError('Theta values must be positive.')
    fusion_theta = fusion_theta / fusion_theta.sum()
    return (fusion_theta, lambda_value)

def apply_dynamic_fusion(s_esm: np.ndarray, s_t5: np.ndarray, s_saprot: np.ndarray):
    fusion_theta, lambda_value = get_fusion_parameters()
    probs = np.vstack([np.clip(s_esm, EPS, 1.0 - EPS), np.clip(s_t5, EPS, 1.0 - EPS), np.clip(s_saprot, EPS, 1.0 - EPS)])
    logits = logit(probs)
    uncertainty = np.vstack([binary_entropy(p) for p in probs])
    q = fusion_theta[:, None] * np.exp(-lambda_value * uncertainty)
    weights = q / np.clip(q.sum(axis=0, keepdims=True), 1e-12, None)
    final_logit = np.sum(weights * logits, axis=0)
    final_prob = expit(final_logit)
    return (final_prob, uncertainty, weights)

def report_metrics(labels: np.ndarray, probs: np.ndarray) -> None:
    preds = (probs >= 0.5).astype(int)
    print('Evaluation metrics')
    if len(np.unique(labels)) == 2:
        print(f'AUC : {roc_auc_score(labels, probs):.5f}')
        print(f'AP  : {average_precision_score(labels, probs):.5f}')
        print(f'MCC : {matthews_corrcoef(labels, preds):.5f}')
    print(f'ACC : {accuracy_score(labels, preds):.5f}')

def validate_required_paths() -> None:
    required_paths = {'Prediction table': PREDICT_DATA_PATH, 'Structural NPZ': STRUCTURE_NPZ_PATH, 'ESM-2 base model': ESM2_MODEL_PATH, 'ESM-2 checkpoint directory': ESM2_CHECKPOINT_DIR, 'ProtT5 base model': PROTT5_MODEL_PATH, 'ProtT5 checkpoint directory': PROTT5_CHECKPOINT_DIR, 'SaProt base model': SAPROT_MODEL_PATH, 'SaProt checkpoint directory': SAPROT_CHECKPOINT_DIR}
    missing = [f'{name}: {path}' for name, path in required_paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError('\nRequired prediction path(s) not found:\n  - ' + '\n  - '.join(missing))
    find_fold_dirs(ESM2_CHECKPOINT_DIR, 'ESM2')
    find_fold_dirs(PROTT5_CHECKPOINT_DIR, 'ProtT5')
    find_saprot_checkpoints(SAPROT_CHECKPOINT_DIR)

def configure_paths() -> None:
    names = ('PREDICT_DATA_PATH', 'STRUCTURE_NPZ_PATH', 'OUTPUT_PATH', 'ESM2_MODEL_PATH', 'PROTT5_MODEL_PATH', 'SAPROT_MODEL_PATH', 'ESM2_CHECKPOINT_DIR', 'PROTT5_CHECKPOINT_DIR', 'SAPROT_CHECKPOINT_DIR',)
    missing = [name for name in names if not str(globals()[name]).strip()]
    if missing:
        raise ValueError("Set the following paths before running: " + ", ".join(missing))
    for name in names:
        globals()[name] = Path(globals()[name]).expanduser()

def main() -> None:
    configure_paths()
    set_seed(SEED)
    device = pick_device(DEVICE_NAME)
    get_fusion_parameters()
    validate_required_paths()
    df = load_prediction_table(PREDICT_DATA_PATH)
    arrays = load_prediction_npz(STRUCTURE_NPZ_PATH, df)
    sequences = df['Sequence'].astype(str).tolist()
    print(f'Running inference on {len(df)} samples using {device}.')
    print('Running ESM-2 inference.')
    esm_prob = predict_esm2(sequences=sequences, base_model_path=ESM2_MODEL_PATH, checkpoint_root=ESM2_CHECKPOINT_DIR, device=device, batch_size=ESM2_BATCH_SIZE)
    print('Running ProtT5 inference.')
    t5_prob = predict_prott5(sequences=sequences, base_model_path=PROTT5_MODEL_PATH, checkpoint_root=PROTT5_CHECKPOINT_DIR, device=device, batch_size=PROTT5_BATCH_SIZE)
    print('Running SaProt inference.')
    saprot_prob = predict_saprot(df=df, arrays=arrays, base_model_path=SAPROT_MODEL_PATH, checkpoint_root=SAPROT_CHECKPOINT_DIR, device=device, batch_size=SAPROT_BATCH_SIZE, num_workers=NUM_WORKERS)
    final_prob, entropy, weights = apply_dynamic_fusion(esm_prob, t5_prob, saprot_prob)
    output = pd.DataFrame({'UniProt_ID': df['UniProt ID'].astype(str), 'Position': df['Position'].astype(int), 'Sequence': df['Sequence'].astype(str), 'ESM2_Prob': esm_prob, 'ProtT5_Prob': t5_prob, 'SaProt_Prob': saprot_prob, 'ESM2_Entropy': entropy[0], 'ProtT5_Entropy': entropy[1], 'SaProt_Entropy': entropy[2], 'ESM2_Weight': weights[0], 'ProtT5_Weight': weights[1], 'SaProt_Weight': weights[2], 'Final_Prob': final_prob, 'Prediction': (final_prob >= 0.5).astype(int), 'foldseek_fallback': df['_foldseek_fallback'].astype(int), 'valid_structure_residues': arrays['valid_mask'].sum(axis=1).astype(int)})
    label_col = next((col for col in ['Label', 'label'] if col in df.columns), None)
    if label_col is not None:
        labels = pd.to_numeric(df[label_col], errors='coerce')
        if labels.notna().all() and labels.isin([0, 1]).all():
            labels_np = labels.astype(int).to_numpy()
            output.insert(3, 'Label', labels_np)
            report_metrics(labels_np, final_prob)
        else:
            print('Warning: Label column is not a complete binary vector; metrics were skipped.')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f'Predictions saved to: {OUTPUT_PATH}')
if __name__ == '__main__':
    main()
