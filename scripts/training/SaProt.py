from __future__ import annotations
import contextlib
import math
import os
import random
from pathlib import Path
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer, get_linear_schedule_with_warmup
DATA_PATH = ""  # Training Excel file.
SAPROT_MODEL_PATH = ""  # Local SaProt model directory.
STRUCTURE_NPZ_PATH = ""  # Aligned structural feature NPZ file.
SAVE_DIR = Path(__file__).resolve().parents[3] / "trained_models" / "Saprot_DAGAT"  # Five fold checkpoints; independent of the working directory.
OOF_OUTPUT_PATH = ""  # Output CSV file for out-of-fold predictions.

MAX_LEN = 64
WINDOW_SIZE = 51
CENTER_INDEX = 25
AA_VOCAB_SIZE = 21
AA_UNKNOWN_ID = 20
AA_EMBED_DIM = 16
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_ID = {aa: index for index, aa in enumerate(AA_ALPHABET)}
SS_VOCAB_SIZE = 9
SS_UNKNOWN_ID = 8
SS_EMBED_DIM = 8
PHYS_DIM = AA_EMBED_DIM + SS_EMBED_DIM + 1
GRAPH_HIDDEN_DIM = 256
RBF_DIM = 16
GRAPH_CUTOFF_ANGSTROM = 10.0
COORD_NOISE_STD_ANGSTROM = 0.1
BATCH_SIZE = 16
ACCUM_STEPS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.15
NUM_EPOCHS = 25
EARLY_STOP_PATIENCE = 8
NUM_FOLDS = 5
NUM_WORKERS = 4
SEED = 42
CUDA_AVAILABLE = torch.cuda.is_available()
USE_BFLOAT16 = CUDA_AVAILABLE and torch.cuda.is_bf16_supported()
if CUDA_AVAILABLE:
    torch.cuda.set_device(0)
DEVICE = torch.device('cuda:0' if CUDA_AVAILABLE else 'cpu')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA_AVAILABLE:
        torch.cuda.manual_seed_all(seed)

def autocast_context():
    if USE_BFLOAT16:
        return torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()

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
        expected_physical_dim = self.node_compress.in_features - h_seq.shape[-1]
        if physical_features.shape != (batch_size, length, expected_physical_dim):
            raise ValueError(f'physical_features shape mismatch: expected {(batch_size, length, expected_physical_dim)}, got {tuple(physical_features.shape)}')
        if coords.shape != (batch_size, length, 3):
            raise ValueError(f'coords must be [B,L,3], got {tuple(coords.shape)}')
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
        load_dtype = torch.bfloat16 if USE_BFLOAT16 else torch.float32
        self.saprot = EsmModel.from_pretrained(str(saprot_model_path), torch_dtype=load_dtype)
        self.saprot.gradient_checkpointing_enable()
        lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=64, lora_alpha=128, lora_dropout=0.1, target_modules=['query', 'key', 'value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'])
        self.saprot = get_peft_model(self.saprot, lora_config)
        if hasattr(self.saprot, 'enable_input_require_grads'):
            self.saprot.enable_input_require_grads()
        self.hidden_dim = int(self.saprot.config.hidden_size)
        self.aa_embedding = nn.Embedding(AA_VOCAB_SIZE, AA_EMBED_DIM, padding_idx=AA_UNKNOWN_ID)
        self.ss_embedding = nn.Embedding(SS_VOCAB_SIZE, SS_EMBED_DIM, padding_idx=SS_UNKNOWN_ID)
        self.dagat = DistanceAwareGraphAttention(node_dim=self.hidden_dim, physical_dim=PHYS_DIM)
        self.seq_pooler = AttentionPooling(self.hidden_dim)
        self.geo_pooler = AttentionPooling(self.hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_dim * 2, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
        self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, coords: torch.Tensor, aa_ids: torch.Tensor, ss_ids: torch.Tensor, rsa_vals: torch.Tensor, dssp_mask: torch.Tensor, sequence_mask: torch.Tensor, structure_mask: torch.Tensor, labels: torch.Tensor | None=None):
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
        logits = self.classifier(fused)
        if labels is None:
            return logits
        loss = self.loss_function(logits.view(-1).float(), labels.view(-1).float())
        return (loss, logits)

class AlignedSaProtDataset(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer: EsmTokenizer, coords: np.ndarray, aa_ids: np.ndarray, ss_ids: np.ndarray, rsa_vals: np.ndarray, coordinate_mask: np.ndarray, dssp_mask: np.ndarray, max_len: int=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.coords = coords
        self.aa_ids = aa_ids
        self.ss_ids = ss_ids
        self.rsa_vals = rsa_vals
        self.coordinate_mask = coordinate_mask
        self.dssp_mask = dssp_mask
        self.max_len = max_len
        sample_count = len(self.df)
        expected_shapes = {'coords': (sample_count, WINDOW_SIZE, 3), 'aa_ids': (sample_count, WINDOW_SIZE), 'ss_ids': (sample_count, WINDOW_SIZE), 'rsa_vals': (sample_count, WINDOW_SIZE, 1), 'coordinate_mask': (sample_count, WINDOW_SIZE), 'dssp_mask': (sample_count, WINDOW_SIZE)}
        actual_shapes = {'coords': self.coords.shape, 'aa_ids': self.aa_ids.shape, 'ss_ids': self.ss_ids.shape, 'rsa_vals': self.rsa_vals.shape, 'coordinate_mask': self.coordinate_mask.shape, 'dssp_mask': self.dssp_mask.shape}
        for name, expected in expected_shapes.items():
            if actual_shapes[name] != expected:
                raise ValueError(f'{name} shape mismatch: expected {expected}, got {actual_shapes[name]}')
        self._validate_tokenizer_alignment()

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _clean_sequence(value: object) -> str:
        return str(value).strip().upper().replace('U', 'X').replace('Z', 'X').replace('O', 'X')

    def _prepare_sequences(self, row: pd.Series) -> tuple[str, str, str]:
        sequence = self._clean_sequence(row['Sequence'])
        foldseek = str(row['foldseek_seq']).strip()
        if len(sequence) != WINDOW_SIZE:
            raise ValueError(f'Sequence length is {len(sequence)}, expected {WINDOW_SIZE}')
        if len(foldseek) != WINDOW_SIZE:
            raise ValueError(f'foldseek_seq length is {len(foldseek)}, expected {WINDOW_SIZE}')
        if sequence[CENTER_INDEX] != 'K':
            raise ValueError('Sequence position 26 is not lysine')
        interleaved = ' '.join((aa + structural_token for aa, structural_token in zip(sequence, foldseek)))
        return (sequence, foldseek, interleaved)

    def _tokenize(self, interleaved: str):
        return self.tokenizer(interleaved, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt', add_special_tokens=True, return_special_tokens_mask=True)

    @staticmethod
    def _non_special_positions(encoding) -> torch.Tensor:
        attention_mask = encoding['attention_mask'].squeeze(0).bool()
        active_positions = torch.nonzero(attention_mask, as_tuple=False).flatten()
        if active_positions.numel() < 2:
            raise RuntimeError('Tokenizer output does not contain CLS/EOS positions')
        return active_positions[1:-1]

    def _validate_tokenizer_alignment(self) -> None:
        for _, row in self.df.iterrows():
            sequence, _, interleaved = self._prepare_sequences(row)
            if not sequence:
                continue
            positions = self._non_special_positions(self._tokenize(interleaved))
            token_count = int(positions.numel())
            if token_count == len(sequence):
                return
            raise RuntimeError(f'SaProt must produce one compound AA-3Di token per residue. Residues={len(sequence)}, non-special tokens={token_count}. Do not repeat coordinates to force alignment; inspect the tokenizer input.')
        raise RuntimeError('No non-empty sequence was found in the dataframe.')

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        sequence, _, interleaved = self._prepare_sequences(row)
        residue_count = len(sequence)
        if residue_count == 0:
            raise ValueError(f'Sample {index} has an empty sequence.')
        encoding = self._tokenize(interleaved)
        token_positions = self._non_special_positions(encoding)
        expected_token_count = residue_count
        if int(token_positions.numel()) != expected_token_count:
            raise RuntimeError(f'Sample {index}: expected {expected_token_count} aligned tokens, got {int(token_positions.numel())}.')
        residue_coords = torch.as_tensor(self.coords[index, :residue_count], dtype=torch.float32)
        residue_aa = torch.as_tensor(self.aa_ids[index, :residue_count], dtype=torch.long).clamp(0, AA_UNKNOWN_ID)
        residue_ss = torch.as_tensor(self.ss_ids[index, :residue_count], dtype=torch.long).clamp(0, SS_UNKNOWN_ID)
        residue_rsa = torch.as_tensor(self.rsa_vals[index, :residue_count], dtype=torch.float32).clamp(0.0, 1.0)
        residue_valid = torch.as_tensor(self.coordinate_mask[index, :residue_count], dtype=torch.float32)
        residue_dssp_valid = torch.as_tensor(self.dssp_mask[index, :residue_count], dtype=torch.float32)
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
        return {'input_ids': encoding['input_ids'].squeeze(0), 'attention_mask': encoding['attention_mask'].squeeze(0), 'coords': coords, 'aa_ids': aa_ids, 'ss_ids': ss_ids, 'rsa_vals': rsa_vals, 'dssp_mask': dssp_mask, 'sequence_mask': sequence_mask, 'structure_mask': structure_mask, 'label': torch.tensor(float(row['Label']), dtype=torch.float32)}

def move_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.to(DEVICE, non_blocking=True) for key, value in batch.items()}

def train_one_epoch(model, dataloader, optimizer, scheduler, epoch_index: int) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    batch_count = len(dataloader)
    remainder = batch_count % ACCUM_STEPS
    progress = tqdm(dataloader, desc=f'Epoch {epoch_index + 1} train', leave=False)
    for step, raw_batch in enumerate(progress):
        batch = move_batch(raw_batch)
        final_partial_group = remainder > 0 and step >= batch_count - remainder
        divisor = remainder if final_partial_group else ACCUM_STEPS
        with autocast_context():
            loss, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], coords=batch['coords'], aa_ids=batch['aa_ids'], ss_ids=batch['ss_ids'], rsa_vals=batch['rsa_vals'], dssp_mask=batch['dssp_mask'], sequence_mask=batch['sequence_mask'], structure_mask=batch['structure_mask'], labels=batch['label'])
            scaled_loss = loss / divisor
        scaled_loss.backward()
        should_step = (step + 1) % ACCUM_STEPS == 0 or step + 1 == len(dataloader)
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        progress.set_postfix(loss=f'{loss_value:.4f}')
    return total_loss / max(len(dataloader), 1)

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    for raw_batch in dataloader:
        batch = move_batch(raw_batch)
        with autocast_context():
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], coords=batch['coords'], aa_ids=batch['aa_ids'], ss_ids=batch['ss_ids'], rsa_vals=batch['rsa_vals'], dssp_mask=batch['dssp_mask'], sequence_mask=batch['sequence_mask'], structure_mask=batch['structure_mask'])
        probabilities.extend(torch.sigmoid(logits).float().cpu().numpy().reshape(-1).tolist())
        labels.extend(batch['label'].cpu().numpy().astype(int).reshape(-1).tolist())
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    predictions = (probabilities_array >= 0.5).astype(np.int64)
    accuracy = accuracy_score(labels_array, predictions)
    mcc = matthews_corrcoef(labels_array, predictions)
    auc = roc_auc_score(labels_array, probabilities_array)
    ap = average_precision_score(labels_array, probabilities_array)
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {'accuracy': accuracy, 'auc': auc, 'mcc': mcc, 'ap': ap, 'sensitivity': sensitivity, 'specificity': specificity, 'probabilities': probabilities_array, 'labels': labels_array}

def load_aligned_features(path: Path, dataframe: pd.DataFrame):
    required_keys = {'coords', 'aa_ids', 'ss_ids', 'rsa_vals', 'valid_mask', 'dssp_mask', 'residue_ids', 'uniprot_ids', 'target_positions', 'center_index', 'window_size'}
    with np.load(path, allow_pickle=False) as data:
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(f'{path} is missing NPZ keys: {sorted(missing)}')
        arrays = {key: data[key] for key in required_keys}
    sample_count = len(dataframe)
    expected_shapes = {'coords': (sample_count, WINDOW_SIZE, 3), 'aa_ids': (sample_count, WINDOW_SIZE), 'ss_ids': (sample_count, WINDOW_SIZE), 'rsa_vals': (sample_count, WINDOW_SIZE, 1), 'valid_mask': (sample_count, WINDOW_SIZE), 'dssp_mask': (sample_count, WINDOW_SIZE), 'residue_ids': (sample_count, WINDOW_SIZE), 'uniprot_ids': (sample_count,), 'target_positions': (sample_count,)}
    for key, expected_shape in expected_shapes.items():
        if arrays[key].shape != expected_shape:
            raise ValueError(f'NPZ {key} shape is {arrays[key].shape}, expected {expected_shape}. The Excel and NPZ must have identical filtered rows and order.')
    if int(arrays['center_index']) != CENTER_INDEX:
        raise ValueError(f"NPZ center_index={int(arrays['center_index'])}, expected {CENTER_INDEX}")
    if int(arrays['window_size']) != WINDOW_SIZE:
        raise ValueError(f"NPZ window_size={int(arrays['window_size'])}, expected {WINDOW_SIZE}")
    excel_uids = dataframe['UniProt ID'].astype(str).str.strip().str.upper().to_numpy()
    npz_uids = np.char.upper(np.char.strip(arrays['uniprot_ids'].astype(str)))
    excel_positions = pd.to_numeric(dataframe['Position'], errors='raise').astype(np.int32).to_numpy()
    npz_positions = arrays['target_positions'].astype(np.int32)
    aligned = (excel_uids == npz_uids) & (excel_positions == npz_positions)
    if not aligned.all():
        first = int(np.flatnonzero(~aligned)[0])
        raise ValueError(f'Excel and NPZ sample order mismatch at row {first}: Excel={excel_uids[first]}|{excel_positions[first]}, NPZ={npz_uids[first]}|{npz_positions[first]}')
    if not np.isfinite(arrays['coords']).all():
        raise ValueError('coords contains NaN or infinity')
    if not np.isfinite(arrays['rsa_vals']).all():
        raise ValueError('rsa_vals contains NaN or infinity')
    if not np.isin(arrays['valid_mask'], [0.0, 1.0]).all():
        raise ValueError('valid_mask must contain only 0 and 1')
    if not np.isin(arrays['dssp_mask'], [0.0, 1.0]).all():
        raise ValueError('dssp_mask must contain only 0 and 1')
    if np.any(arrays['dssp_mask'] > arrays['valid_mask']):
        raise ValueError('dssp_mask contains nodes without valid coordinates')
    if arrays['aa_ids'].min() < 0 or arrays['aa_ids'].max() >= AA_VOCAB_SIZE:
        raise ValueError(f'aa_ids must be in [0, {AA_VOCAB_SIZE - 1}]')
    if arrays['ss_ids'].min() < 0 or arrays['ss_ids'].max() >= SS_VOCAB_SIZE:
        raise ValueError(f'ss_ids must be in [0, {SS_VOCAB_SIZE - 1}]')
    if arrays['rsa_vals'].min() < 0.0 or arrays['rsa_vals'].max() > 1.0:
        raise ValueError('rsa_vals must be relative accessibility in [0, 1]')
    stored_centers = arrays['residue_ids'][:, CENTER_INDEX].astype(np.int32)
    center_valid = arrays['valid_mask'][:, CENTER_INDEX].astype(bool)
    bad_centers = center_valid & (stored_centers != excel_positions)
    if bad_centers.any():
        first = int(np.flatnonzero(bad_centers)[0])
        raise ValueError(f'center residue mismatch at row {first}: Excel={excel_positions[first]}, NPZ={stored_centers[first]}')
    expected_aa = np.full((sample_count, WINDOW_SIZE), AA_UNKNOWN_ID, dtype=np.int64)
    for row_index, sequence in enumerate(dataframe['Sequence']):
        cleaned = AlignedSaProtDataset._clean_sequence(sequence)
        if len(cleaned) != WINDOW_SIZE:
            raise ValueError(f'row {row_index}: Sequence length is {len(cleaned)}, expected {WINDOW_SIZE}')
        expected_aa[row_index] = [AA_TO_ID.get(aa, AA_UNKNOWN_ID) for aa in cleaned]
    comparable = arrays['valid_mask'].astype(bool)
    aa_mismatch = comparable & (arrays['aa_ids'] != expected_aa)
    mismatch_count = int(aa_mismatch.sum())
    comparable_count = int(comparable.sum())
    mismatch_ratio = mismatch_count / max(comparable_count, 1)
    if mismatch_ratio > 0.01:
        raise ValueError(f'AA/structure alignment mismatch is {mismatch_ratio:.2%} ({mismatch_count}/{comparable_count} valid nodes), exceeding 1%.')
    return arrays

def configure_paths() -> None:
    names = ('DATA_PATH', 'SAPROT_MODEL_PATH', 'STRUCTURE_NPZ_PATH', 'SAVE_DIR', 'OOF_OUTPUT_PATH',)
    missing = [name for name in names if not str(globals()[name]).strip()]
    if missing:
        raise ValueError("Set the following paths before running: " + ", ".join(missing))
    for name in names:
        globals()[name] = Path(globals()[name]).expanduser()

def main() -> None:
    configure_paths()
    set_seed(SEED)
    required_paths = {'training dataset': DATA_PATH, 'SaProt model directory': SAPROT_MODEL_PATH, 'aligned structural feature NPZ': STRUCTURE_NPZ_PATH}
    missing_paths = [f'{name}: {path}' for name, path in required_paths.items() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError('Required input path(s) not found:\n  - ' + '\n  - '.join(missing_paths))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OOF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
    tokenizer = EsmTokenizer.from_pretrained(str(SAPROT_MODEL_PATH))
    dataframe_raw = pd.read_excel(DATA_PATH)
    required_columns = {'UniProt ID', 'Position', 'Sequence', 'foldseek_seq', 'Label'}
    missing_columns = sorted(required_columns - set(dataframe_raw.columns))
    if missing_columns:
        raise ValueError(f'Training Excel is missing columns: {missing_columns}')
    numeric_labels = pd.to_numeric(dataframe_raw['Label'], errors='coerce')
    keep_rows = numeric_labels.isin([0, 1])
    dataframe = dataframe_raw.loc[keep_rows].copy().reset_index(drop=True)
    dataframe['Label'] = pd.to_numeric(dataframe['Label']).astype(int)
    if dataframe[['UniProt ID', 'Position', 'Sequence', 'foldseek_seq']].isna().any().any():
        missing_counts = dataframe[['UniProt ID', 'Position', 'Sequence', 'foldseek_seq']].isna().sum()
        raise ValueError(f'Required training columns contain missing values:\n{missing_counts[missing_counts > 0]}')
    if sorted(dataframe['Label'].unique().tolist()) != [0, 1]:
        raise ValueError('Filtered training data must contain both labels 0 and 1')
    foldseek_clean = dataframe['foldseek_seq'].astype(str).str.strip()
    foldseek_fallback = foldseek_clean.str.len() != WINDOW_SIZE
    dataframe['_foldseek_fallback'] = foldseek_fallback.to_numpy()
    dataframe['foldseek_seq'] = foldseek_clean
    dataframe.loc[foldseek_fallback, 'foldseek_seq'] = '#' * WINDOW_SIZE
    labels = dataframe['Label'].to_numpy()
    features = load_aligned_features(STRUCTURE_NPZ_PATH, dataframe)
    probe_indices = list(range(min(len(dataframe), 8)))
    fallback_indices = np.flatnonzero(dataframe['_foldseek_fallback'].to_numpy())
    if fallback_indices.size:
        probe_indices.append(int(fallback_indices[0]))
    probe_indices = np.asarray(sorted(set(probe_indices)), dtype=np.int64)
    probe_dataset = AlignedSaProtDataset(dataframe.iloc[probe_indices], tokenizer, features['coords'][probe_indices], features['aa_ids'][probe_indices], features['ss_ids'][probe_indices], features['rsa_vals'][probe_indices], features['valid_mask'][probe_indices], features['dssp_mask'][probe_indices])
    _ = [probe_dataset[i] for i in range(len(probe_dataset))]
    splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    out_of_fold_probabilities = np.zeros(len(dataframe), dtype=np.float64)
    out_of_fold_assignments = np.zeros(len(dataframe), dtype=np.int64)
    best_fold_aucs: list[float] = []
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(dataframe['Sequence'], labels), start=1):
        print(f'Fold {fold}/{NUM_FOLDS}')
        train_dataset = AlignedSaProtDataset(dataframe.iloc[train_indices], tokenizer, features['coords'][train_indices], features['aa_ids'][train_indices], features['ss_ids'][train_indices], features['rsa_vals'][train_indices], features['valid_mask'][train_indices], features['dssp_mask'][train_indices])
        validation_dataset = AlignedSaProtDataset(dataframe.iloc[validation_indices], tokenizer, features['coords'][validation_indices], features['aa_ids'][validation_indices], features['ss_ids'][validation_indices], features['rsa_vals'][validation_indices], features['valid_mask'][validation_indices], features['dssp_mask'][validation_indices])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=CUDA_AVAILABLE)
        validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=CUDA_AVAILABLE)
        model = SaProtAlignedDAGATClassifier(SAPROT_MODEL_PATH).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        updates_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
        total_updates = updates_per_epoch * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_updates * 0.1), num_training_steps=total_updates)
        best_auc = -np.inf
        patience_counter = 0
        checkpoint_path = SAVE_DIR / f'SaProt_branch_fold_{fold}.pth'
        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch)
            metrics = evaluate(model, validation_loader)
            print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | loss={train_loss:.4f} | AUC={metrics['auc']:.4f} | AP={metrics['ap']:.4f} | ACC={metrics['accuracy']:.4f} | MCC={metrics['mcc']:.4f} | SEN={metrics['sensitivity']:.4f} | SPEC={metrics['specificity']:.4f}")
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print('Early stopping.')
                    break
        best_fold_aucs.append(float(best_auc))
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        validation_metrics = evaluate(model, validation_loader)
        out_of_fold_probabilities[validation_indices] = validation_metrics['probabilities']
        out_of_fold_assignments[validation_indices] = fold
        print(f'Fold {fold} best AUC: {best_auc:.4f}')
        del model, optimizer, scheduler, train_loader, validation_loader
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
    output_path = OOF_OUTPUT_PATH
    if (out_of_fold_assignments == 0).any():
        raise RuntimeError('OOF predictions are incomplete')
    pd.DataFrame({'UniProt ID': dataframe['UniProt ID'].astype(str), 'Position': dataframe['Position'].astype(int), 'label': labels, 'score': out_of_fold_probabilities, 'fold': out_of_fold_assignments, 'foldseek_fallback': dataframe['_foldseek_fallback'].astype(int)}).to_csv(output_path, index=False)
    print(f'Mean best-fold AUC: {np.mean(best_fold_aucs):.4f}')
    print(f'OOF predictions saved to: {output_path}')
if __name__ == '__main__':
    main()
