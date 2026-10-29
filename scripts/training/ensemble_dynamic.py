import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from scipy.special import expit, logit
from scipy.optimize import minimize
import json
from pathlib import Path
FILE_ESM = ""  # ESM-2 out-of-fold prediction CSV file.
FILE_T5 = ""  # ProtT5 out-of-fold prediction CSV file.
FILE_SAPROT = ""  # SaProt out-of-fold prediction CSV file.
FUSION_OUTPUT = ""  # Output JSON file for fitted fusion parameters.
PREDICTION_OUTPUT = ""  # Output CSV file for fused out-of-fold predictions.

# Naming metadata only: fusion fitting reads OOF CSVs, not model weights.
BRANCH_CHECKPOINTS = {
    'ESM2': [f'ESM2_branch_fold_{fold}' for fold in range(1, 6)],
    'ProtT5': [f'ProtT5_branch_fold_{fold}' for fold in range(1, 6)],
    'SaProt': [f'SaProt_branch_fold_{fold}.pth' for fold in range(1, 6)],
}


def calculate_entropy(probs):
    eps = 1e-09
    entropy = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
    return entropy / np.log(2)

def configure_paths() -> None:
    names = ('FILE_ESM', 'FILE_T5', 'FILE_SAPROT', 'FUSION_OUTPUT', 'PREDICTION_OUTPUT',)
    missing = [name for name in names if not str(globals()[name]).strip()]
    if missing:
        raise ValueError("Set the following paths before running: " + ", ".join(missing))
    for name in names:
        globals()[name] = Path(globals()[name]).expanduser()

def main():
    configure_paths()
    dfs = {name: pd.read_csv(path) for name, path in {'ESM2': FILE_ESM, 'ProtT5': FILE_T5, 'SaProt': FILE_SAPROT}.items()}
    labels = dfs['ESM2']['label'].values
    eps = 1e-06
    s_esm = np.clip(dfs['ESM2']['score'].values, eps, 1 - eps)
    s_t5 = np.clip(dfs['ProtT5']['score'].values, eps, 1 - eps)
    s_saprot = np.clip(dfs['SaProt']['score'].values, eps, 1 - eps)
    l_esm = logit(s_esm)
    l_t5 = logit(s_t5)
    l_saprot = logit(s_saprot)
    u_esm = calculate_entropy(s_esm)
    u_t5 = calculate_entropy(s_t5)
    u_saprot = calculate_entropy(s_saprot)

    def objective(params):
        theta = np.exp(params[:3]) / np.sum(np.exp(params[:3]))
        lam = params[3]
        w_esm = theta[0] * np.exp(-lam * u_esm)
        w_t5 = theta[1] * np.exp(-lam * u_t5)
        w_saprot = theta[2] * np.exp(-lam * u_saprot)
        w_total = w_esm + w_t5 + w_saprot
        w_esm, w_t5, w_saprot = (w_esm / w_total, w_t5 / w_total, w_saprot / w_total)
        final_logit = w_esm * l_esm + w_t5 * l_t5 + w_saprot * l_saprot
        final_prob = expit(final_logit)
        return -roc_auc_score(labels, final_prob)
    initial_guess = [1.0, 1.0, 1.0, 0.1]
    res = minimize(objective, initial_guess, method='Nelder-Mead', options={'maxiter': 1000})
    best_theta = np.exp(res.x[:3]) / np.sum(np.exp(res.x[:3]))
    best_lam = res.x[3]
    best_w_esm = best_theta[0] * np.exp(-best_lam * u_esm)
    best_w_t5 = best_theta[1] * np.exp(-best_lam * u_t5)
    best_w_saprot = best_theta[2] * np.exp(-best_lam * u_saprot)
    w_total = best_w_esm + best_w_t5 + best_w_saprot
    best_w_esm, best_w_t5, best_w_saprot = (best_w_esm / w_total, best_w_t5 / w_total, best_w_saprot / w_total)
    final_prob = expit(best_w_esm * l_esm + best_w_t5 * l_t5 + best_w_saprot * l_saprot)
    final_auc = roc_auc_score(labels, final_prob)
    preds = (final_prob >= 0.5).astype(int)
    print(f'Theta: ESM2={best_theta[0]:.3f}, ProtT5={best_theta[1]:.3f}, SaProt={best_theta[2]:.3f}')
    print(f'Lambda: {best_lam:.4f}')
    print(f'AUC: {final_auc:.5f}')
    print(f'ACC: {accuracy_score(labels, preds):.5f}')
    print(f'MCC: {matthews_corrcoef(labels, preds):.5f}')
    FUSION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FUSION_OUTPUT.open('w', encoding='utf-8') as handle:
        json.dump({'theta_esm2': float(best_theta[0]), 'theta_prott5': float(best_theta[1]), 'theta_saprot': float(best_theta[2]), 'lambda': float(best_lam), 'auc': float(final_auc), 'branch_checkpoints': BRANCH_CHECKPOINTS}, handle, indent=2)
    pd.DataFrame({'label': labels, 'ESM2_score': s_esm, 'ProtT5_score': s_t5, 'SaProt_score': s_saprot, 'score': final_prob, 'prediction': preds}).to_csv(PREDICTION_OUTPUT, index=False)
    print(f'Fusion parameters saved to: {FUSION_OUTPUT}')
    print(f'OOF predictions saved to: {PREDICTION_OUTPUT}')
if __name__ == '__main__':
    main()
