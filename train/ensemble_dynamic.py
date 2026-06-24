import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from scipy.special import expit, logit
from scipy.optimize import minimize
import os
FILE_ESM = "./scores/ESM2_BiGating_results.csv"
FILE_T5 = "./scores/ProtT5_Fusion_results.csv"
FILE_SAPROT = "./scores/SaProt_modify_results.csv"
def calculate_entropy(probs):
    eps = 1e-9
    entropy = - (probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
    return entropy / np.log(2)
def main():
    print("Starting uncertainty-aware ensemble optimization...")
    dfs = {name: pd.read_csv(path) for name, path in
           {"ESM2": FILE_ESM, "ProtT5": FILE_T5, "SaProt": FILE_SAPROT}.items()}
    labels = dfs["ESM2"]['label'].values
    eps = 1e-6
    s_esm = np.clip(dfs["ESM2"]['score'].values, eps, 1 - eps)
    s_t5 = np.clip(dfs["ProtT5"]['score'].values, eps, 1 - eps)
    s_saprot = np.clip(dfs["SaProt"]['score'].values, eps, 1 - eps)
    l_esm, l_t5, l_saprot = logit(s_esm), logit(s_t5), logit(s_saprot)
    u_esm, u_t5, u_saprot = calculate_entropy(s_esm), calculate_entropy(s_t5), calculate_entropy(s_saprot)
    def objective(params):
        theta = np.exp(params[:3]) / np.sum(np.exp(params[:3]))
        lam = params[3]
        w_esm = theta[0] * np.exp(-lam * u_esm)
        w_t5 = theta[1] * np.exp(-lam * u_t5)
        w_saprot = theta[2] * np.exp(-lam * u_saprot)
        w_total = w_esm + w_t5 + w_saprot
        w_esm, w_t5, w_saprot = w_esm / w_total, w_t5 / w_total, w_saprot / w_total
        final_logit = w_esm * l_esm + w_t5 * l_t5 + w_saprot * l_saprot
        return -roc_auc_score(labels, expit(final_logit))
    res = minimize(objective, [1.0, 1.0, 1.0, 0.1], method='Nelder-Mead', options={'maxiter': 1000})
    best_theta = np.exp(res.x[:3]) / np.sum(np.exp(res.x[:3]))
    best_lam = res.x[3]
    best_w_esm = best_theta[0] * np.exp(-best_lam * u_esm)
    best_w_t5 = best_theta[1] * np.exp(-best_lam * u_t5)
    best_w_saprot = best_theta[2] * np.exp(-best_lam * u_saprot)
    w_total = best_w_esm + best_w_t5 + best_w_saprot
    w_esm, w_t5, w_saprot = best_w_esm / w_total, best_w_t5 / w_total, best_w_saprot / w_total
    final_prob = expit(w_esm * l_esm + w_t5 * l_t5 + w_saprot * l_saprot)
    final_auc = roc_auc_score(labels, final_prob)
    preds = (final_prob >= 0.5).astype(int)
    print("\n" + "=" * 50)
    print(f"Baseline weights (Theta): ESM={best_theta[0]:.3f}, T5={best_theta[1]:.3f}, SaProt={best_theta[2]:.3f}")
    print(f"Uncertainty penalty (Lambda): {best_lam:.4f}")
    print(f"Final AUC: {final_auc:.5f}")
    print(f"Final ACC: {accuracy_score(labels, preds):.5f}")
    print(f"Final MCC: {matthews_corrcoef(labels, preds):.5f}")
    print("=" * 50)

if __name__ == "__main__":
    main()