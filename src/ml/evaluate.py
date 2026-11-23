"""Avaliação, limiar e faixas de risco (Integrante 2).

Regra de ouro: limiares são escolhidos SÓ com predições fora da amostra do
treino (oof_proba.csv). O conjunto de teste é usado uma única vez, no fim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
LIMIARES_PADRAO = np.round(np.arange(0.01, 0.995, 0.005), 3)


# ----------------------------------------------------------------------
# Limiar de decisão
# ----------------------------------------------------------------------
def tabela_limiares(y_true, proba, limiares=LIMIARES_PADRAO) -> pd.DataFrame:
    """Uma linha por limiar: % sinalizadas, recall, precisão, F2 e custo por 1.000."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    n = len(y)
    linhas = []
    for t in limiares:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f2 = (5 * prec * rec / (4 * prec + rec)) if (tp > 0) else 0.0
        linhas.append(
            {
                "limiar": float(t),
                "pct_sinalizadas": 100 * (tp + fp) / n,
                "recall": rec,
                "precisao": prec,
                "f2": f2,
                "fn_por_1000": 1000 * fn / n,
                "fp_por_1000": 1000 * fp / n,
            }
        )
    return pd.DataFrame(linhas)


def escolher_limiar_f2(tabela: pd.DataFrame) -> float:
    """limiar_decisao = limiar que maximiza F2 (empate: o maior limiar)."""
    melhor = tabela["f2"].max()
    return float(tabela.loc[tabela["f2"] == melhor, "limiar"].max())


def limiar_com_recall_minimo(tabela: pd.DataFrame, recall_min: float = 0.70) -> float:
    """Alternativa se >40% forem sinalizadas: maior limiar com recall >= recall_min."""
    ok = tabela[tabela["recall"] >= recall_min]
    if ok.empty:
        raise ValueError("Nenhum limiar atinge o recall mínimo pedido.")
    return float(ok["limiar"].max())


def escolher_limiar_alto(y_true, proba, limiar_decisao: float, prec_min: float = 0.30) -> float:
    """Menor limiar (>= limiar_decisao) com precisão >= prec_min.

    Se nunca chegar, usa o percentil 90 das probabilidades (regra do plano).
    """
    tab = tabela_limiares(y_true, proba)
    tab = tab[(tab["limiar"] >= limiar_decisao) & (tab["precisao"] >= prec_min)]
    if tab.empty:
        return float(np.percentile(proba, 90))
    return float(tab["limiar"].min())


def classificar_faixa(proba, limiar_decisao: float, limiar_alto: float):
    """Devolve 'baixo' | 'medio' | 'alto' (aceita escalar ou array)."""
    p = np.asarray(proba, dtype=float)
    faixa = np.where(p >= limiar_alto, "alto", np.where(p >= limiar_decisao, "medio", "baixo"))
    return faixa if faixa.ndim else str(faixa)


def taxa_por_faixa(y_true, proba, limiar_decisao: float, limiar_alto: float) -> pd.DataFrame:
    """Taxa observada de baixo peso por faixa. Precisa crescer baixo < médio < alto."""
    df = pd.DataFrame({"y": np.asarray(y_true), "faixa": classificar_faixa(proba, limiar_decisao, limiar_alto)})
    out = df.groupby("faixa")["y"].agg(n="size", baixo_peso="sum", taxa="mean").reindex(["baixo", "medio", "alto"])
    out["pct_gestantes"] = 100 * out["n"] / out["n"].sum()
    return out.reset_index()


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def metricas(y_true, proba, limiar: float) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "recall": recall_score(y, pred, zero_division=0),
        "precisao": precision_score(y, pred, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ic(y_true, proba, limiar: float, n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """IC de 95% por bootstrap para recall e PR-AUC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    rec, pr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        rec.append(recall_score(y[i], (p[i] >= limiar).astype(int), zero_division=0))
        pr.append(average_precision_score(y[i], p[i]))
    linhas = []
    for nome, v in [("recall", rec), ("pr_auc", pr)]:
        linhas.append({"metrica": nome, "media": np.mean(v),
                       "ic95_inf": np.percentile(v, 2.5), "ic95_sup": np.percentile(v, 97.5)})
    return pd.DataFrame(linhas)


def recall_por_grupo(y_true, proba, grupo, limiar: float) -> pd.DataFrame:
    """Recall, precisão e n por subgrupo (faixa etária, escolaridade, raça/cor)."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int),
                       "pred": (np.asarray(proba) >= limiar).astype(int),
                       "grupo": np.asarray(grupo)})
    linhas = []
    for g, d in df.groupby("grupo", dropna=False):
        linhas.append({"grupo": g, "n": len(d), "prevalencia": d["y"].mean(),
                       "recall": recall_score(d["y"], d["pred"], zero_division=0),
                       "precisao": precision_score(d["y"], d["pred"], zero_division=0)})
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def plot_pr(y_true, proba, limiar_decisao=None, limiar_alto=None, caminho=None):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, proba):.3f}")
    ax.axhline(np.mean(y_true), ls="--", c="gray", label="Prevalência (baseline)")
    for t, nome in [(limiar_decisao, "limiar de decisão"), (limiar_alto, "limiar alto")]:
        if t is not None:
            pred = np.asarray(proba) >= t
            r = (pred & (np.asarray(y_true) == 1)).sum() / max(1, np.sum(y_true))
            pc = (pred & (np.asarray(y_true) == 1)).sum() / max(1, pred.sum())
            ax.scatter([r], [pc], zorder=3, label=f"{nome} = {t:.3f}")
    ax.set(xlabel="Recall", ylabel="Precisão", title="Curva precisão-recall")
    ax.legend()
    _salvar(fig, caminho)
    return fig


def plot_matriz_confusao(y_true, proba, limiar: float, caminho=None):
    cm = confusion_matrix(y_true, (np.asarray(proba) >= limiar).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}".replace(",", "."), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Prev. 0", "Prev. 1"],
           yticklabels=["Real 0", "Real 1"], title=f"Matriz de confusão (limiar {limiar:.3f})")
    _salvar(fig, caminho)
    return fig


def plot_calibracao(y_true, proba, n_bins: int = 10, caminho=None):
    frac, medio = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Perfeita")
    ax.plot(medio, frac, marker="o", label=f"Modelo (Brier = {brier_score_loss(y_true, proba):.3f})")
    ax.set(xlabel="Score médio previsto", ylabel="Fração observada de baixo peso",
           title="Curva de calibração")
    ax.legend()
    _salvar(fig, caminho)
    return fig


def _salvar(fig, caminho):
    fig.tight_layout()
    if caminho:
        fig.savefig(caminho, dpi=150)
