"""Figures (figures/) and the LaTeX tables/number macros used by the report
(results/latex/) from results/*.csv."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = "results"
FIG = "figures"
TEX = "results/latex"  # tables + number macros \input by the report
os.makedirs(FIG, exist_ok=True)

# Reference palette (dataviz skill): sequential blue for epsilon (an ordered
# magnitude, light = strong privacy, dark = weak), neutral ink for non-DP.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
EPS = [0.1, 1, 10, 100, 1000]
EPS_COLOR = dict(zip(EPS, BLUES))
NONDP = "#52514e"
ACCENT = "#eb6834"  # categorical slot 2, used once for the "claimed epsilon" line

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.facecolor": "white", "lines.linewidth": 2, "lines.markersize": 5,
    "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def eps_label(e):
    return "non-DP" if not np.isfinite(e) else f"ε={e:g}"


def load():
    nd = pd.read_csv(f"{R}/nondp.csv")
    dp = pd.read_csv(f"{R}/dp.csv")
    return nd, dp


def fig_heatmaps(nd, dp):
    fig, axes = plt.subplots(1, 6, figsize=(11.5, 2.35), sharey=True)
    frames = [("non-DP", nd)] + [(f"ε = {e:g}", dp[dp.eps == e]) for e in EPS]
    for ax, (title, d) in zip(axes, frames):
        p = d.pivot(index="B", columns="lr", values="val_auroc")
        im = ax.imshow(p.values, cmap="Blues", vmin=0.50, vmax=0.66, aspect="auto")
        ax.set_xticks(range(p.shape[1]), [f"{v:g}" for v in p.columns])
        ax.set_yticks(range(p.shape[0]), p.index)
        ax.set_title(title, fontsize=9, color=INK)
        ax.grid(False)
        best = np.unravel_index(np.nanargmax(p.values), p.shape)
        for (i, j), v in np.ndenumerate(p.values):
            ax.text(j, i, f"{v:.3f}"[1:], ha="center", va="center", fontsize=6.3,
                    color="white" if v > 0.6 else INK,
                    fontweight="bold" if (i, j) == best else "normal")
        ax.set_xlabel("learning rate")
    axes[0].set_ylabel("batch size")
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("validation AUROC")
    fig.savefig(f"{FIG}/fig1_heatmaps.png")
    plt.close(fig)


def best_over_lr(df):
    """For each (B, eps), the lr chosen on validation; report its test numbers."""
    idx = df.groupby(["eps", "B"])["val_auroc"].idxmax()
    return df.loc[idx]


def fig_batch_eps(nd, dp):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    b = best_over_lr(dp)
    bn = best_over_lr(nd.assign(eps=np.inf))
    for ax, metric, ylab in [(axes[0], "test_auroc", "test AUROC (macro, OvR)"),
                             (axes[1], "test_acc", "test accuracy")]:
        for e in EPS:
            s = b[b.eps == e].sort_values("B")
            ax.plot(s.B, s[metric], "-o", color=EPS_COLOR[e], label=eps_label(e))
        s = bn.sort_values("B")
        ax.plot(s.B, s[metric], "--s", color=NONDP, label="non-DP")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512, 1024], ["64", "128", "256", "512", "1024"])
        ax.set_xlabel("batch size (fixed compute: B × steps = 614,400)")
        ax.set_ylabel(ylab)
    axes[1].axhline(0.5948, color=INK2, lw=1, ls=":")
    axes[1].text(1024, 0.5948, "majority class ", va="top", ha="right", fontsize=7, color=INK2)
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 0.87, 1))
    fig.legend(h, l, loc="center right", fontsize=7.5, title="privacy", title_fontsize=7.5)
    fig.savefig(f"{FIG}/fig2_batch_vs_eps.png")
    plt.close(fig)


def fig_steps():
    if not os.path.exists(f"{R}/steps.csv"):
        return
    s = pd.read_csv(f"{R}/steps.csv")
    fig, ax = plt.subplots(figsize=(3.7, 2.6))
    for e in sorted(s.eps.unique()):
        d = s[s.eps == e].sort_values("epochs_equiv")
        col = EPS_COLOR.get(e, NONDP)
        ax.plot(d.epochs_equiv, d.test_auroc, "-o" if np.isfinite(e) else "--s",
                color=col, label=eps_label(e))
    ax.set_xscale("log", base=2)
    ax.set_xticks([5, 10, 20, 40, 80], ["5", "10", "20", "40", "80"])
    ax.set_xlabel("training length (passes over the data)")
    ax.set_ylabel("test AUROC")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_steps.png")
    plt.close(fig)


def fig_audit():
    a = pd.read_csv(f"{R}/audit.csv")
    roc = np.load(f"{R}/audit_roc.npz")
    has_c = os.path.exists(f"{R}/audit_canaries.csv")
    fig, axes = plt.subplots(1, 3 if has_c else 2, figsize=(10.5 if has_c else 7, 2.8))
    ax = axes[0]
    for e in EPS + [np.inf]:
        k = str(float(e)) if np.isfinite(e) else "inf"
        f, t = roc[f"{k}_fpr"], roc[f"{k}_tpr"]
        ax.plot(f, t, color=EPS_COLOR.get(e, NONDP), lw=1.6, ls="-" if np.isfinite(e) else "--",
                label=eps_label(e))
    ax.plot([1e-4, 1], [1e-4, 1], color=INK2, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-4, 1); ax.set_ylim(1e-4, 1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("LOSS attack on all 59,970 patients", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    if has_c:
        c = pd.read_csv(f"{R}/audit_canaries.csv")
        ax = axes[1]
        x = np.arange(len(c))
        ax.bar(x - 0.2, c.auroc, 0.38, color=[EPS_COLOR.get(e, NONDP) for e in c.eps])
        ax.bar(x + 0.2, a.set_index("eps").loc[c.eps, "auroc"].values, 0.38,
               color="white", edgecolor=INK2, hatch="////", lw=0.6)
        ax.set_xticks(x, [eps_label(e).replace("ε=", "") for e in c.eps], fontsize=7)
        ax.set_ylim(0.45, max(0.8, c.auroc.max() + 0.03))
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("training ε")
        ax.set_ylabel("attack AUROC")
        ax.set_title("canaries (solid) vs natural patients (hatched)", fontsize=9)
    ax = axes[-1]
    src = pd.read_csv(f"{R}/audit_canaries.csv") if has_c else a
    fin = src[np.isfinite(src.eps)]
    ax.plot(fin.eps, fin.eps, color=ACCENT, lw=1.2, label="claimed ε (upper bound)")
    ax.plot(fin.eps, fin.eps_lb_one_run_fdp, "o-", color=BLUES[3],
            label="audited lower bound (f-DP one-run)")
    ax.plot(fin.eps, fin.eps_lb_one_run, "s--", color=BLUES[1], label="audited lower bound (one-run)")
    nd = src[~np.isfinite(src.eps)]
    if len(nd):
        ax.axhline(float(nd.eps_lb_one_run_fdp.iloc[0]), color=NONDP, ls="--", lw=1)
        ax.text(0.1, float(nd.eps_lb_one_run_fdp.iloc[0]), "non-DP model", fontsize=7,
                va="bottom", color=INK2)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(0, 2000)
    ax.set_xlabel("claimed ε"); ax.set_ylabel("ε")
    ax.set_title("audit: lower bound vs guarantee" + (" (canaries)" if has_c else ""), fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_audit.png")
    plt.close(fig)


def tables(nd, dp):
    os.makedirs(TEX, exist_ok=True)
    b = best_over_lr(dp)
    lines = []
    for e in EPS:
        row = [f"{e:g}"]
        for B in [64, 128, 256, 512, 1024]:
            r = b[(b.eps == e) & (b.B == B)].iloc[0]
            row.append(f"{r.test_auroc:.3f} ({r.lr:g})")
        lines.append(" & ".join(row) + r" \\")
        bn = best_over_lr(nd.assign(eps=np.inf))
    row = [r"$\infty$ (non-DP)"]
    for B in [64, 128, 256, 512, 1024]:
        r = bn[bn.B == B].iloc[0]
        row.append(f"{r.test_auroc:.3f} ({r.lr:g})")
    lines.append(r"\midrule " + " & ".join(row) + r" \\")
    head = (r"\begin{tabular}{lccccc}" "\n" r"\toprule" "\n"
            r"$\varepsilon$ & $B=64$ & $B=128$ & $B=256$ & $B=512$ & $B=1024$\\" "\n" r"\midrule")
    open(f"{TEX}/tab_utility.tex", "w").write("\n".join([head] + lines + [r"\bottomrule", r"\end{tabular}"]))

    a = pd.read_csv(f"{R}/audit.csv")
    rows = []
    c = pd.read_csv(f"{R}/audit_canaries.csv") if os.path.exists(f"{R}/audit_canaries.csv") else None
    for _, r in a.iterrows():
        e = r"$\infty$" if not np.isfinite(r.eps) else f"{r.eps:g}"
        cells = [e, f"{r.test_auroc:.3f}", f"{r.auroc:.4f}", f"{r.bal_acc_yeom:.4f}",
                 f"{r.bal_acc_best:.4f}", f"{100*r['tpr@0.001']:.2f}", f"{100*r['tpr@0.01']:.2f}",
                 f"{100*r['tpr@0.1']:.1f}", f"{r.eps_lb_one_run_fdp:.3f}"]
        if c is not None:
            cr = c[np.isclose(c.eps, r.eps) | (~np.isfinite(c.eps) & ~np.isfinite(r.eps))].iloc[0]
            cells += [f"{cr.auroc:.3f}", f"{100*cr['tpr@0.01']:.1f}", f"{cr.eps_lb_one_run_fdp:.2f}"]
        rows.append(" & ".join(cells) + r" \\")
    head = "\n".join([
        r"\begin{tabular}{lc|ccccccc|ccc}", r"\toprule",
        r"& & \multicolumn{7}{c|}{natural patients (30,200 in / 29,770 out)} & \multicolumn{3}{c}{OOD canaries}\\",
        r"$\varepsilon$ & test AUROC & AUROC & BA (mean-loss) & BA (best) & TPR@0.1 & TPR@1 & TPR@10 & $\hat\varepsilon$ & AUROC & TPR@1 & $\hat\varepsilon$\\",
        r"\midrule"])
    open(f"{TEX}/tab_audit.tex", "w").write("\n".join([head] + rows + [r"\bottomrule", r"\end{tabular}"]))


def numbers(nd, dp):
    """Every number quoted in the report text, as LaTeX macros."""
    sel = json.load(open(f"{R}/selected_config.json"))
    a = pd.read_csv(f"{R}/audit.csv").set_index("eps")
    b = best_over_lr(dp)
    ndb = nd.loc[nd.val_auroc.idxmax()]
    m = {
        "selB": sel["dp_B"], "selLR": f"{sel['dp_lr']:g}",
        "ndB": sel["nondp_B"], "ndLR": f"{sel['nondp_lr']:g}",
        "ndTestAUROC": f"{ndb.test_auroc:.3f}", "ndTestAcc": f"{100*ndb.test_acc:.1f}",
        "nTrain": "30{,}200", "nTest": "29{,}770",
    }
    for e, tag in zip(EPS, ["A", "B", "C", "D", "E"]):
        r = b[(b.eps == e)].sort_values("val_auroc").iloc[-1]  # select on validation
        m[f"bestAUROCeps{tag}"] = f"{r.test_auroc:.3f}"
        m[f"bestBeps{tag}"] = int(r.B)
        m[f"auditAUROCeps{tag}"] = f"{a.loc[e, 'auroc']:.4f}"
        m[f"auditLBeps{tag}"] = f"{a.loc[e, 'eps_lb_one_run_fdp']:.3f}"
    m["auditAUROCnd"] = f"{a.loc[np.inf, 'auroc']:.4f}"
    # Utility cost of DP at the audited configuration vs the tuned non-DP model
    m["costEpsOne"] = f"{100*(ndb.test_auroc - a.loc[1, 'test_auroc']):.1f}"
    m["costEpsTen"] = f"{100*(ndb.test_auroc - a.loc[10, 'test_auroc']):.1f}"
    for e, tag in [(0.1, "A"), (1, "B"), (10, "C"), (1000, "E"), (np.inf, "ND")]:
        m[f"pLtThirty{tag}"] = f"{a.loc[e, 'mean_p_lt30']:.3f}"
        m[f"aurocLtThirty{tag}"] = f"{a.loc[e, 'auroc_lt30']:.3f}"
        m[f"aurocNo{tag}"] = f"{a.loc[e, 'auroc_no']:.3f}"
        m[f"testAUROCaudit{tag}"] = f"{a.loc[e, 'test_auroc']:.3f}"
    m["rateLtThirty"] = f"{a.loc[np.inf, 'rate_lt30']:.3f}"
    m["auditLBnd"] = f"{a.loc[np.inf, 'eps_lb_one_run_fdp']:.3f}"
    if os.path.exists(f"{R}/audit_canaries.csv"):
        c = pd.read_csv(f"{R}/audit_canaries.csv")
        c["eps"] = c.eps.astype(float)
        c = c.set_index("eps")
        for e, tag in zip(EPS, ["A", "B", "C", "D", "E"]):
            m[f"canAUROCeps{tag}"] = f"{c.loc[e, 'auroc']:.3f}"
            m[f"canLBeps{tag}"] = f"{c.loc[e, 'eps_lb_one_run_fdp']:.2f}"
        m["canAUROCnd"] = f"{c.loc[np.inf, 'auroc']:.3f}"
        m["canLBnd"] = f"{c.loc[np.inf, 'eps_lb_one_run_fdp']:.2f}"
    ab = ablation_numbers()
    m.update(ab)
    with open(f"{TEX}/numbers.tex", "w") as f:
        for k, v in m.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def ablation_numbers():
    m = {}
    if os.path.exists(f"{R}/lowlr.csv"):
        lo = pd.read_csv(f"{R}/lowlr.csv")
        m["lowlrFiveEpsA"] = "0.05"
        m["lowlrMaxHigh"] = f"{lo[lo.eps >= 1].test_auroc.max():.3f}"
    if os.path.exists(f"{R}/steps.csv"):
        s = pd.read_csv(f"{R}/steps.csv")
        one, ten, nd = s[s.eps == 1].sort_values("steps"), s[s.eps == 10].sort_values("steps"), \
            s[~np.isfinite(s.eps)].sort_values("steps")
        m["stepsSigmaLo"] = f"{one.sigma.iloc[0]:.2f}"
        m["stepsSigmaHi"] = f"{one.sigma.iloc[-1]:.2f}"
        m["stepsBestOne"] = f"{one.test_auroc.max():.3f}"
        m["stepsLongOne"] = f"{one.test_auroc.iloc[-1]:.3f}"
        m["stepsLongTen"] = f"{ten.test_auroc.iloc[0]:.3f}$\\to${ten.test_auroc.iloc[-1]:.3f}"
        m["stepsLongND"] = f"{nd.test_auroc.max():.3f}$\\to${nd.test_auroc.iloc[-1]:.3f}"
    if os.path.exists(f"{R}/clip.csv"):
        c = pd.read_csv(f"{R}/clip.csv")
        dp = pd.read_csv(f"{R}/dp.csv")
        sel = json.load(open(f"{R}/selected_config.json"))
        base = dp[(dp.B == sel["dp_B"]) & (dp.lr == sel["dp_lr"]) & dp.eps.isin([1, 10])].assign(clip=1.0)
        c = pd.concat([c, base])
        rng = c.groupby("eps").test_auroc.agg(lambda v: v.max() - v.min()).max()
        m["clipAUROCrange"] = f"{100*rng:.1f}"
        m["clipLLone"] = f"{c[(c.eps == 10) & (c["clip"] == 1.0)].test_logloss.iloc[0]:.2f}"
        m["clipLLsmall"] = f"{c[(c.eps == 10) & (c["clip"] == 0.01)].test_logloss.iloc[0]:.2f}"
    if os.path.exists(f"{R}/seeds.csv"):
        sd = pd.read_csv(f"{R}/seeds.csv")
        a = pd.read_csv(f"{R}/audit.csv")
        spread = {}
        for e, g in sd.groupby("eps"):
            vals = list(g.test_auroc) + list(a[a.eps == e].test_auroc)
            spread[e] = max(vals) - min(vals)
        m["seedSpreadHigh"] = f"{max(v for e, v in spread.items() if e >= 1):.3f}"
        m["seedSpreadLow"] = f"{spread.get(0.1, float('nan')):.3f}"
    if os.path.exists(f"{R}/accounting_check.csv"):
        ac = pd.read_csv(f"{R}/accounting_check.csv")
        m["accRdpSmall"] = f"{ac[(ac.target_eps == 0.1) & (ac.B == 64)].eps_rdp.iloc[0]:.3f}"
        m["accMaxRatio"] = f"{(ac.eps_pld_fine / ac.target_eps).max():.3f}"
    if os.path.exists(f"{R}/audit_canaries_mislabel.csv"):
        mc = pd.read_csv(f"{R}/audit_canaries_mislabel.csv")
        m["canMisAUROCnd"] = f"{mc[~np.isfinite(mc.eps)].auroc.iloc[0]:.3f}"
    return m


if __name__ == "__main__":
    nd, dp = load()
    fig_heatmaps(nd, dp)
    fig_batch_eps(nd, dp)
    fig_steps()
    if os.path.exists(f"{R}/audit.csv"):
        fig_audit()
        os.makedirs(TEX, exist_ok=True)
        tables(nd, dp)
        numbers(nd, dp)
    print("figures in figures/, LaTeX tables and numbers in results/latex/")
