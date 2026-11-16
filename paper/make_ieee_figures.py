"""Column-width figures for the IEEE paper (paper/OxyDrop_paper.tex).

Reads results/photon/TABLE.csv and results/photon/scores/distill2__L0.csv.
Writes paper/figures_ieee/fig_bar.pdf, fig_stress_gain.pdf, fig_reliability.pdf.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "results", "photon", "TABLE.csv")
SCORES = os.path.join(ROOT, "results", "photon", "scores", "distill2__L0.csv")
OUT = os.path.join(ROOT, "paper", "figures_ieee")

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "pdf.fonttype": 42})
W = 3.4  # IEEE column width in inches


def row(model, level):
    with open(TABLE) as f:
        for r in csv.DictReader(f):
            if r["model"] == model and r["level"] == level:
                return r
    raise KeyError((model, level))


def fig_bar():
    names = [("chain_v1", "Logistic chain v1\n(baseline)"), ("distill2", "PhotonDistill2\n(shipped)")]
    fig, ax = plt.subplots(figsize=(W, 2.1))
    for i, (key, label) in enumerate(names):
        r = row(key, "0")
        tm, lo, hi = float(r["tm_bacc"]), float(r["tm_lo"]), float(r["tm_hi"])
        ax.bar(i, tm, width=0.55, color=["#9aa5b1", "#2f6f9f"][i])
        ax.errorbar(i, tm, yerr=[[tm - lo], [hi - tm]], color="k", capsize=4, lw=0.9)
        ax.text(i, hi + 0.012, f"{tm:.3f} [{lo:.3f}, {hi:.3f}]", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, ls=":", color="k", lw=0.7)
    ax.text(1.55, 0.505, "chance", fontsize=6.5, ha="right", va="bottom")
    ax.set_xticks([0, 1], [n[1] for n in names])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Time-matched balanced acc.")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_bar.pdf"))
    plt.close(fig)


def fig_stress_gain():
    levels = [("4", "L4"), ("5", "L5"), ("X1", "X1")]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for i, (lv, lab) in enumerate(levels):
        r = row("distill2", lv)
        d, lo, hi = float(r["diff_vs_chain_v1"]), float(r["diff_lo"]), float(r["diff_hi"])
        ax.errorbar(i, d, yerr=[[d - lo], [hi - d]], fmt="o", color="#2f6f9f", capsize=4, ms=4, lw=0.9)
        ax.text(i, hi + 0.012, f"{d:+.3f}\n[{lo:+.3f}, {hi:+.3f}]\n{r['n_people_improved']}/{r['n_people_compared']} better",
                ha="center", va="bottom", fontsize=6.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(3), ["L4 (sim.)", "L5 (sim.)", "X1 (out of family)"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.03, 0.42)
    ax.set_ylabel("Paired TM gain vs chain")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_stress_gain.pdf"))
    plt.close(fig)


def fig_reliability():
    y, s, p, ans = [], [], [], []
    with open(SCORES) as f:
        for r in csv.DictReader(f):
            y.append(int(r["y"]))
            s.append(float(r["score"]))
            p.append(int(r["pred"]))
            ans.append(r["answer"] == "True")
    y, s, p, ans = map(np.asarray, (y, s, p, ans))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1.15, 1]})
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(s, edges) - 1, 0, 9)
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 20:
            xs.append(s[m].mean())
            ys.append(y[m].mean())
            ns.append(m.sum())
    a1.plot([0, 1], [0, 1], ls=":", color="k", lw=0.7)
    a1.plot(xs, ys, "o-", color="#2f6f9f", ms=3, lw=0.9)
    a1.set_xlabel("Model score (bin mean)")
    a1.set_ylabel("Share hypoxic")
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_title("(a) Reliability", fontsize=8)
    cm = np.array([[np.sum((y == t) & (p == q)) for q in (0, 1)] for t in (0, 1)])
    a2.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for t in range(2):
        for q in range(2):
            frac = cm[t, q] / cm[t].sum()
            a2.text(q, t, f"{cm[t, q]}\n({frac:.2f})", ha="center", va="center", fontsize=6.5,
                    color="white" if frac > 0.6 else "black")
    a2.set_xticks([0, 1], ["Normal", "Low"])
    a2.set_yticks([0, 1], ["Normal", "Low"])
    a2.set_xlabel("Forced decision")
    a2.set_ylabel("Reference")
    a2.set_title("(b) Confusion", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_reliability.pdf"))
    plt.close(fig)
    print("windows", len(y), "hypoxic", int(y.sum()), "answered", int(ans.sum()),
          "answered acc", round(float((p[ans] == y[ans]).mean()), 4), "cm", cm.tolist())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_bar()
    fig_stress_gain()
    fig_reliability()
