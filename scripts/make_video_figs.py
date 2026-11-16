"""Video figures for OxyDrop (1920x1080 PNGs plus a short GIF) in docs/video/.

Numbers are read from results/photon/TABLE.csv and the held-out LOSO score file
results/photon/scores/distill2__L0.csv. Run from the repo root:
    python scripts/make_video_figs.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "video")
TABLE = os.path.join(ROOT, "results", "photon", "TABLE.csv")
SCORES = os.path.join(ROOT, "results", "photon", "scores", "distill2__L0.csv")
HELD_OUT_REC = "100004L"

W, H, DPI = 19.2, 10.8, 100  # 1920x1080
plt.rcParams.update({"font.size": 26, "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREY, GREEN, RED = "#2b6cb0", "#dd6b20", "#718096", "#2f855a", "#c53030"


def table():
    rows = {}
    with open(TABLE) as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["level"])] = r
    return rows


def box(ax, x, y, w, h, text, color, fs=26):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=color, ec="black", lw=2, alpha=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="white",
            weight="bold", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=40, lw=3, color="black"))


def canvas(title):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.93, title, ha="center", va="center", fontsize=44, weight="bold")
    return fig, ax


def fig_a():
    fig, ax = canvas("PhotonDistill v2: from fingertip to answer")
    xs = [0.10, 0.295, 0.49, 0.685]
    labels = ["Fingertip on\nrear camera\n+ flash",
              "Colour video\n\nmean R, G, B\nper frame",
              "39 per-window\nfeatures\n\neach relative to\nthe person's own\nfirst minute",
              "3 small MLPs\n(32 / 32 / 64)\naveraged\n\nP(SpO2<90)\n+ abstain head"]
    colors = [GREY, BLUE, BLUE, ORANGE]
    for x, t, c in zip(xs, labels, colors):
        box(ax, x, 0.52, 0.165, 0.46, t, c, fs=24)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 0.09, 0.52, b - 0.09, 0.52)
    outs = [("Low", RED, 0.72), ("Normal", GREEN, 0.52), ("Unsure", GREY, 0.32)]
    for t, c, y in outs:
        box(ax, 0.90, y, 0.15, 0.13, t, c, fs=32)
        arrow(ax, 0.775, 0.52, 0.815, y)
    ax.text(0.5, 0.12, "Causal: each window uses only past frames. Runs in the browser.",
            ha="center", fontsize=28, color="#2d3748")
    fig.savefig(os.path.join(OUT, "a_pipeline.png"), dpi=DPI)
    plt.close(fig)


def fig_b():
    fig, ax = canvas("Distillation inside every leave-one-person-out fold")
    ax.add_patch(FancyBboxPatch((0.04, 0.17), 0.70, 0.66,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="#edf2f7", ec=BLUE, lw=4, ls="--"))
    ax.text(0.39, 0.79, "One fold: 5 training people", ha="center", fontsize=30,
            color=BLUE, weight="bold")
    box(ax, 0.17, 0.52, 0.20, 0.30,
        "Training data\n\nclean copies\n+ physics-degraded\n(simulated\ndistant flash)", GREY, fs=23)
    box(ax, 0.42, 0.65, 0.17, 0.17, "Teacher 1\nlogistic\nchain v1", BLUE, fs=24)
    box(ax, 0.42, 0.38, 0.17, 0.17, "Teacher 2\nDC-colour\nlogistic", BLUE, fs=24)
    box(ax, 0.645, 0.52, 0.15, 0.30, "Student\n\n3 MLPs\naveraged\n\nlabels +\nsoft targets", ORANGE, fs=23)
    arrow(ax, 0.27, 0.58, 0.333, 0.65)
    arrow(ax, 0.27, 0.46, 0.333, 0.38)
    arrow(ax, 0.505, 0.65, 0.57, 0.57)
    arrow(ax, 0.505, 0.38, 0.57, 0.47)
    box(ax, 0.89, 0.52, 0.17, 0.30, "Held-out\nperson\n\nnever seen\nin training", RED, fs=24)
    arrow(ax, 0.72, 0.52, 0.805, 0.52)
    ax.text(0.89, 0.30, "scored once", ha="center", fontsize=24, color=RED)
    ax.text(0.5, 0.08, "Repeated 6 times, once per held-out person.",
            ha="center", fontsize=30, color="#2d3748")
    fig.savefig(os.path.join(OUT, "b_distillation.png"), dpi=DPI)
    plt.close(fig)


def fig_c(rows):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0.12, 0.13, 0.82, 0.70])
    names = [("chain_v1", "Chain v1 (logistic)", BLUE),
             ("distill2", "PhotonDistill v2 (shipped)", ORANGE)]
    for i, (m, label, c) in enumerate(names):
        r = rows[(m, "0")]
        v, lo, hi = float(r["tm_bacc"]), float(r["tm_lo"]), float(r["tm_hi"])
        ax.bar(i, v, width=0.55, color=c, edgecolor="black", lw=2)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=25,
                    lw=3, capthick=3)
        ax.text(i + 0.30, v, f"{v:.3f}\n[{lo:.3f}, {hi:.3f}]", va="center",
                ha="left", fontsize=30, weight="bold")
    d = rows[("distill2", "0")]
    ax.axhline(0.5, color=GREY, ls=":", lw=3)
    ax.text(1.62, 0.51, "chance 0.5", color=GREY, fontsize=24, ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([n[1] for n in names], fontsize=30)
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Time-matched balanced accuracy", fontsize=28)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.93, "Clean recordings (L0), leave-one-person-out, 6 people",
             ha="center", fontsize=40, weight="bold")
    fig.text(0.5, 0.03,
             f"95% person-bootstrap CIs. Paired difference distill2 - chain: "
             f"{float(d['diff_vs_chain_v1']):+.3f} [{float(d['diff_lo']):+.3f}, "
             f"{float(d['diff_hi']):+.3f}], on par.",
             ha="center", fontsize=26)
    fig.savefig(os.path.join(OUT, "c_bacc_l0.png"), dpi=DPI)
    plt.close(fig)


def fig_d(rows):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0.12, 0.20, 0.82, 0.63])
    levels = [("4", "L4\n(8 mm)"), ("5", "L5\n(10 mm)"),
              ("X1", "X1\n(6 mm, extra stress)")]
    for i, (lv, label) in enumerate(levels):
        r = rows[("distill2", lv)]
        v, lo, hi = (float(r["diff_vs_chain_v1"]), float(r["diff_lo"]),
                     float(r["diff_hi"]))
        ax.bar(i, v, width=0.55, color=ORANGE, edgecolor="black", lw=2)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="black", capsize=25,
                    lw=3, capthick=3)
        ax.text(i, hi + 0.008, f"{v:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="bottom",
                ha="center", fontsize=28, weight="bold")
    ax.axhline(0, color="black", lw=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l[1] for l in levels], fontsize=28)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.05, 0.34)
    ax.set_ylabel("Balanced-accuracy gain vs chain v1", fontsize=28)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.93, "Stress test: physics-degraded recordings",
             ha="center", fontsize=40, weight="bold")
    fig.text(0.5, 0.03, "Paired person-bootstrap 95% CIs; all three exclude zero.",
             ha="center", fontsize=26)
    fig.savefig(os.path.join(OUT, "d_stress_gain.png"), dpi=DPI)
    plt.close(fig)


def fig_e(seconds=15, fps=6):
    t, spo2, p, ans = [], [], [], []
    with open(SCORES) as f:
        for r in csv.DictReader(f):
            if r["rec"] != HELD_OUT_REC:
                continue
            t.append(float(r["t"]) / 60.0)
            spo2.append(float(r["y_spo2"]))
            p.append(float(r["score"]))
            ans.append(r["answer"] == "True")
    n_frames = seconds * fps
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax1 = fig.add_axes([0.09, 0.56, 0.86, 0.32])
    ax2 = fig.add_axes([0.09, 0.11, 0.86, 0.36], sharex=ax1)
    fig.text(0.5, 0.94, f"distill2 scoring held-out recording {HELD_OUT_REC} window by window",
             ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.905, "LOSO prediction: this person was not in that fold's training set",
             ha="center", fontsize=14, color="#4a5568")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylim(min(spo2) - 2, 101)
    ax1.axhline(90, color=RED, ls="--", lw=1.5)
    ax1.set_ylabel("Reference SpO2 (%)", fontsize=14)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("P(SpO2 < 90)", fontsize=14)
    ax2.set_xlabel("Time (min)", fontsize=14)
    for a in (ax1, ax2):
        a.tick_params(labelsize=12)
        a.spines[["top", "right"]].set_visible(False)
    (l1,) = ax1.plot([], [], color=BLUE, lw=2)
    (l2,) = ax2.plot([], [], color=ORANGE, lw=2)
    (dot,) = ax2.plot([], [], "o", color="black", ms=8)
    label = ax2.text(0.99, 0.92, "", transform=ax2.transAxes, ha="right",
                     fontsize=18, weight="bold")
    ax2.plot([], [], color=GREY, alpha=0.3, lw=8, label="abstain (Unsure)")
    ax2.legend(loc="upper left", fontsize=12, frameon=False)
    shaded = [0]
    dt = (t[1] - t[0]) if len(t) > 1 else 1 / 30

    def update(k):
        i = max(1, int(round((k + 1) * len(t) / n_frames)))
        l1.set_data(t[:i], spo2[:i])
        l2.set_data(t[:i], p[:i])
        dot.set_data([t[i - 1]], [p[i - 1]])
        for j in range(shaded[0], i):
            if not ans[j]:
                ax2.axvspan(t[j] - dt / 2, t[j] + dt / 2, color=GREY, alpha=0.3, lw=0)
        shaded[0] = i
        if not ans[i - 1]:
            label.set_text("Unsure")
            label.set_color(GREY)
        elif p[i - 1] >= 0.5:
            label.set_text("Low")
            label.set_color(RED)
        else:
            label.set_text("Normal")
            label.set_color(GREEN)
        return l1, l2, dot, label

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(os.path.join(OUT, "e_heldout_scoring.gif"), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = table()
    fig_a()
    fig_b()
    fig_c(rows)
    fig_d(rows)
    fig_e()
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
