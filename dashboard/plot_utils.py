import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dark theme to match dashboard
PLOT_STYLE = {
    "figure.facecolor": "#1e293b",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#64748b",
    "axes.labelcolor": "#94a3b8",
    "text.color": "#f1f5f9",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
}

def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def plot_lines(df, x, y, title, xlabel, ylabel):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#06b6d4", "#22d3ee", "#a78bfa", "#f472b6", "#34d399"]
        for i, (name, g) in enumerate(df.groupby("experiment")):
            ax.plot(g[x], g[y], label=name, color=colors[i % len(colors)], linewidth=2)
        ax.set_title(title, fontsize=14, fontweight=600)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, facecolor="#0f172a", edgecolor="#334155")
        ax.grid(True, alpha=0.2)
        fig.patch.set_facecolor("#1e293b")
        return fig_to_base64(fig)