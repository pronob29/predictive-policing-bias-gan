"""
src/visualization/figures.py
=============================
Publication-quality figure generation for the GAN predictive policing paper.

Academic colour palette (Wong 2011 colorblind-safe + ColorBrewer):
  - Distinct, accessible colours for all categorical variables
  - Serif font, clean axes, 300 DPI
  - Exported as both PDF and PNG
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy import stats as sp_stats

# ── Academic Style Sheet ──────────────────────────────────────────────────────
STYLE = {
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         100,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "grid.color":         "#cccccc",
    "lines.linewidth":    1.8,
    "patch.linewidth":    0.8,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
}
plt.rcParams.update(STYLE)

# ── Colorblind-safe Academic Palette (Wong 2011) ──────────────────────────────
# Blue, Vermillion/Red-Orange, Teal-Green — clearly distinct, print-safe
PAL = {
    "Black":   "#0072B2",   # deep blue
    "White":   "#D55E00",   # vermillion / burnt orange
    "Neither": "#009E73",   # teal green
}

# Year palette — sequential blue family
YEAR_PAL = {
    2017: "#1f78b4",
    2018: "#33a02c",
    2019: "#e31a1c",
    2022: "#ff7f00",
}

# Mode palette
MODE_PAL = {
    "detected": "#0072B2",   # deep blue
    "reported": "#D55E00",   # vermillion
}

RACES = ["Black", "White", "Neither"]

MONTH_LABELS = ["Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _race_legend(ax: plt.Axes, loc: str = "upper right") -> None:
    patches = [mpatches.Patch(facecolor=PAL[r], edgecolor="white",
                              linewidth=0.5, label=f"{r}-majority")
               for r in RACES]
    ax.legend(handles=patches, loc=loc, framealpha=0.9,
              edgecolor="#cccccc", fancybox=False)


def save_figure(fig: plt.Figure, out_dir: str, name: str,
                formats: tuple = ("pdf", "png")) -> None:
    """Save figure in all requested formats and close it."""
    for fmt in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{fmt}"),
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Figure Functions ──────────────────────────────────────────────────────────

def fig1_monthly_detection(sim: pd.DataFrame, out_dir: str,
                            city: str = "Baltimore", year: int = 2019,
                            mode: str = "detected") -> None:
    """Fig 1 — Monthly detection rate by neighbourhood racial composition."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    sub = sim[(sim["City"] == city) & (sim["Year"] == year) &
              (sim["Mode"] == mode)].copy()
    sub["Det_Rate"] = sub["N_Detected"] / sub["N_Total"].clip(lower=1)

    styles = {
        "Black":   ("-",  "o", 7),
        "White":   ("--", "s", 7),
        "Neither": (":",  "^", 7),
    }
    for race in RACES:
        d = sub[sub["Race_Category"] == race].sort_values("Month")
        ls, mk, ms = styles[race]
        ax.plot(d["Month"], d["Det_Rate"] * 100,
                color=PAL[race], linestyle=ls, marker=mk,
                markersize=ms, linewidth=2.0,
                markerfacecolor=PAL[race], markeredgecolor="white",
                markeredgewidth=0.8, label=f"{race}-majority",
                zorder=3)

    ax.set_xlabel("Month (Feb–Dec)", labelpad=6)
    ax.set_ylabel("Detection Rate (%)", labelpad=6)
    ax.set_title(f"Monthly Crime Detection Rate by Neighbourhood Racial Composition\n"
                 f"{city} {year} — GAN {mode.capitalize()} Mode",
                 pad=10, fontweight="bold")
    ax.set_xticks(range(2, 13))
    ax.set_xticklabels(MONTH_LABELS, rotation=30, ha="right")
    _race_legend(ax)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig1_monthly_detection_rate")


def fig2_multiyear_dir(annual: pd.DataFrame, out_dir: str,
                       city: str = "Baltimore") -> None:
    """Fig 2 — Multi-year Disparate Impact Ratio with ±1 SD band."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    for mode in ["detected", "reported"]:
        d = annual[(annual["City"] == city) & (annual["Mode"] == mode)
                   ].sort_values("Year")
        color = MODE_PAL[mode]
        ls    = "-" if mode == "detected" else "--"
        mk    = "o" if mode == "detected" else "s"
        ax.plot(d["Year"], d["Avg_DIR"],
                color=color, linestyle=ls, marker=mk,
                markersize=7, linewidth=2.0,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8,
                label=f"{mode.capitalize()} Mode")
        ax.fill_between(d["Year"],
                        d["Avg_DIR"] - d["Std_DIR"],
                        d["Avg_DIR"] + d["Std_DIR"],
                        alpha=0.15, color=color)

    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.4,
               label="Parity (DIR = 1)", zorder=2)
    ax.set_xlabel("Year", labelpad=6)
    ax.set_ylabel("Disparate Impact Ratio  (Black / White)", labelpad=6)
    ax.set_title(f"Annual Disparate Impact Ratio — {city}\n"
                 f"Shaded Bands: ±1 SD Across Months",
                 pad=10, fontweight="bold")
    ax.set_xticks(sorted(annual[annual["City"] == city]["Year"].unique()))
    ax.legend(framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig2_multiyear_DIR")


def fig3_parity_gap(monthly: pd.DataFrame, out_dir: str,
                    city: str = "Baltimore", mode: str = "detected") -> None:
    """Fig 3 — Monthly demographic parity gap, multi-year overlay."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    years = sorted(monthly[monthly["City"] == city]["Year"].unique())

    styles = ["-", "--", ":"]
    markers = ["o", "s", "^"]
    for yr, ls, mk in zip(years, styles, markers):
        d = monthly[(monthly["City"] == city) & (monthly["Year"] == yr) &
                    (monthly["Mode"] == mode)].sort_values("Month")
        color = YEAR_PAL.get(yr, "#333333")
        ax.plot(d["Month"], d["Parity_Gap"] * 100,
                color=color, linestyle=ls, marker=mk,
                markersize=6, linewidth=2.0,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.8, label=str(yr))

    ax.axhline(0, color="#555555", linewidth=1.2, linestyle="-", zorder=2)
    ax.fill_between(range(2, 13), 0, 15, alpha=0.04, color="#D55E00")
    ax.fill_between(range(2, 13), -15, 0, alpha=0.04, color="#0072B2")
    ax.set_xlabel("Month (Feb–Dec)", labelpad=6)
    ax.set_ylabel("Parity Gap  (Black% − White%, pp)", labelpad=6)
    ax.set_title(f"Monthly Demographic Parity Gap — {city}\n"
                 f"GAN {mode.capitalize()} Mode; Positive = Black Areas Detected More",
                 pad=10, fontweight="bold")
    ax.set_xticks(range(2, 13))
    ax.set_xticklabels(MONTH_LABELS, rotation=30, ha="right")
    ax.legend(title="Year", framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig3_parity_gap")


def fig4_cross_city(annual: pd.DataFrame, out_dir: str) -> None:
    """Fig 4 — Cross-city DIR comparison (Baltimore vs Chicago)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    city_years = [("Baltimore", 2019), ("Chicago", 2022)]

    for ax, (city, yr) in zip(axes, city_years):
        bar_w = 0.35
        offsets  = {"detected": -0.2, "reported": 0.2}
        for mode in ["detected", "reported"]:
            d = annual[(annual["City"] == city) & (annual["Year"] == yr) &
                       (annual["Mode"] == mode)]
            if d.empty:
                continue
            color = MODE_PAL[mode]
            bar = ax.bar([0 + offsets[mode]], d["Avg_DIR"].values, width=bar_w,
                         color=color, alpha=0.88,
                         label=f"{mode.capitalize()} Mode",
                         yerr=d["Std_DIR"].values, capsize=5,
                         error_kw={"linewidth": 1.2, "ecolor": "#444444",
                                   "capthick": 1.2})
            # value label on bar
            for b in bar:
                ht = b.get_height()
                if ht < 500:
                    ax.text(b.get_x() + b.get_width() / 2, ht + 0.05,
                            f"{ht:.2f}", ha="center", va="bottom",
                            fontsize=8.5, color="#333333")

        ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.4)
        ax.set_xticks([0])
        ax.set_xticklabels([f"{city}\n{yr}"], fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Mean Disparate Impact Ratio  (Black / White)", labelpad=6)
            ax.legend(framealpha=0.9, edgecolor="#cccccc", fancybox=False)
        ax.set_title(f"{city} {yr}", pad=8, fontweight="bold")

    fig.suptitle("Cross-City Comparison of Disparate Impact Ratio\n"
                 "Baltimore 2019 vs. Chicago 2022  (Error bars: ±1 SD)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig4_cross_city_DIR")


def fig5_sensitivity(sens: pd.DataFrame, out_dir: str) -> None:
    """Fig 5 — Three-panel sensitivity analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    panels = [
        ("Radius_ft",      "Detection Radius (ft)",   "Detection Radius"),
        ("N_Officers",     "Number of Officers",       "Officers Deployed"),
        ("Reporting_Prob", "Reporting Probability",    "Reporting Probability"),
    ]

    panel_colors = ["#0072B2", "#009E73", "#D55E00"]

    for ax, (param, xlabel, title), color in zip(axes, panels, panel_colors):
        d = sens[sens["Parameter"] == param].sort_values("Value")
        if d.empty:
            continue
        ax.plot(d["Value"], d["DIR_Black_White"],
                color=color, marker="o", markersize=8, linewidth=2.2,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.0, zorder=4)
        ax.fill_between(d["Value"], 1.0, d["DIR_Black_White"],
                        where=d["DIR_Black_White"] >= 1.0,
                        alpha=0.18, color="#D55E00",
                        label="DIR > 1  (over-detection, Black)")
        ax.fill_between(d["Value"], d["DIR_Black_White"], 1.0,
                        where=d["DIR_Black_White"] < 1.0,
                        alpha=0.18, color="#0072B2",
                        label="DIR < 1  (under-detection, Black)")
        ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.4, zorder=3)
        ax.set_xlabel(xlabel, labelpad=6)
        if ax is axes[0]:
            ax.set_ylabel("Disparate Impact Ratio  (Black / White)", labelpad=6)
        ax.set_title(title, pad=8, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc",
                  fancybox=False, loc="best")

    fig.suptitle("Sensitivity Analysis — Disparate Impact Ratio Under Varying Simulation Parameters\n"
                 "Baltimore 2019, GAN Detected Mode",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig5_sensitivity_analysis")


def fig6_ctgan_debiasing(debias: pd.DataFrame, out_dir: str) -> None:
    """Fig 6 — CTGAN debiasing: detection rates and DIR comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    conditions = debias["Condition"].tolist()
    x = np.arange(len(conditions))
    bar_w = 0.32

    # Panel A — detection rates
    ax = axes[0]
    race_colors = {"Black": PAL["Black"], "White": PAL["White"]}
    for i, (col, race) in enumerate([("Pct_Det_Black", "Black"),
                                      ("Pct_Det_White", "White")]):
        bars = ax.bar(x + (i - 0.5) * bar_w, debias[col] * 100,
                      width=bar_w, color=race_colors[race],
                      alpha=0.88, label=f"{race}-majority",
                      edgecolor="white", linewidth=0.8)
        for b in bars:
            ht = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, ht + 0.05,
                    f"{ht:.2f}%", ha="center", va="bottom",
                    fontsize=8.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=12, ha="right")
    ax.set_ylabel("Detection Rate (%)", labelpad=6)
    ax.set_title("Detection Rate by Race", pad=8, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="#cccccc", fancybox=False)

    # Panel B — DIR
    ax = axes[1]
    dir_colors = ["#0072B2", "#D55E00"]
    bars = ax.bar(x, debias["DIR"],
                  color=dir_colors, alpha=0.88,
                  edgecolor="white", linewidth=0.8, width=0.5)
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.4,
               label="Parity (DIR = 1)")
    for bar, (_, row) in zip(bars, debias.iterrows()):
        ht = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, ht + 0.04,
                f"{row['DIR']:.2f}", ha="center", va="bottom",
                fontsize=9.5, color="#333333", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=12, ha="right")
    ax.set_ylabel("Disparate Impact Ratio", labelpad=6)
    ax.set_title("Disparate Impact Ratio", pad=8, fontweight="bold")
    ax.legend(framealpha=0.9, edgecolor="#cccccc", fancybox=False)

    fig.suptitle("CTGAN Debiasing — Biased vs. Race-Balanced GAN Training\n"
                 "Baltimore 2019, February Deployment",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig6_ctgan_debiasing")


def fig7_socioeconomic_scatter(nbhd: pd.DataFrame, out_dir: str) -> None:
    """Fig 7 — 2x3 scatter grid: socioeconomic correlates of detection rate."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    axes = axes.flatten()

    plot_vars = [
        ("Pct_Black",     "% Black Residents"),
        ("Pct_White",     "% White Residents"),
        ("Median_Income", "Median Household Income ($)"),
        ("Poverty_Rate",  "Poverty Rate"),
        ("Crime_Rate",    "Total Crime Count"),
    ]

    valid = nbhd[["Pct_Black", "Pct_White", "Median_Income",
                  "Poverty_Rate", "Crime_Rate", "Det_Rate",
                  "Race_Category"]].dropna()

    scatter_alpha = 0.65
    scatter_size  = 22

    for ax, (var, xlabel) in zip(axes[:5], plot_vars):
        for race in RACES:
            sub = valid[valid["Race_Category"] == race]
            ax.scatter(sub[var], sub["Det_Rate"] * 100,
                       color=PAL[race], alpha=scatter_alpha,
                       s=scatter_size, edgecolors="white",
                       linewidths=0.4, label=f"{race}-majority",
                       zorder=3)

        # Overall OLS regression line
        x_vals = valid[var]
        y_vals = valid["Det_Rate"] * 100
        slope, intercept, r, p, _ = sp_stats.linregress(x_vals, y_vals)
        x_fit = np.linspace(x_vals.min(), x_vals.max(), 200)
        p_label = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.plot(x_fit, slope * x_fit + intercept,
                color="#333333", linewidth=1.4, linestyle="--",
                label=f"r = {r:.2f}, {p_label}", zorder=4)

        ax.set_xlabel(xlabel, labelpad=5)
        ax.set_ylabel("Detection Rate (%)", labelpad=5)
        ax.set_title(xlabel, pad=7, fontweight="bold")
        ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc",
                  fancybox=False, loc="best")

    axes[5].axis("off")
    fig.suptitle("Socioeconomic Correlates of Crime Detection Rate\n"
                 "Baltimore Neighbourhoods (2017–2019 Pooled, n = 279)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig7_socioeconomic_scatter")


def fig8_dir_heatmap(monthly: pd.DataFrame, out_dir: str,
                     city: str = "Baltimore") -> None:
    """Fig 8 — DIR heatmap (month × year) for detected and reported modes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    modes = ["detected", "reported"]

    # Use a diverging colormap centred on DIR=1
    cmap_use = "RdYlBu_r"   # red = high DIR, blue = low DIR

    for ax, mode in zip(axes, modes):
        pivot = monthly[(monthly["City"] == city) & (monthly["Mode"] == mode)
                        ].pivot(index="Month", columns="Year", values="DIR")
        if pivot.empty:
            continue

        vmax = min(pivot.values.max(), 10)   # cap display at 10 for readability
        im = ax.imshow(pivot, aspect="auto",
                       cmap=cmap_use, vmin=0, vmax=vmax,
                       interpolation="nearest")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([MONTH_LABELS[i] for i in range(len(pivot.index))],
                           fontsize=9)
        ax.set_xlabel("Year", labelpad=6)
        if ax is axes[0]:
            ax.set_ylabel("Month", labelpad=6)
        ax.set_title(f"{mode.capitalize()} Mode", pad=8, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, label="DIR (Black / White)",
                            shrink=0.82, aspect=20, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

        # Annotate cells
        thresh = vmax * 0.55
        for (r, c), val in np.ndenumerate(pivot.values):
            disp = f"{min(val, 999):.1f}" if val < 1000 else ">999"
            ax.text(c, r, disp, ha="center", va="center",
                    fontsize=7.5,
                    color="white" if val > thresh else "#222222",
                    fontweight="bold")

    fig.suptitle(f"Disparate Impact Ratio Heatmap — {city}\n"
                 f"Month × Year by Simulation Mode  (Red = High DIR, Blue = Low DIR)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig8_DIR_heatmap")


def fig9_gini_trend(monthly: pd.DataFrame, out_dir: str) -> None:
    """Fig 9 — Gini coefficient trend across all simulation steps."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    city_configs = [("Baltimore", [2017, 2018, 2019]), ("Chicago", [2022])]

    for ax, (city, years) in zip(axes, city_configs):
        for mode in ["detected", "reported"]:
            d = monthly[(monthly["City"] == city) &
                        (monthly["Mode"] == mode)].sort_values(["Year", "Month"])
            if d.empty:
                continue
            color = MODE_PAL[mode]
            ls    = "-" if mode == "detected" else "--"
            ax.plot(range(len(d)), d["Gini"],
                    color=color, linestyle=ls, linewidth=2.0,
                    label=f"{mode.capitalize()} Mode")
            ax.fill_between(range(len(d)), d["Gini"], alpha=0.08, color=color)

        # Year boundary markers for Baltimore
        if city == "Baltimore":
            cumulative = 0
            for yr in years:
                cnt = len(monthly[(monthly["City"] == city) &
                                   (monthly["Year"] == yr) &
                                   (monthly["Mode"] == "detected")])
                if cumulative > 0:
                    ax.axvline(cumulative, color="#888888",
                               linestyle=":", linewidth=1.0, alpha=0.7)
                y_top = monthly[(monthly["City"] == city) &
                                (monthly["Mode"] == "detected")]["Gini"].max()
                ax.text(cumulative + cnt / 2, y_top * 1.01,
                        str(yr), ha="center", va="bottom",
                        fontsize=8.5, color="#444444", fontweight="bold")
                cumulative += cnt

        year_label = "2017–2019" if city == "Baltimore" else "2022"
        ax.set_xlabel("Simulation Step  (month)", labelpad=6)
        if ax is axes[0]:
            ax.set_ylabel("Gini Coefficient of Detection Inequality", labelpad=6)
        ax.set_title(f"{city} {year_label}", pad=8, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.legend(framealpha=0.9, edgecolor="#cccccc", fancybox=False)

    fig.suptitle("Gini Coefficient of Detection Inequality Over Time\n"
                 "Higher Values Indicate Greater Racial Inequality in Crime Detection",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig9_gini_trend")


# ── Master Call ───────────────────────────────────────────────────────────────

def generate_all_figures(
    sim: pd.DataFrame,
    monthly: pd.DataFrame,
    annual: pd.DataFrame,
    sens: pd.DataFrame,
    debias: pd.DataFrame,
    nbhd: pd.DataFrame,
    out_dir: str,
) -> None:
    """Generate all 9 paper figures and save to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving figures to: {out_dir}")

    fig1_monthly_detection(sim, out_dir)
    fig2_multiyear_dir(annual, out_dir)
    fig3_parity_gap(monthly, out_dir)
    fig4_cross_city(annual, out_dir)
    fig5_sensitivity(sens, out_dir)
    fig6_ctgan_debiasing(debias, out_dir)
    fig7_socioeconomic_scatter(nbhd, out_dir)
    fig8_dir_heatmap(monthly, out_dir)
    fig9_gini_trend(monthly, out_dir)

    figs = sorted(f for f in os.listdir(out_dir) if f.endswith(".pdf"))
    print(f"  ✓ {len(figs)} figures saved (PDF + PNG)")
    for f in figs:
        print(f"    {f}")
