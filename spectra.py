#!/usr/bin/env python3
"""
Nuclear Spectra Analyzer

Workflow:
  1. Load spectrum CSV (single 'counts' column; row index = channel)
  2. Display spectrum — user clicks on N peaks
  3. User types the known energy (keV) for each peak
  4. Fit Gaussian + linear background to each peak
     → centroid, FWHM, resolution (FWHM/centroid %), net area
  5. Fit energy calibration polynomial (channel → keV)
  6. Display final calibrated spectrum (counts vs keV)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from scipy.optimize import curve_fit
import pandas as pd

# 2√(2 ln 2) ≈ 2.3548  — converts sigma to FWHM
FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ── data loading ──────────────────────────────────────────────────────────────

def load_spectrum(path: str):
    df = pd.read_csv(path)
    counts = df.iloc[:, 0].values.astype(float)
    channels = np.arange(len(counts), dtype=float)
    return channels, counts


# ── peak model: Gaussian on top of a linear background ───────────────────────

def gaussian_bg(x, a, b, amplitude, centroid, sigma):
    """a + b·x  +  A·exp(-(x-μ)²/(2σ²))"""
    return a + b * x + amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2)


def fit_peak(channels, counts, center_ch, window=80):
    """
    Fit gaussian_bg to a window of ±`window` channels around `center_ch`.
    Returns popt, perr, x_slice, y_slice  (raises on failure).
    """
    lo = max(0, int(round(center_ch)) - window)
    hi = min(len(channels) - 1, int(round(center_ch)) + window) + 1
    x = channels[lo:hi].astype(float)
    y = counts[lo:hi].astype(float)

    # background estimate from ends of window
    y_bg_est = (float(y[0]) + float(y[-1])) / 2.0
    A0 = max(float(y.max()) - y_bg_est, 1.0)
    b0 = (float(y[-1]) - float(y[0])) / (x[-1] - x[0])
    a0 = float(y[0]) - b0 * x[0]

    # sigma guess from half-maximum crossings
    half_max = y_bg_est + A0 / 2.0
    above = np.where(y > half_max)[0]
    if len(above) > 1:
        sigma0 = max((above[-1] - above[0]) / FWHM_FACTOR, 1.0)
    else:
        sigma0 = window / 4.0

    bounds = (
        [-np.inf, -np.inf, 0.0,  x[0],          0.5],
        [ np.inf,  np.inf, np.inf, x[-1], window * 0.9],
    )

    popt, pcov = curve_fit(
        gaussian_bg, x, y,
        p0=[a0, b0, A0, float(center_ch), sigma0],
        bounds=bounds,
        maxfev=15000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, x, y


def peak_metrics(popt):
    """Return (centroid_ch, fwhm_ch, resolution_%, net_area_counts)."""
    a, b, amplitude, centroid, sigma = popt
    fwhm = FWHM_FACTOR * abs(sigma)
    resolution_pct = fwhm / abs(centroid) * 100.0
    # Gaussian integral (background already excluded by design)
    net_area = amplitude * abs(sigma) * np.sqrt(2.0 * np.pi)
    return centroid, fwhm, resolution_pct, net_area


# ── plotting helpers ──────────────────────────────────────────────────────────

def _semilogy_counts(ax, x, y, **kw):
    """Plot counts on a log y-axis, masking zeros so they don't appear."""
    safe = np.where(y > 0, y, np.nan)
    ax.semilogy(x, safe, **kw)
    ax.set_ylim(bottom=0.5)


def _maximize(fig):
    """Maximize the figure window — tries each backend's API in turn."""
    mgr = fig.canvas.manager
    try:
        mgr.window.state("zoomed")       # TkAgg (Windows)
        return
    except AttributeError:
        pass
    try:
        mgr.window.showMaximized()       # Qt5/Qt6
        return
    except AttributeError:
        pass
    try:
        mgr.frame.Maximize(True)         # WxAgg
    except AttributeError:
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="Nuclear Spectra Analyzer")
    ap.add_argument("file", help="Path to spectrum CSV")
    ap.add_argument(
        "-o", "--output",
        help="Output directory (default: ./output/<csv_stem>/)",
        default=None,
    )
    return ap.parse_args()


def _draw_spectrum(ax, channels, counts, last_nonzero):
    _semilogy_counts(ax, channels, counts, color="black", lw=1)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts  (log scale)")
    border = last_nonzero * 0.02
    ax.set_xlim(-border, last_nonzero + border)


def main():
    args = _parse_args()
    path = os.path.abspath(args.file)
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")

    stem = os.path.splitext(os.path.basename(path))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output if args.output else os.path.join(script_dir, "output", stem)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 52)
    print("     Nuclear Spectra Analyzer")
    print("=" * 52)

    channels, counts = load_spectrum(path)
    print(f"\nInput:  {path}")
    print(f"Output: {out_dir}")
    print(f"Channels: {len(channels):,}")

    last_nonzero = int(np.flatnonzero(counts)[-1]) if np.any(counts > 0) else len(channels) - 1

    # ── Single figure, open the whole session ─────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    _maximize(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 1 — peak selection
    # ══════════════════════════════════════════════════════════════════════════
    ax_sel = fig.add_subplot(111)
    _draw_spectrum(ax_sel, channels, counts, last_nonzero)

    clicks, _vlines, _done = [], [], [False]

    def _update_sel_title(n):
        ax_sel.set_title(
            f"Shift+click to mark peaks  ({n} selected)  —  "
            "Shift+right-click to undo  —  Enter to continue",
            fontsize=11,
        )
        fig.canvas.draw_idle()

    _update_sel_title(0)

    def _on_click(event):
        if event.inaxes != ax_sel or event.xdata is None or event.key != "shift":
            return
        if event.button == 1:
            clicks.append((event.xdata, event.ydata))
            _vlines.append(ax_sel.axvline(event.xdata, color="red", lw=1.2, ls="--", alpha=0.8))
            _update_sel_title(len(clicks))
        elif event.button == 3 and clicks:
            clicks.pop()
            _vlines.pop().remove()
            _update_sel_title(len(clicks))

    def _on_key(event):
        if event.key == "enter":
            _done[0] = True

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    print("\nShift+click peaks, Shift+right-click to undo, Enter to continue.")
    plt.show(block=False)
    while not _done[0]:
        plt.pause(0.05)

    if not clicks:
        plt.close(fig)
        sys.exit("No peaks selected — exiting.")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 2 — energy entry: spectrum left, info panel right, cycle peaks
    # ══════════════════════════════════════════════════════════════════════════
    n = len(clicks)
    known_energies = [None] * n
    current_idx = [0]
    phase2_done = [False]

    fig.clear()

    # Spectrum — left portion
    ax_sp2 = fig.add_axes([0.04, 0.08, 0.62, 0.87])
    _draw_spectrum(ax_sp2, channels, counts, last_nonzero)

    # All peaks dim; first highlighted
    vlines2 = []
    for ch, _ in clicks:
        vl = ax_sp2.axvline(ch, color="red", lw=1.0, ls="--", alpha=0.2)
        vlines2.append(vl)
    vlines2[0].set_alpha(0.9)
    vlines2[0].set_linewidth(2.0)

    # Right info panel — light background, no ticks
    ax_info = fig.add_axes([0.69, 0.08, 0.28, 0.87])
    ax_info.set_facecolor("#f2f2f2")
    ax_info.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax_info.spines.values():
        sp.set_color("#cccccc")

    ax_info.text(0.5, 0.90, "Peak Energies", ha="center", va="center",
                 fontsize=13, fontweight="bold", transform=ax_info.transAxes, color="#333333")

    # Large counter — updates each cycle
    t_num = ax_info.text(0.5, 0.76, f"1 / {n}", ha="center", va="center",
                         fontsize=30, fontweight="bold",
                         transform=ax_info.transAxes, color="#cc2222")
    t_ch = ax_info.text(0.5, 0.63, f"Channel {clicks[0][0]:.0f}", ha="center", va="center",
                        fontsize=12, transform=ax_info.transAxes, color="#555555")
    ax_info.text(0.5, 0.52, "Energy (keV)", ha="center", va="center",
                 fontsize=11, transform=ax_info.transAxes, color="#333333")

    # TextBox — centred in the right panel
    tb_w = 0.28 * 0.74
    tb_h = 0.055
    tb_x = 0.69 + (0.28 - tb_w) / 2
    tb_cy = 0.08 + 0.87 * 0.43          # 43 % up the panel
    ax_tb2 = fig.add_axes([tb_x, tb_cy - tb_h / 2, tb_w, tb_h])
    tb2 = TextBox(ax_tb2, "", initial="")

    ax_info.text(0.5, 0.32, "Press Enter to confirm", ha="center", va="center",
                 fontsize=9, transform=ax_info.transAxes, color="#888888")
    t_err = ax_info.text(0.5, 0.24, "", ha="center", va="center",
                         fontsize=9, transform=ax_info.transAxes, color="#cc2222")

    def _submit_energy(text):
        i = current_idx[0]
        try:
            val = float(text.strip())
        except ValueError:
            t_err.set_text("Enter a valid number")
            fig.canvas.draw_idle()
            return
        known_energies[i] = val
        t_err.set_text("")

        vlines2[i].set_alpha(0.2)
        vlines2[i].set_linewidth(1.0)

        if i + 1 >= n:
            phase2_done[0] = True
            return

        current_idx[0] = i + 1
        j = current_idx[0]
        vlines2[j].set_alpha(0.9)
        vlines2[j].set_linewidth(2.0)
        t_num.set_text(f"{j + 1} / {n}")
        t_ch.set_text(f"Channel {clicks[j][0]:.0f}")
        tb2.set_val("")
        fig.canvas.draw_idle()

    tb2.on_submit(_submit_energy)
    fig.canvas.draw()

    while not phase2_done[0]:
        plt.pause(0.05)

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 3 — fit
    # ══════════════════════════════════════════════════════════════════════════
    fit_results, cal_ch, cal_keV, fit_data = [], [], [], []

    for i, ((ch_click, _), energy) in enumerate(zip(clicks, known_energies)):
        try:
            popt, _, _xs, _ys = fit_peak(channels, counts, ch_click)
            centroid, fwhm, res_pct, net_area = peak_metrics(popt)
            cal_ch.append(centroid)
            cal_keV.append(energy)
            fit_results.append(dict(index=i + 1, energy_keV=energy, centroid_ch=centroid,
                                    fwhm_ch=fwhm, resolution_pct=res_pct, net_area=net_area))
            fit_data.append((popt, ch_click))
        except Exception as exc:
            fit_results.append(None)
            fit_data.append(None)
            print(f"  [!] Peak {i + 1} fit failed: {exc}")

    # summary table
    col = [8, 14, 15, 12, 14, 14]
    print("\n" + "=" * 80)
    print(f"{'Peak':<{col[0]}} {'Energy(keV)':<{col[1]}} {'Centroid(ch)':<{col[2]}} "
          f"{'FWHM(ch)':<{col[3]}} {'Resolution(%)':<{col[4]}} {'Net Area(cts)'}")
    print("-" * 80)
    for r in fit_results:
        if r:
            print(f"{r['index']:<{col[0]}} {r['energy_keV']:<{col[1]}.3f} "
                  f"{r['centroid_ch']:<{col[2]}.2f} {r['fwhm_ch']:<{col[3]}.2f} "
                  f"{r['resolution_pct']:<{col[4]}.3f} {r['net_area']:>14,.0f}")
    print("=" * 80)

    if not cal_ch:
        plt.close(fig)
        print("\nNo successful fits.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 4 — calibration
    # ══════════════════════════════════════════════════════════════════════════
    cal_ch_arr, cal_keV_arr = np.array(cal_ch), np.array(cal_keV)

    if len(cal_ch_arr) == 1:
        slope = cal_keV_arr[0] / cal_ch_arr[0]
        energy_axis = slope * channels
        print(f"\nCalibration (1 pt, origin): E = {slope:.6f} × ch  keV")
    else:
        deg = min(2, len(cal_ch_arr) - 1)
        poly_coeffs = np.polyfit(cal_ch_arr, cal_keV_arr, deg)
        energy_axis = np.polyval(poly_coeffs, channels)
        print(f"\nCalibration (deg-{deg} poly): " +
              " + ".join(f"({c:.4e})·ch^{deg-i}" for i, c in enumerate(poly_coeffs)) + "  keV")
        residuals = cal_keV_arr - np.polyval(poly_coeffs, cal_ch_arr)
        for r, res in zip(fit_results, residuals):
            if r:
                print(f"  Peak {r['index']} ({r['energy_keV']:.3f} keV): {res:+.4f} keV")

    # ══════════════════════════════════════════════════════════════════════════
    # Phase 5 — results: calibrated spectrum top, peak fits bottom
    # ══════════════════════════════════════════════════════════════════════════
    nf = len(fit_data)
    fig.clear()
    gs = fig.add_gridspec(2, nf, height_ratios=[3, 2], hspace=0.5, wspace=0.3)
    ax_cal = fig.add_subplot(gs[0, :])

    e_last = energy_axis[last_nonzero]
    e_border = e_last * 0.02
    pos = energy_axis > 0
    _semilogy_counts(ax_cal, energy_axis[pos], counts[pos], color="black", lw=1)
    ax_cal.set_xlabel("Energy (keV)", fontsize=12)
    ax_cal.set_ylabel("Counts  (log scale)", fontsize=12)
    ax_cal.set_title("Calibrated Nuclear Spectrum", fontsize=13)
    ax_cal.set_xlim(-e_border, e_last + e_border)

    for r in fit_results:
        if r:
            ax_cal.axvline(r["energy_keV"], color="red", alpha=0.65, lw=1.0, ls="--")

    for i, (fd, r) in enumerate(zip(fit_data, fit_results)):
        ax = fig.add_subplot(gs[1, i])
        if fd is None or r is None:
            ax.set_title(f"Peak {i + 1} — FIT FAILED", fontsize=9, color="red")
            continue
        popt, ch_click = fd
        dw = 80 * 3
        lo = max(0, int(round(ch_click)) - dw)
        hi = min(len(channels) - 1, int(round(ch_click)) + dw) + 1
        x_d, y_d = channels[lo:hi], counts[lo:hi]
        x_dense = np.linspace(x_d[0], x_d[-1], 800)
        ax.plot(x_d, y_d, "b.", ms=3, alpha=0.7)
        ax.plot(x_dense, gaussian_bg(x_dense, *popt), "r-", lw=1.5)
        ax.plot(x_dense, popt[0] + popt[1] * x_dense, color="gray", lw=1.0, ls="--")
        ax.axvline(r["centroid_ch"], color="darkorange", lw=1.2, ls=":")
        ax.set_title(f"{r['energy_keV']:.2f} keV   Res {r['resolution_pct']:.2f}%", fontsize=9)
        ax.set_xlabel("Channel", fontsize=8)
        ax.set_ylabel("Counts", fontsize=8)
        # stats as unobtrusive in-plot text
        ax.text(0.97, 0.97,
                f"FWHM {r['fwhm_ch']:.1f} ch\nArea {r['net_area']:,.0f}",
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right", color="#333333")

    fig.canvas.draw()
    plt.savefig(os.path.join(out_dir, "analysis.png"), dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nSaved to {out_dir}:")
    print("  analysis.png")


if __name__ == "__main__":
    main()
