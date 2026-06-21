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
import matplotlib.transforms as mtransforms
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

    # ── Step 1: interactive peak selection ───────────────────────────────────
    fig_sel, ax_sel = plt.subplots(figsize=(14, 6))
    _semilogy_counts(ax_sel, channels, counts, color="black", lw=1)
    ax_sel.set_xlabel("Channel")
    ax_sel.set_ylabel("Counts  (log scale)")
    border = last_nonzero * 0.02
    ax_sel.set_xlim(-border, last_nonzero + border)

    def _update_title(n):
        ax_sel.set_title(
            f"Shift+click to mark peaks  ({n} selected)  —  Shift+right-click to undo  —  Enter to finish",
            fontsize=11,
        )
        fig_sel.canvas.draw_idle()

    _update_title(0)
    _maximize(fig_sel)

    clicks = []
    _vlines = []
    _done = [False]

    def _on_click(event):
        if event.inaxes != ax_sel or event.xdata is None or event.key != "shift":
            return
        if event.button == 1:
            clicks.append((event.xdata, event.ydata))
            vl = ax_sel.axvline(event.xdata, color="red", lw=1.2, ls="--", alpha=0.8)
            _vlines.append(vl)
            _update_title(len(clicks))
        elif event.button == 3 and clicks:
            clicks.pop()
            _vlines.pop().remove()
            _update_title(len(clicks))

    def _on_key(event):
        if event.key == "enter":
            _done[0] = True

    fig_sel.canvas.mpl_connect("button_press_event", _on_click)
    fig_sel.canvas.mpl_connect("key_press_event", _on_key)

    print("\nSpectrum window open.")
    print("  → Zoom/pan freely with the toolbar.")
    print("  → Shift+click to mark a peak centroid.")
    print("  → Shift+right-click to remove the last mark.")
    print("  → Press Enter when finished.")

    plt.show(block=False)
    while not _done[0]:
        plt.pause(0.05)

    if not clicks:
        plt.close(fig_sel)
        sys.exit("No peaks selected — exiting.")

    print(f"\nReceived {len(clicks)} peak(s).")

    # ── Step 2: collect known energies (plot stays open) ─────────────────────
    _label_trans = mtransforms.blended_transform_factory(
        ax_sel.transData, ax_sel.transAxes
    )
    known_energies = []
    for i, (ch, _) in enumerate(clicks):
        _vlines[i].set_color("orange")
        _vlines[i].set_linewidth(2.0)
        ax_sel.set_title(
            f"Peak {i + 1} of {len(clicks)} highlighted — enter energy in terminal",
            fontsize=11,
        )
        fig_sel.canvas.draw()
        fig_sel.canvas.flush_events()

        e = float(input(f"  Peak {i + 1} near ch {ch:.0f}  [keV]: ").strip())
        known_energies.append(e)

        _vlines[i].set_color("red")
        _vlines[i].set_linewidth(1.2)
        ax_sel.text(
            ch, 0.97, f" {e:.1f} keV",
            transform=_label_trans, fontsize=8, color="darkred",
            rotation=90, va="top", ha="left",
        )
        fig_sel.canvas.draw_idle()

    ax_sel.set_title("Calibration complete", fontsize=11)
    fig_sel.canvas.draw()
    plt.close(fig_sel)

    # ── Step 3: fit each peak ────────────────────────────────────────────────
    n_fits = len(clicks)
    fig_fits, axes = plt.subplots(1, n_fits, figsize=(6 * n_fits, 5), squeeze=False)

    fit_results = []   # list of dicts
    cal_ch = []
    cal_keV = []

    for i, ((ch_click, _), energy) in enumerate(zip(clicks, known_energies)):
        ax = axes[0, i]
        try:
            popt, perr, x_slice, y_slice = fit_peak(channels, counts, ch_click)
            centroid, fwhm, res_pct, net_area = peak_metrics(popt)

            cal_ch.append(centroid)
            cal_keV.append(energy)
            fit_results.append(
                dict(
                    index=i + 1,
                    energy_keV=energy,
                    centroid_ch=centroid,
                    fwhm_ch=fwhm,
                    resolution_pct=res_pct,
                    net_area=net_area,
                )
            )

            # wider display window (3× fit window) so background tails are visible
            disp_window = 80 * 3
            lo_d = max(0, int(round(ch_click)) - disp_window)
            hi_d = min(len(channels) - 1, int(round(ch_click)) + disp_window) + 1
            x_disp = channels[lo_d:hi_d]
            y_disp = counts[lo_d:hi_d]

            a, b, _, _, _ = popt
            x_dense = np.linspace(x_disp[0], x_disp[-1], 800)
            y_fit_dense = gaussian_bg(x_dense, *popt)
            y_bg_dense = a + b * x_dense

            ax.plot(x_disp, y_disp, "b.", ms=3.5, label="Data")
            ax.plot(x_dense, y_fit_dense, "r-", lw=1.6, label="Gaussian + BG")
            ax.plot(x_dense, y_bg_dense, "g--", lw=1.2, label="Linear BG")
            ax.axvline(centroid, color="darkorange", lw=1.5, ls=":", label=f"μ = {centroid:.1f} ch")
            ax.set_title(
                f"Peak {i+1}:  {energy:.3f} keV\n"
                f"μ = {centroid:.1f} ch  |  FWHM = {fwhm:.1f} ch\n"
                f"Resolution = {res_pct:.2f}%  |  Net area = {net_area:,.0f} cts",
                fontsize=9,
            )

        except Exception as exc:
            ax.set_title(f"Peak {i+1} — FIT FAILED\n{exc}", fontsize=9, color="red")
            print(f"  [!] Peak {i+1} fit failed: {exc}")

        ax.set_xlabel("Channel")
        ax.set_ylabel("Counts")
        ax.legend(fontsize=7.5)

    fig_fits.suptitle("Peak Fits with Linear Background Subtraction", fontsize=12)
    fig_fits.tight_layout()
    _maximize(fig_fits)
    plt.savefig(os.path.join(out_dir, "peak_fits.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # ── Step 4: print summary table ──────────────────────────────────────────
    col = [8, 14, 15, 12, 14, 14]
    header = (
        f"{'Peak':<{col[0]}} {'Energy(keV)':<{col[1]}} "
        f"{'Centroid(ch)':<{col[2]}} {'FWHM(ch)':<{col[3]}} "
        f"{'Resolution(%)':<{col[4]}} {'Net Area(cts)'}"
    )
    print("\n" + "=" * 80)
    print(header)
    print("-" * 80)
    for r in fit_results:
        print(
            f"{r['index']:<{col[0]}} {r['energy_keV']:<{col[1]}.3f} "
            f"{r['centroid_ch']:<{col[2]}.2f} {r['fwhm_ch']:<{col[3]}.2f} "
            f"{r['resolution_pct']:<{col[4]}.3f} {r['net_area']:>14,.0f}"
        )
    print("=" * 80)

    if not cal_ch:
        print("\nNo successful fits — cannot build calibration.  Exiting.")
        return

    # ── Step 5: energy calibration ───────────────────────────────────────────
    cal_ch_arr = np.array(cal_ch)
    cal_keV_arr = np.array(cal_keV)

    if len(cal_ch_arr) == 1:
        # single point: straight line through origin
        slope = cal_keV_arr[0] / cal_ch_arr[0]
        energy_axis = slope * channels
        print(f"\nCalibration (1 point, forced through origin):")
        print(f"  E = {slope:.6f} × ch   [keV]")
    else:
        deg = min(2, len(cal_ch_arr) - 1)
        poly_coeffs = np.polyfit(cal_ch_arr, cal_keV_arr, deg)
        energy_axis = np.polyval(poly_coeffs, channels)
        poly_str = "  E = " + " + ".join(
            f"({c:.5e})·ch^{deg - i}" for i, c in enumerate(poly_coeffs)
        )
        print(f"\nCalibration (degree-{deg} polynomial):")
        print(poly_str + "   [keV]")

        # residuals
        cal_fit_keV = np.polyval(poly_coeffs, cal_ch_arr)
        residuals = cal_keV_arr - cal_fit_keV
        print(f"\n  Calibration residuals (keV):")
        for r, res in zip(fit_results, residuals):
            print(f"    Peak {r['index']} ({r['energy_keV']:.3f} keV): {res:+.4f} keV")

    # ── Step 6: calibrated spectrum display ──────────────────────────────────
    fig_cal, ax_cal = plt.subplots(figsize=(14, 6))

    pos_mask = energy_axis > 0
    _semilogy_counts(
        ax_cal,
        energy_axis[pos_mask],
        counts[pos_mask],
        color="black",
        lw=1,
        label="Spectrum",
    )

    ax_cal.set_xlabel("Energy (keV)", fontsize=12)
    ax_cal.set_ylabel("Counts  (log scale)", fontsize=12)
    ax_cal.set_title("Calibrated Nuclear Spectrum", fontsize=13)
    e_last = energy_axis[last_nonzero]
    e_border = e_last * 0.02
    ax_cal.set_xlim(-e_border, e_last + e_border)

    # mark calibration peaks with vertical lines + labels
    # use blended transform: x in data coords, y in axes (0–1) coords
    trans = mtransforms.blended_transform_factory(
        ax_cal.transData, ax_cal.transAxes
    )
    for r in fit_results:
        ax_cal.axvline(r["energy_keV"], color="red", alpha=0.65, lw=1.0, ls="--")
        ax_cal.text(
            r["energy_keV"] + (ax_cal.get_xlim()[1] - ax_cal.get_xlim()[0]) * 0.003,
            0.97,
            f"{r['energy_keV']:.1f} keV",
            transform=trans,
            fontsize=8,
            color="darkred",
            rotation=90,
            va="top",
            ha="left",
        )

    fig_cal.tight_layout()
    _maximize(fig_cal)
    plt.savefig(os.path.join(out_dir, "calibrated_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nSaved to {out_dir}:")
    print("  peak_fits.png")
    print("  calibrated_spectrum.png")


if __name__ == "__main__":
    main()
