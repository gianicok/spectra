# Spectra Analyzer

Interactive nuclear spectrum analysis tool. Load a pulse-height histogram, graphically select peaks, fit Gaussians, and get a calibrated energy spectrum.

![Demo](spectra.gif)

## Install

Requires Python 3.9+ with numpy, matplotlib, scipy, and pandas.

```
git clone https://github.com/gianicok/spectra.git
cd spectra
pip install -e .
```

This adds `spectra` as a command available anywhere in your terminal.

## Usage

```
spectra path/to/spectrum.csv
```

The CSV should be a single column of counts (one bin per row, header optional). The row index is the channel number.

## Workflow

1. **Select peaks** — the spectrum opens in a zoomable window. Shift+click to mark a peak centroid. Shift+right-click to undo. Press Enter when done.
2. **Enter energies** — for each marked peak, type its known energy in keV in the terminal. The plot stays open and highlights the current peak.
3. **Review fits** — a window shows the Gaussian + linear background fit for each peak, with centroid, FWHM, resolution (%), and net area printed to the terminal.
4. **Calibrated spectrum** — the final spectrum is displayed in keV and saved alongside the fits.

## Output

Results are saved to `output/<filename>/` next to the script:

- `peak_fits.png` — individual peak fits with background subtraction
- `calibrated_spectrum.png` — final spectrum in keV
