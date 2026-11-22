# Public measurement data

## Raw files

`data/raw/` contains the original MATLAB EIS measurement files for 24 soil specimens plus `f.mat`, which stores the repeated 201-point frequency grid.

Each specimen file contains an `Imp` structure with:

- `Imp.Z`: impedance magnitude in ohms, shape `[201 frequencies, 20 repeats, 6 stages]`;
- `Imp.phi`: phase in degrees with the same shape.

The frequency range is 20 Hz to 1 MHz. The loader verifies that all 20 stored frequency sweeps are identical and strictly increasing.

## Cohorts

- 21 original soils were prepared through sequential concentration increments on the same soil material.
- 3 historical Z-series soils were divided into six aliquots and prepared directly at the target concentrations.

Preparation protocol is completely associated with cohort identity. The data cannot isolate a causal preparation-protocol effect.

## Deterministic correction

One confirmed instrument sentinel (`9.9e37`) occurs in the `V18%_01.mat` magnitude and phase arrays. `scripts/build_dataset.py` replaces this single point using same-spectrum interpolation in log-frequency coordinates. The original raw file is preserved unchanged, and the correction record is embedded in the generated processed dataset.

Confirmed density and water-content metadata corrections are declared explicitly in `src/frequency_eis/metadata.py`.

## Building the processed dataset

```powershell
python scripts/build_dataset.py
```

The generated file is written to `data/processed/measured_eis_dataset.npz` and is intentionally not tracked because it is reproducible from the public raw files.

## License

The raw measurement files are released under CC BY 4.0. See `DATA_LICENSE.md`.

