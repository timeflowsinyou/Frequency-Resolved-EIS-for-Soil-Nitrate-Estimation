# Main Figure Plan

All main figures must use public scientific names. Internal experiment identifiers and development-stage labels must not appear in titles, legends, filenames, or captions.

## Figure 1 — Frequency-Resolved Impedance Information and CNN Input Design

Figure 1 is limited to two panels.

### Panel A — Full-spectrum measured response

Show representative measured impedance magnitude and phase across the complete 20 Hz–1 MHz range for all six concentration stages.

Recommended design:

- logarithmic frequency axis;
- consistent concentration color scale;
- measured technical repeats shown with low opacity or summarized by a band;
- magnitude and phase aligned vertically inside the same panel;
- no smoothing or interpolation that changes the measured frequency grid.

Scientific message:

> Concentration information is distributed across broadband impedance level, phase, and frequency-dependent spectral shape.

### Panel B — Input-range evidence

Compare information retention or controlled CNN performance for:

- all 201 measured frequencies;
- reduced points distributed across the complete frequency range;
- a contiguous low-frequency subset.

Use the same metric definition and evaluation population for every input design. Clearly indicate that frequency reduction and network parameter count may be partially confounded where applicable.

Scientific message:

> Complete-range sampling preserves more useful spectral structure than contiguous low-frequency truncation, particularly for the spectral level/shape representation.

## Figure 2 — Feature Representation Comparison and Model Selection

Figure 2 is limited to two panels. No separate model-complementarity scatter panel is required.

### Panel A — Physical representation schematic

Show the two retained representations:

1. conductance and susceptance derived from complex admittance;
2. broadband impedance level, centered spectral shape, and phase.

Corrected density and water content should be shown as a separate physical-condition input rather than repeated across every frequency.

### Panel B — Frozen reference-seed performance

Compare:

- measured-data admittance CNN;
- matched replay control;
- physics-guided augmentation;
- measured-data spectral level/shape CNN;
- fixed equal-weight representation ensemble.

Report both coefficient of determination and RMSE. Highlight the ensemble result without hiding unfavorable augmentation results.

## Figure 3 — Experimental Design and Model Architecture

Show:

- 19 soil specimens;
- specimen-grouped five-fold cross-validation;
- all stages and repeats from one soil retained within one fold;
- training-only preprocessing;
- 201-frequency EIS branches;
- separate corrected-density and corrected-water-content branch;
- two frozen CNN representations;
- fixed equal-weight prediction ensemble;
- exclusively measured validation observations.

## Figure 4 — Multi-Seed Stability

Use two panels:

- Panel A: 19-soil grouped-CV RMSE across ten network seeds;
- Panel B: historical Z-transfer RMSE across the same seeds.

Show every seed as an individual point. Connect models trained with the same seed where legibility permits. Overlay the mean and sample standard deviation and identify the reference seed without visually privileging it as the selected best seed.

## Figure 5 — Transfer Prediction: Predicted vs. Nominal

Use one panel for each historical Z-series soil. All panels must share axis limits.

Show:

- nominal nitrate-N on the horizontal axis;
- predicted nitrate-N on the vertical axis;
- identity line;
- individual technical measurements with transparency;
- seed-level or across-seed summary predictions;
- specimen-level RMSE and signed bias.

The caption must state that the specimens are historically held out for the frozen full-cohort fit but are not a pristine prospective external cohort.

## Figure 6 — Concentration-Stage Error and Endpoint Bias

Use two panels:

- Panel A: grouped-CV signed error by concentration stage;
- Panel B: historical Z-transfer signed error by concentration stage.

Plot the ten-seed mean and seed-level variability. Include a zero-error reference line. Emphasize persistent low-concentration overestimation and high-concentration underestimation without presenting stage-specific coefficient of determination for near-constant targets.

## Visual consistency

- Use one color per public model name across all figures.
- Use accessible colors and verify grayscale legibility.
- State nitrate-N units on every concentration or error axis.
- Keep full computational precision in source tables; round only displayed labels.
- Do not use internal experiment codes.
- Do not omit difficult soils or unfavorable seeds.

