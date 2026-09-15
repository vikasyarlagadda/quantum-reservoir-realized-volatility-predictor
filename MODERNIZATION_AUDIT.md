# Modernization audit

## What the project does

The original study predicts next month's S&P 500 realized volatility. A fixed
quantum system transforms three past monthly observations into nonlinear
features; a fitted ridge regression produces the forecast. It simulates ten
qubits: seven input qubits and three hidden qubits. The hidden state is reset
for each three-month example, not carried continuously across the entire sample.
QR1 reads ten Z expectations once; QR2 reads them at two times, giving twenty
features. Only the regression readout is trained. This is local classical
simulation of quantum dynamics.

The comparison includes HAR/HARX and autoregressive statistical models, LSTM
neural networks, and classical echo-state reservoirs. Statistical and quantum
models forecast log volatility; modern evaluation also converts those forecasts
to positive volatility and variance.

## Work already present

Git history at starting commit `85208b5` records the following progression:

- April 10–11: input preprocessing and notebook execution repairs.
- April 12: prediction-scale correction, paper hyperparameter/methodology
  alignment, documentation, and saved comparison results.
- April 15–16: Python/Qiskit implementation, Trotter circuit implementation,
  repository organization, and path fixes.
- April 19–20: checked-in prediction, plot, and statistical artifacts and merges.
- September 10: a repository audit and semester research plan, with no new
  training recorded in that planning milestone.

The history attributes the Python quantum circuit introduction to `aidanccc`;
Vikas's commits include preprocessing, notebook fixes, scale/methodology
alignment, organization, and results. The work is collaborative.

## Problems addressed by the modern pipeline

1. The source CSV is already transformed, and ends in December 2017. Appending
   new raw values would mix incompatible scales.
2. Notebook and quantum-driver sample lengths are hard-coded to 816 total rows
   and 245 forecasts. The master notebook consumes the author prediction file.
3. Preprocessing chooses differencing using the full historical sample. The
   original scaling and economic predictor release timing lack full provenance.
4. Legacy QLIKE uses absolute normalized/log values rather than positive
   realized variances. Direction accuracy sometimes compares successive
   predictions instead of the forecast with the last observed actual value.
5. Saved Python/reference quantum agreement (about 1.2e-6) does not establish
   clean execution provenance or exact reproduction of all published analyses.
6. Saved DM p-values for QR2 versus QR1/HARX/ARMAX are .780297/.847091/.983356.
   These contradict the previous README's blanket superiority claim. MCS
   p-value one is not proof of a uniquely best model.
7. Original QR1 and QR2 change both inputs and coupling matrices. Their direct
   comparison does not isolate the number of temporal readouts.

The modern runner addresses these with dated records, a frozen pre-2018 scaler,
causal price features, fixed training horizons, paired quantum couplings,
corrected losses, common-date comparisons, and run identity checks. Legacy
code is executed separately so its original behavior remains reviewable.

## Data acquisition and reconciliation

The canonical snapshot is `data/snapshots/2026-09-14-v2/`. It contains original
Yahoo and FRED payloads, retrieval timestamps, hashes, daily closes, monthly
features, the full historical comparison, and every price correction.

Yahoo returned 19,317 daily observations from December 1, 1949 through
September 14, 2026 when queried with explicit timestamps and `interval=1d`.
The `range=max` endpoint instead downsampled the series, so it is rejected.
Daily log returns are computed before monthly aggregation, retaining the
preceding month's close for the first trading-day return.

### January 1950 boundary resolved

The bundled January 1950 log RV is **-3.4822828507512336**; the reconstructed
value is **-3.4512888011515233**. Dropping the first January return from the
reconstruction gives **-3.482282850751236**, matching the bundled target.
That return is **-0.007772898037821108**. Thus the legacy boundary omits the
first cross-month return. The modern series retains it consistently. All other
815 historical monthly targets agree within 1e-8. This is a verified boundary
explanation, not a fitted adjustment to improve forecasting results.

### Calendar and provider differences

Exchange-calendar validation initially reported pre-1970 holiday omissions.
The calendar's regular historical holiday rules produce those dates correctly,
but pandas' default holiday horizon starts at 1970. Explicitly subtracting the
calendar's historical regular holidays resolves the issue; no unexplained
missing sessions remain in the snapshot.

FRED overlaps 2,512 daily closes. Seven observations differ from Yahoo by more
than 0.011 index points. The canonical series uses FRED on those dates because
FRED identifies the index provider as its source. This is a transparent provider
precedence decision, not a claim that every historical revision is known.
Both original feeds and the correction ledger remain available. The largest
difference is 5.28984375 points on August 11, 2021. All seven differences precede
the final twelve-month case study.

An initial processed snapshot and two small diagnostic pilots are retained
under their original names. The v2 snapshot excludes the first return-incomplete
month **before** computing rolling features; this leaves 909 valid feature
months starting December 1950. Only v2 enters the final modern run.

## Modern experiment choices

- Forecast dates: January 2018–August 2026 (104 scored months), plus one unscored
  September 2026 forecast using information only through August 31.
- Inputs: log RV; trailing three/twelve-month mean log RV; cumulative monthly
  log return; downside share of squared returns; largest absolute daily return;
  and log maximum/minimum daily close ratio.
- Calibration: min–max feature scaling fit through December 2017, frozen for
  all later origins. Out-of-range feature values clip to [-1,0], with clipping
  fractions reported. Target scaling is never clipped.
- Windows: 571 and 120 valid monthly target observations, shared across model
  families. Pre-window history supplies lagged inputs. Original neural notebook
  windows instead contain 571 rows before dropping three sequence warmup rows;
  that behavior remains only in the legacy run.
- HAR uses the three volatility features; HARX uses all seven, avoiding
  duplicate HAR regressors. AR1/AR3 use lagged un-clipped target values and OLS
  with intercept. ARMAX preserves ARIMA(3,0,0), no trend, with lagged exogenous
  inputs. It is an AR model with exogenous regressors and autoregressive errors,
  despite the legacy ARMAX name and absence of moving-average terms.
- The ARMAX numerical iteration ceiling is increased from the library default
  to 1,000 because the default failed the pilot. Unconverged fits remain explicit
  failures. No alternative algorithm fills those predictions.
- LSTMs preserve two layers, hidden sizes 60/50, Adam .001, batch size 64,
  three-month sequences, and 100 epochs. Each fit starts fresh. Deterministic
  per-origin seeds make resumed fits independent of scheduling order.
- Classical reservoirs preserve sizes 50/20, leak .6, spectral radius .9,
  input scaling .1, ridge 1e-7, and reset-per-sequence behavior.
- Quantum reservoirs preserve ten qubits, tau=1, memory=3, and ridge 1e-8.
  Modern QR1/QR2 share seven inputs and the same seeded coupling matrix within
  each of seeds 0–4. This differs deliberately from the original QR1/QR2 setup.
- All new model selection is frozen; neither the final year nor post-2017 scores
  choose hyperparameters, a winning seed, or a preferred training window.

## Evaluation interpretation

Primary loss is MSE on log RV. Modern QLIKE is `expm1(2*(actual_log-pred_log))
- 2*(actual_log-pred_log)`, equivalent to the positive-variance ratio formula.
Exponentiated log forecasts are plug-in volatility forecasts; no test-set
smearing or bias correction is fitted. Direction is relative to the last
observed month. Five random seeds are reported separately and averaged at each
date for statistical comparisons, not counted as five independent months.

Full-coverage models form the primary full-period ranking/MCS. Models with
failed fits are omitted from that ranking and reported in a separate table
restricted to dates on which every model and seed succeeded. No successful
model is evaluated on easier dates without disclosing the comparison universe.
MCS and paired temporal uncertainty use 10,000 stationary-bootstrap draws,
expected block length six months, seed zero. The twelve-month view is descriptive.

## Reproducibility and limitations

Run identities bind dataset hashes, model source hashes, versions, scaler,
dates, seeds, and configurations. Completed modern origins checkpoint atomically.
Deterministic seed derivation allows resume without depending on prior RNG calls.
Reusable quantum/classical feature arrays include checksums. Cache artifacts are
excluded from Git because they can be regenerated; they remain on disk.

These are retrospective tests on retrieved historical prices, not archived
live forecasts, trading strategies, or quantum hardware experiments. The fixed
encoding can saturate under novel market conditions; clipping is reported.
Replacing economic predictors changes the experiment, so the modern results
are not an exact extension of the author's input design. The separate legacy
execution measures what the existing code reproduces, not unpublished author
feature searches, Julia dependencies, or hardware performance.

Sources: [paper](https://arxiv.org/html/2505.13933v2),
[FRED S&P 500](https://fred.stlouisfed.org/series/SP500),
[MCS documentation](https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html).

## GitHub publication policy

Publish the implementation, frozen configuration, derived monthly features,
source/retrieval manifests, reconciliation ledgers, aggregate date-indexed
predictions, metrics, plots, statistical results, and validation receipts.
Raw provider payloads, daily price files, and full daily provider comparisons
remain local; the FRED source identifies restrictions on redistribution of the
index series. Acquisition code and hashes preserve provenance without uploading
the full raw price history.

Per-origin checkpoint files, operator/feature caches, isolated copies of legacy
notebooks, and diagnostic pilot/resume directories remain on disk but are not
committed. Their role is execution/resume support, and thousands of small
checkpoint files would obscure a review. Published aggregate predictions retain
every successful and failed origin. The report loader can validate and regenerate
results from the published prediction CSV and its checksum receipt when local
checkpoints are absent. The first diagnostic snapshot remains local and is
explicitly superseded by v2, not silently discarded.
