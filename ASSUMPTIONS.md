# Modern experiment assumptions

These choices are explicit extensions of the legacy reproduction. Details and
evidence are in [the audit](MODERNIZATION_AUDIT.md); original historical notes
remain in `docs/ASSUMPTIONS.md`.

1. **Forecast objective:** monthly S&P 500 log realized volatility; January
   2018–August 2026 is the retrospective evaluation. September 2026 is unscored.
2. **Current public inputs:** seven price-derived features replace unavailable
   or delayed economic predictors, as approved in the plan. The new results are
   not exact author-feature replication.
3. **Price precedence:** use FRED on overlapping closes differing by more than
   0.011 index points; preserve both raw feeds and record each correction. This
   favors the index-provider-sourced feed without claiming historical vintages.
4. **Boundary:** include the cross-month first daily return consistently. The
   original January 1950 omitted it; the difference is numerically reconciled.
5. **Warmup/calendar:** discard the first return-incomplete month before rolling
   features. Explicitly apply pre-1970 NYSE regular holidays to work around the
   default pandas holiday horizon. No missing sessions are zero-filled.
6. **Normalization:** one calibration through December 2017, fixed thereafter.
   Clip input features to [-1,0] and report saturation; never clip targets.
7. **Fair training:** 571 or 120 valid monthly training targets per model; prior
   history may supply their lagged inputs. Legacy notebooks retain their own
   sequence warmup convention separately.
8. **Hyperparameters:** preserve original architectural choices and freeze
   them before post-2017 evaluation. Increase ARMAX optimizer iterations to
   1,000 for numerical convergence; retain failures when still unconverged.
9. **Paired reservoirs:** modern QR1 and QR2 use identical input definitions and
   paired seeded couplings. Seeds 0–4 are fixed before scoring; stochastic
   results are not selected by winning seed.
10. **Estimation target:** exponentiated log predictions are plug-in volatility
    estimates; no final-test bias correction is fitted. QLIKE uses their squared
    values against observed realized variance.
11. **Statistical universe:** complete models use the full period; incomplete
    models appear in a separately labeled common-date comparison. Average seed
    losses by date for inference, and report seed variation separately.
12. **Claim limits:** local ideal quantum simulation, retrospective data, and a
    twelve-month descriptive view do not establish hardware advantage, live
    trading performance, or uniquely superior forecasting.
