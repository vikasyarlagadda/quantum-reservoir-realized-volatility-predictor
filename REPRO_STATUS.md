# Reproduction status

## Completed — September 14, 2026 market cutoff

- Downloaded 19,317 daily observations through September 14, 2026; scored
  complete months through August 2026. Original raw payloads remain local.
- Reconciled January 1950 to an omitted first return. The other 815 historical
  targets agree within 1e-8. Seven provider discrepancies have a correction ledger.
- Completed all 7,560 modern model/seed/window/origin records: 7,558 successful
  forecasts and two explicitly failed ARMAX fits (571-month window, January 2019
  and August 2020). Seventy-two forecasts for September 2026 remain unscored.
- Completed the isolated 11-model legacy benchmark: 2,695 predictions over 245
  months. Fresh quantum predictions match the author reference within 1.2e-6.
- Modern execution took 1,167.7 seconds (19.5 minutes); the concurrent legacy
  execution took 1,332.1 seconds (22.2 minutes), on the existing local environment.
- Ten tests pass, including future-data isolation, target/sequence alignment,
  losses, published-result integrity, and deterministic LSTM reproduction.
- Actual interrupted/resumed execution matched uninterrupted predictions exactly.
  Portable report regeneration without raw prices/checkpoints produced byte-identical
  predictions, metrics, MCS, confidence intervals, and forecast exports.
- All result-notebook code cells executed, both plots were visually inspected,
  and the dependency consistency check passed.

## Results

For September 2025–August 2026, HAR has the lowest log-RV MSE with 571 training
months (0.08551); LSTMX leads with 120 months (0.06770). QR1/QR2 score
0.09424/0.08891 with the long window and 0.10371/0.11147 with the short window.
Neither quantum model leads the final year.

Across January 2018–August 2026, QR1 ranks second by mean log-RV MSE in both
windows. With 571 months, CRLX scores 0.17099, QR1 0.17139, and QR2 0.17602.
With 120 months, AR1 scores 0.18236, QR1 0.18587, and QR2 0.21650. These ranks
do not establish unique statistical superiority; see the MCS and uncertainty tables.

The two ARMAX failures are excluded transparently: the full-period 571-month
ranking omits ARMAX, and a separate all-model comparison uses 102 common months.
The 120-month comparison has all 104 months. The final twelve months have no failures.

## Artifacts and source

- [Modern report](results/modern-2026-09-14/report.md)
- [HTML report](results/modern-2026-09-14/report.html)
- [Legacy results](results/legacy-2026-09-14/legacy_metrics.csv)
- [Validation receipt](results/validation_receipt.json)
- [Audit](MODERNIZATION_AUDIT.md), [assumptions](ASSUMPTIONS.md), [run order](RUN_ORDER.md)

Executed model sources match commit `6e7ddf0`. The subsequent exact-simulator
change was verified to be whitespace only. Per-run execution revision receipts
document strict resume requirements.
Raw downloads, caches, checkpoint files, and diagnostic runs are preserved locally;
publication includes derived data, provenance, complete aggregate results and reports.

## Personal repository review

Development and publication now use only
`vikasyarlagadda/quantum-reservoir-realized-volatility-predictor`. The unwanted
external pull request was closed and its feature branch deleted; the external
remote was removed. Historical source attribution and immutable execution
receipts are preserved. The personal PR review rechecked forecast alignment,
frozen scaling, loss formulas, failure handling, and saved-artifact validation.
All ten tests and the dependency consistency check passed again before merge.
