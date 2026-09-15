"""Artifact-only validation, volatility losses, uncertainty, and result reports."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import digest, write_json
from .models import STOCHASTIC


def losses(actual, predicted, previous):
    a,p,b = np.broadcast_arrays(np.asarray(actual,float),np.asarray(predicted,float),np.asarray(previous,float))
    if not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite scored predictions/targets")
    log_ratio=2*(a-p)
    with np.errstate(over="raise",invalid="raise"):
        qlike=np.expm1(log_ratio)-log_ratio
    return {"mse_log_rv":(a-p)**2,"mae_log_rv":np.abs(a-p),"mse_rv":(np.exp(a)-np.exp(p))**2,"mae_rv":np.abs(np.exp(a)-np.exp(p)),"qlike_variance":qlike,"directional_accuracy":(np.sign(p-b)==np.sign(a-b)).astype(float)}


def stationary_indices(n,reps=10000,block_length=6,seed=0):
    rng=np.random.default_rng(seed)
    indices=np.empty((reps,n),dtype=int)
    indices[:,0]=rng.integers(n,size=reps)
    for t in range(1,n):
        fresh=rng.integers(n,size=reps)
        indices[:,t]=np.where(rng.random(reps)<1/block_length,fresh,(indices[:,t-1]+1)%n)
    return indices


def load_validated(folder):
    folder=Path(folder)
    manifest=json.loads((folder/"manifest.json").read_text())
    config=manifest["config"]
    from .run import identity
    if identity(config)!=manifest["identity"]:raise ValueError("Manifest identity invalid")
    data=Path(config["data"])
    # Prefer the local published snapshot in a clone; retain the original path
    # for custom external datasets and execution provenance.
    local_data=Path(__file__).resolve().parents[1]/"data"/"snapshots"/data.parent.name/data.name
    if local_data.exists():data=local_data
    if digest(data)!=config["data_sha256"]:raise ValueError("Dataset changed")
    target=pd.read_csv(data,index_col=0,parse_dates=True).log_rv.loc[:config["end"]]
    dates=pd.date_range(config["start"],config["end"],freq="ME")
    all_dates=dates.append(pd.DatetimeIndex([dates[-1]+pd.offsets.MonthEnd()]))
    rows=[]
    for path in sorted((folder/"checkpoints").glob("*/*.json")):
        r=json.loads(path.read_text())
        if r["run_id"]!=manifest["identity"]:raise ValueError("Prediction belongs to another run")
        rows.append(r)
    if not rows and (folder/"predictions.csv").exists():
        receipt=json.loads((folder/"report_receipt.json").read_text())
        if digest(folder/"predictions.csv")!=receipt["prediction_sha256"]:
            raise ValueError("Published prediction checksum mismatch")
        rows=pd.read_csv(folder/"predictions.csv",float_precision="round_trip").to_dict("records")
        if any(r["run_id"]!=manifest["identity"] for r in rows):
            raise ValueError("Published predictions belong to another run")
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError("No checkpoint predictions")
    frame["target_month"]=pd.to_datetime(frame.target_month)
    keys=["model","seed","window","target_month"]
    if frame.duplicated(keys).any():raise ValueError("Duplicate forecasts")
    expected={(m,s,w,d) for m in config["models"] for s in (config["seeds"] if m in STOCHASTIC else [0]) for w in config["windows"] for d in all_dates}
    observed=set(frame[keys].itertuples(index=False,name=None))
    if expected!=observed:
        raise ValueError(f"Incomplete/mismatched run: missing {len(expected-observed)}, unexpected {len(observed-expected)}")
    for r in frame.itertuples():
        t=r.target_month
        if pd.Timestamp(r.forecast_origin)!=t-pd.offsets.MonthEnd():raise ValueError("Forecast origin mismatch")
        if pd.Timestamp(r.training_end)!=pd.Timestamp(r.forecast_origin):raise ValueError("Training reaches into target")
        if pd.Timestamp(r.training_start)!=t-pd.offsets.MonthEnd(r.window):raise ValueError("Training window mismatch")
        if r.configuration!="modern" or r.status not in {"ok","failed"}:raise ValueError("Invalid record contract")
        if t in target.index:
            if not np.isclose(r.actual_log_rv,target.loc[t],rtol=0,atol=1e-12):raise ValueError("Actual target mismatch")
        elif pd.notna(r.actual_log_rv):raise ValueError("Unobserved target has an actual value")
        previous=target.loc[t-pd.offsets.MonthEnd()]
        if not np.isclose(previous,r.previous_log_rv,rtol=0,atol=1e-12):raise ValueError("Previous target mismatch")
        if r.status=="ok" and not np.isfinite(r.predicted_log_rv):raise ValueError("Invalid successful forecast")
    frame.to_csv(folder/"predictions.csv",index=False)
    return frame,config


def table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)):return f"{x:.6g}"
        return str(x).replace("|","/")
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join("| "+" | ".join(fmt(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None))


def report(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from arch.bootstrap import MCS

    folder=Path(folder)
    frame,config=load_validated(folder)
    scored=frame.loc[frame.actual_log_rv.notna() & frame.status.eq("ok")].copy()
    values=losses(scored.actual_log_rv,scored.predicted_log_rv,scored.previous_log_rv)
    for name,value in values.items():scored[name]=value
    metric_names=list(values)
    end=pd.Timestamp(config["end"])
    final_year=end-pd.offsets.MonthEnd(11)
    summaries=[];seed_tables=[]
    for period,part in [("post2017",scored),("last12",scored.loc[scored.target_month>=final_year])]:
        n_dates=len(pd.date_range(config["start"] if period=="post2017" else final_year,end,freq="ME"))
        per_seed=part.groupby(["window","model","seed"])[metric_names].mean().reset_index()
        per_seed["period"]=period
        seed_tables.append(per_seed)
        for (w,m),g in part.groupby(["window","model"]):
            n_seeds=len(config["seeds"]) if m in STOCHASTIC else 1
            complete=len(g)==n_dates*n_seeds
            row={"period":period,"window":w,"model":m,"complete":complete,"successful_forecasts":len(g),"expected_forecasts":n_dates*n_seeds}
            row.update({name:g[name].mean() for name in metric_names})
            seed_mse=per_seed.loc[(per_seed.window==w)&(per_seed.model==m),"mse_log_rv"]
            row["mse_seed_std"]=float(seed_mse.std(ddof=0))
            summaries.append(row)
    summary=pd.DataFrame(summaries).sort_values(["period","window","mse_log_rv"])
    summary.to_csv(folder/"metrics.csv",index=False)
    pd.concat(seed_tables).to_csv(folder/"metrics_by_seed.csv",index=False)
    scored.to_csv(folder/"losses.csv",index=False)
    failures=frame.loc[frame.status.ne("ok")]
    failures.to_csv(folder/"failures.csv",index=False)
    stats=[];cis=[];common_results=[]
    for window in config["windows"]:
        eligible=summary.loc[(summary.period=="post2017")&(summary.window==window)&summary.complete,"model"].tolist()
        part=scored.loc[scored.window==window]
        # An incomplete model is excluded from the full-period ranking/MCS.
        # Also provide a separate all-model table on common successful dates.
        per_date=part.groupby(["target_month","model"])[metric_names].mean()
        complete_seed_counts=part.groupby(["target_month","model"]).size()
        valid_keys=[key for key,n in complete_seed_counts.items() if n==(len(config["seeds"]) if key[1] in STOCHASTIC else 1)]
        valid_per_date=per_date.loc[valid_keys]
        shared=valid_per_date.reset_index().groupby("target_month").model.nunique()
        shared=shared.index[shared==len(config["models"])]
        if len(shared):
            g=valid_per_date.loc[valid_per_date.index.get_level_values(0).isin(shared)].groupby("model").mean().reset_index()
            g["window"]=window;g["common_months"]=len(shared)
            common_results.append(g)
        for metric in ["mse_log_rv","qlike_variance"]:
            matrix=per_date[metric].unstack("model").loc[:,eligible]
            if matrix.isna().any().any():raise ValueError("Unequal dates in inference")
            mcs=MCS(matrix,size=.05,reps=10000,block_size=6,method="R",bootstrap="stationary",seed=0)
            mcs.compute()
            pvals=mcs.pvalues.reset_index()
            pvals.columns=["model","mcs_pvalue"]
            pvals["window"]=window;pvals["loss"]=metric
            stats.append(pvals)
            if "HAR" in matrix:
                difference=matrix.subtract(matrix.HAR,axis=0)
                idx=stationary_indices(len(matrix))
                sampled=difference.to_numpy()[idx].mean(axis=1)
                lo,hi=np.quantile(sampled,[.025,.975],axis=0)
                for i,m in enumerate(matrix.columns):cis.append({"window":window,"model":m,"loss":metric,"mean_loss_minus_HAR":difference[m].mean(),"ci_lower":lo[i],"ci_upper":hi[i],"months":len(matrix)})
    pd.concat(stats).to_csv(folder/"mcs.csv",index=False)
    pd.DataFrame(cis).to_csv(folder/"loss_difference_ci.csv",index=False)
    if common_results:pd.concat(common_results).to_csv(folder/"common_dates_metrics.csv",index=False)
    live=frame.loc[frame.actual_log_rv.isna() & frame.status.eq("ok")].copy()
    live["forecast_monthly_volatility_percent"]=np.exp(live.predicted_log_rv)*100
    live.to_csv(folder/"unscored_forecasts.csv",index=False)
    last=scored.loc[scored.target_month>=final_year]
    last.to_csv(folder/"last12_forecasts.csv",index=False)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,constrained_layout=True)
    for ax,w in zip(axes,config["windows"]):
        sub=last.loc[last.window==w]
        actual=sub.groupby("target_month").actual_log_rv.first()
        ax.plot(actual.index,np.exp(actual)*100,color="black",linewidth=2.5,label="Observed")
        for name in ["QR1","QR2","HAR","LSTMX","CRLX"]:
            g=sub.loc[sub.model==name]
            if g.empty:continue
            # Plot the average volatility forecast; statistical tests average losses.
            pred=g.assign(rv=np.exp(g.predicted_log_rv)*100).groupby("target_month").rv.mean()
            ax.plot(pred.index,pred,label=name,alpha=.85)
        ax.set_title(f"{w}-month training window")
        ax.set_ylabel("Monthly volatility (%)")
        ax.grid(alpha=.2)
        ax.legend(ncol=3,fontsize=9)
    fig.suptitle("S&P 500: observed and one-month-ahead forecasts",fontsize=15)
    axes[-1].set_xticks(actual.index)
    axes[-1].set_xticklabels([d.strftime("%b\n%Y") for d in actual.index],fontsize=9)
    fig.savefig(folder/"last12_forecasts.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,w in zip(axes,config["windows"]):
        g=summary.loc[(summary.period=="post2017")&(summary.window==w)&summary.complete].sort_values("mse_log_rv",ascending=False)
        ax.barh(g.model,g.mse_log_rv,color=["#7254ad" if m.startswith("QR") else "#427e9a" for m in g.model])
        ax.set_title(f"{w}-month window: Jan 2018–Aug 2026")
        ax.set_xlabel("Mean squared error on log volatility")
    fig.savefig(folder/"model_comparison.png",dpi=180)
    plt.close(fig)
    text=["# Current-market quantum reservoir results", "",f"Evaluation: **{config['start']} through {config['end']}**. Final-year case study: **{final_year:%Y-%m} through {end:%Y-%m}**.","","## Results", "", "Models below have every required scored prediction. Results average losses across seeds; seeds are not extra market observations. ARMAX or other incomplete results are reported separately, with a common-date comparison where available."]
    for w in config["windows"]:
        g=summary.loc[(summary.period=="post2017")&(summary.window==w)&summary.complete].set_index("model")
        winner=g.mse_log_rv.idxmin()
        pieces=[f"**{w}-month window:** {winner} has the lowest mean log-RV MSE ({g.loc[winner,'mse_log_rv']:.5f})."]
        if "HAR" in g.index:
            for m in ["QR1","QR2"]:
                if m in g.index:
                    change=100*(g.loc[m,"mse_log_rv"]/g.loc["HAR","mse_log_rv"]-1)
                    pieces.append(f"{m} MSE is {abs(change):.1f}% {'higher' if change>0 else 'lower'} than HAR.")
        text.extend([""," ".join(pieces)])
    text.extend(["","### Final-year interpretation",""])
    for w in config["windows"]:
        g=summary.loc[(summary.period=="last12")&(summary.window==w)&summary.complete].set_index("model")
        best=g.mse_log_rv.idxmin()
        pieces=[f"With {w} months of training, **{best}** has the lowest final-year log-RV MSE ({g.loc[best,'mse_log_rv']:.5f})."]
        for m in ["QR1","QR2"]:
            if m in g.index:pieces.append(f"{m}: {g.loc[m,'mse_log_rv']:.5f}.")
        text.extend([" ".join(pieces),""])
    text.extend(["The quantum models are competitive in the longer retrospective comparison, but neither has the lowest error in the final twelve months. The reported MCS results do not establish a uniquely superior quantum model. Differences across the two training windows describe this fixed experiment and do not justify selecting a window after observing the final year.",""])
    for period in ["post2017","last12"]:
        text.extend(["",f"### {period}","",table(summary.loc[(summary.period==period)&summary.complete,["window","model","mse_log_rv","mae_log_rv","qlike_variance","directional_accuracy","mse_seed_std"]])])
    text.extend(["","Lines show mean volatility predictions across seeds; loss tables average each seed's loss.","","![Forecast comparison](last12_forecasts.png)","","![Full-period model comparison](model_comparison.png)","","## Statistical evidence","","The Model Confidence Set uses 10,000 stationary-bootstrap replications, expected block length six months, and seed zero. A p-value of one does not establish unique superiority. Confidence intervals in `loss_difference_ci.csv` compare mean per-date loss against HAR, using paired resampling of dates and averaging seed losses before resampling. These intervals describe temporal uncertainty conditional on the chosen five seeds; they are not adjusted for all pairwise comparisons. Use MCS for the familywise comparison.","",table(pd.concat(stats)),"","## Data, timing, and adaptation","","The dataset was rebuilt from daily S&P 500 closes. Monthly realized volatility is the square root of summed squared daily log returns. Forecast month t uses three completed monthly input vectors ending at t−1. Targets use log volatility; QLIKE compares positive variances with no absolute-log substitution. Input scaling is calibrated through 2017, frozen, and clipped; targets are never clipped. All models receive the same training target dates within each window. Deterministic models run once; reservoir/LSTM models use seeds 0–4.","","The seven inputs are log RV, trailing three- and twelve-month mean log RV, monthly log return, downside variance share, maximum absolute daily return, and log close range. QR1 and QR2 use identical inputs and paired coupling matrices; only temporal readout count changes. Economic predictors are replaced explicitly. These are modern extensions, not unchanged author models.","","The 571-month window preserves the original training horizon. The 120-month window tests more recent estimation history. Both are refitted each month. Neither window is selected using final-year performance. Twelve monthly forecasts are a descriptive case study, not strong standalone evidence of superiority.","","Snapshots preserve Yahoo/FRED payloads and hashes. FRED-sourced closes replace seven material provider discrepancies in the canonical series; every change is in `price_corrections.csv`. January 1950 is the sole legacy target difference above 1e-8; see the audit report. The initial partial return month is discarded before calculating rolling features.","","## Execution and limitations","",f"Recorded forecasts: {len(frame)}; failed model/origin records: **{len(failures)}**. The failed records and their errors are in `failures.csv`. No model substitutes another model's predictions.","","A forecast for September 2026, based only on information through August 31, is saved in `unscored_forecasts.csv`. September daily observations remain in the data snapshot but are excluded from training and scoring. These are retrospective simulations using the retrieved historical data, not forecasts recorded live at their historical origins.","","The quantum models use ideal local density-matrix simulation and exact expectation values. This study makes no quantum-hardware speedup or trading-profit claim. Legacy replication, including historical loss definitions, lives in the separate legacy run.","","## Reproduce","","Run `python run_study.py report --run " + str(folder) + "` to regenerate this report from checkpoints. `manifest.json` records source hashes, dataset identity, versions, seeds, dates, and protocol; resume rejects a changed identity. `metrics_by_seed.csv`, `losses.csv`, `mcs.csv`, and the prediction files supply the underlying evidence.","","Sources: [base paper](https://arxiv.org/html/2505.13933v2), [FRED S&P 500](https://fred.stlouisfed.org/series/SP500), [MCS documentation](https://bashtage.github.io/arch/multiple-comparison/multiple-comparison_examples.html)."])
    (folder/"report.md").write_text("\n".join(text)+"\n")
    import mistune
    body=mistune.create_markdown(plugins=["table"])((folder/"report.md").read_text())
    style="""body{margin:0;background:#f4f6f9;color:#172537;font:16px/1.65 system-ui,sans-serif}main{max-width:1100px;margin:36px auto;padding:40px;background:white;border-radius:12px}h1{font-size:34px;line-height:1.2;color:#16324f}h2{margin-top:40px;border-top:1px solid #dce3eb;padding-top:24px}h3{color:#314b69}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}th,td{padding:8px 10px;border-bottom:1px solid #e1e6ed;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}tr:nth-child(even){background:#f8fafc}img{width:100%;height:auto}a{color:#265e9e}code{background:#eef2f7;padding:2px 5px;border-radius:4px}@media(max-width:800px){main{margin:0;padding:20px}table{display:block;overflow:auto}}"""
    (folder/"report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Current-market quantum reservoir results</title><style>'+style+'</style><main>'+body+'</main></html>')
    write_json(folder/"report_receipt.json",{"prediction_sha256":digest(folder/"predictions.csv"),"report_sha256":digest(folder/"report.md"),"report_source_sha256":digest(__file__),"records":len(frame),"failures":len(failures),"scored_months":len(scored.target_month.unique()),"mcs_reps":10000,"block_length":6,"seed":0})
    print(f"Report: {folder/'report.md'}; {len(frame)} records, {len(failures)} failures")
