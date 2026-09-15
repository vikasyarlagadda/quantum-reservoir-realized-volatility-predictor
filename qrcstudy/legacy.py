"""Execute the unchanged academic workflow in an isolated artifact directory."""
import contextlib
import json
import os
from pathlib import Path
import runpy
import shutil
import time

import numpy as np
import pandas as pd

from .data import digest, write_json


def run_legacy(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from IPython.display import display
    from .run import checked_manifest
    from quantum_reservoir_qiskit import MIN_RV, DIF, load_coupling_matrices

    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    source_files = ["preprocess.py", "run_qrc_simulation.py", "quantum_reservoir_qiskit.py", "quantum_reservoir_trotter.py", "LSTM.ipynb", "classical_reservoir.ipynb", "Reservoir_Learning.ipynb", "data/Data.CSV", "data/coeff_10.jld2", "data/predict_result.csv"]
    config = {"protocol": "legacy-executed-v1", "seed": 0, "source_hashes": {p:digest(root/p) for p in source_files}, "runner_sha256": digest(__file__), "changes": ["isolated output paths", "fixed numpy/torch/reservoir/MCS seeds", "master loads freshly simulated quantum predictions"]}
    run_id = checked_manifest(output,config)
    if (output / "complete.json").exists():
        print("Legacy execution already complete; preserved artifacts reused")
        return
    work = output / "workspace"
    for path in source_files:
        dest = work/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(root/path,dest)
    for path in ["results/predictions/LSTM", "results/predictions/Classical_Reservoir_learning", "results/plots", "results/stats"]:
        (work/path).mkdir(parents=True,exist_ok=True)
    if load_coupling_matrices(str(work/"data/coeff_10.jld2")) is None:
        raise ValueError("Original coupling matrices required")
    previous = Path.cwd()
    start = time.perf_counter()
    stages=[]
    torch.set_num_threads(2)
    try:
        os.chdir(work)
        with (output/"execution.log").open("a",buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            for script in ["preprocess.py", "run_qrc_simulation.py"]:
                done=output/(script+".done.json")
                if not done.exists():
                    ts=time.perf_counter()
                    runpy.run_path(script,run_name="__main__")
                    write_json(done,{"seconds":time.perf_counter()-ts})
            original=pd.read_csv("data/predict_result.csv")
            fresh=pd.read_csv("results/predictions/qrc_predict_result.csv")
            verification={c:float((fresh[c]-original[c]).abs().max()) for c in original}
            write_json(output/"quantum_reference_comparison.json",verification)
            for notebook in ["LSTM.ipynb","classical_reservoir.ipynb","Reservoir_Learning.ipynb"]:
                np.random.seed(0)
                torch.manual_seed(0)
                torch.use_deterministic_algorithms(True)
                namespace={"__name__":"__legacy_notebook__","display":display}
                nb=json.loads(Path(notebook).read_text())
                ts=time.perf_counter()
                print(f"EXECUTING {notebook}",flush=True)
                for i,cell in enumerate(nb["cells"]):
                    if cell["cell_type"]!="code":continue
                    code="".join(cell["source"])
                    if notebook=="classical_reservoir.ipynb":
                        code=code.replace("input_scaling=0.1)","input_scaling=0.1, seed=0)")
                    if notebook=="Reservoir_Learning.ipynb":
                        code=code.replace('pd.read_csv("data/predict_result.csv",','pd.read_csv("results/predictions/qrc_predict_result.csv",')
                        code=code.replace("bootstrap='stationary')","bootstrap='stationary', seed=0)")
                    print(f"CELL {i}",flush=True)
                    exec(compile(code,f"{notebook}:cell{i}","exec"),namespace)
                    plt.close("all")
                stages.append({"notebook":notebook,"seconds":time.perf_counter()-ts})
                if notebook=="Reservoir_Learning.ipynb":
                    dates=pd.read_csv("data/Data.CSV",index_col=0,parse_dates=True).index
                    target=pd.read_csv("data/Data.CSV",index_col=0).RV.to_numpy()
                    names={"HAR":"har","HARX":"harx","AR1":"ar1","AR3":"ar3","ARMAX":"armax","LSTM":"lstm","LSTMX":"lstmx","CRL":"crl","CRLX":"crlx","QR1":"qr1","QR2":"qr2"}
                    rows=[]
                    for model,key in names.items():
                        pred=np.asarray(namespace["predictions_"+key]).reshape(-1)
                        if len(pred)!=245 or not np.isfinite(pred).all():raise ValueError(f"Invalid legacy {model} predictions")
                        for j,v in enumerate(pred):
                            t=571+j
                            rows.append({"run_id":run_id,"configuration":"legacy","model":model,"seed":0,"window":571,"target_month":str(dates[t].date()),"forecast_origin":str(dates[t-1].date()),"training_start":str(dates[t-571].date()),"training_end":str(dates[t-1].date()),"actual_log_rv":(target[t]+1)*DIF+MIN_RV,"previous_log_rv":(target[t-1]+1)*DIF+MIN_RV,"predicted_log_rv":(v+1)*DIF+MIN_RV,"status":"ok"})
                    pd.DataFrame(rows).to_csv(output/"predictions.csv",index=False)
                    legacy_metrics=[]
                    for model,g in pd.DataFrame(rows).groupby("model"):
                        y=g.actual_log_rv.to_numpy();p=g.predicted_log_rv.to_numpy()
                        yn=(y-MIN_RV)/DIF-1;pn=(p-MIN_RV)/DIF-1
                        q=np.abs(yn)/np.abs(pn)
                        legacy_metrics.append({"model":model,"mse_log_rv":float(np.mean((y-p)**2)),"legacy_normalized_qlike_sum":float(np.sum(q-np.log(q)-1))})
                    pd.DataFrame(legacy_metrics).to_csv(output/"legacy_metrics.csv",index=False)
        write_json(output/"complete.json",{"run_id":run_id,"seconds":time.perf_counter()-start,"stages":stages,"quantum_max_absolute_error":verification})
    except Exception as exc:
        write_json(output/"failure.json",{"error":repr(exc),"completed_stages":stages,"seconds":time.perf_counter()-start})
        raise
    finally:
        os.chdir(previous)
    print(f"Legacy benchmark complete: {output}")
