import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from qrcstudy.data import FEATURES, fit_scaler, inverse_target, monthly_features, transform, validate_prices
from qrcstudy.models import sequences, forecast
from qrcstudy.run import checked_manifest
from qrcstudy.report import losses, stationary_indices, load_validated
from qrcstudy.data import digest, write_json


class StudyTests(unittest.TestCase):
    def prices(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2000-01-03", "2020-03-15")
        return pd.Series(100*np.exp(np.cumsum(rng.normal(0,.01,len(dates)))), index=dates)

    def test_aggregation_and_incomplete_month(self):
        p = self.prices()
        f = monthly_features(p, "2020-03-15")
        self.assertEqual(f.index[-1], pd.Timestamp("2020-02-29"))
        self.assertEqual(f.index[0], pd.Timestamp("2001-01-31"))
        r = np.log(p).diff().loc["2020-02"]
        self.assertAlmostEqual(f.iloc[-1].log_rv, np.log(np.sqrt((r*r).sum())))
        self.assertAlmostEqual(f.iloc[-1].downside_share, (r[r<0]**2).sum()/(r*r).sum())

    def test_future_does_not_change_past(self):
        p = self.prices()
        altered = p.copy()
        altered.loc["2019":] *= 1.5
        f = monthly_features(p,"2020-03-15")
        g = monthly_features(altered,"2020-03-15")
        pd.testing.assert_frame_equal(f.loc[:"2018"], g.loc[:"2018"])
        self.assertEqual(fit_scaler(f), fit_scaler(g))
        s = fit_scaler(f)
        x,y,_ = transform(f,s)
        xx,yy,_ = transform(g,s)
        t = x.index.get_loc("2018-05-31")
        for name in ["HAR","HARX","AR1","AR3","Persistence"]:
            args = (name,x.to_numpy(),y.to_numpy(),None,None,t,120,0)
            changed = (name,xx.to_numpy(),yy.to_numpy(),None,None,t,120,0)
            self.assertEqual(forecast(*args),forecast(*changed))

    def test_sequence_alignment(self):
        x = np.arange(30).reshape(10,3)
        seq = sequences(x)
        np.testing.assert_array_equal(seq[3],x[:3])
        np.testing.assert_array_equal(seq[-1],x[-3:])
        self.assertTrue(np.isnan(seq[:3]).all())

    def test_scaling_roundtrip_and_unclipped_target(self):
        f = monthly_features(self.prices(),"2020-03-15")
        s = fit_scaler(f)
        f.loc[f.index[-1],"log_rv"] = 100
        x,y,clip = transform(f,s)
        self.assertTrue(clip.iloc[-1].log_rv)
        self.assertEqual(x.iloc[-1].log_rv,0)
        np.testing.assert_allclose(inverse_target(y,s),f.log_rv,atol=1e-12)

    def test_reject_missing_duplicate_invalid(self):
        p = self.prices().iloc[:5]
        with self.assertRaises(ValueError):validate_prices(p.iloc[:-1],p.index)
        with self.assertRaises(ValueError):validate_prices(pd.concat([p,p]))
        p.iloc[0] = 0
        with self.assertRaises(ValueError):validate_prices(p)

    def test_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            checked_manifest(d,{"data":"a"})
            checked_manifest(d,{"data":"a"})
            with self.assertRaises(ValueError):checked_manifest(d,{"data":"b"})

    def test_variance_qlike_and_direction(self):
        a=np.log(np.array([2.,1.]))
        p=np.log(np.array([1.,2.]))
        result=losses(a,p,np.log([1.5,1.5]))
        np.testing.assert_allclose(result["qlike_variance"],[4-np.log(4)-1,.25-np.log(.25)-1])
        np.testing.assert_array_equal(result["directional_accuracy"],[0,0])
        np.testing.assert_allclose(losses(a,a,a)["qlike_variance"],0)
        with self.assertRaises(ValueError):losses([np.nan],[1],[0])

    def test_bootstrap_reproducibility(self):
        a=stationary_indices(104,reps=30)
        np.testing.assert_array_equal(a,stationary_indices(104,reps=30))
        self.assertTrue(((a>=0)&(a<104)).all())
        self.assertGreater(np.mean(a[:,1:]==(a[:,:-1]+1)%104),.7)

    def test_seeded_lstm_resume_order(self):
        from qrcstudy.models import lstm_forecast
        rng=np.random.default_rng(7)
        x=rng.uniform(-1,0,(130,7));y=rng.uniform(-1,0,130)
        seq=sequences(x)
        a=lstm_forecast(seq,y,3,123,"LSTMX",42)
        lstm_forecast(seq,y,4,124,"LSTMX",99)
        b=lstm_forecast(seq,y,3,123,"LSTMX",42)
        self.assertEqual(a,b)

    def test_report_rejects_wrong_actual_and_missing_forecast(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            dates=pd.date_range("1990-01-31","2001-01-31",freq="ME")
            data=folder/"data.csv"
            pd.DataFrame({"log_rv":np.linspace(-4,-2,len(dates))},index=dates).to_csv(data)
            config={"data":str(data),"data_sha256":digest(data),"start":"2001-01-31","end":"2001-01-31","models":["Persistence"],"seeds":[0],"windows":[120]}
            run_id=checked_manifest(folder,config)
            records=[]
            for t in [pd.Timestamp("2001-01-31"),pd.Timestamp("2001-02-28")]:
                prev=dates.get_loc(t-pd.offsets.MonthEnd())
                records.append({"run_id":run_id,"configuration":"modern","model":"Persistence","seed":0,"window":120,"target_month":str(t.date()),"forecast_origin":str((t-pd.offsets.MonthEnd()).date()),"training_start":str((t-pd.offsets.MonthEnd(120)).date()),"training_end":str((t-pd.offsets.MonthEnd()).date()),"actual_log_rv":-2 if t.month==1 else None,"previous_log_rv":float(np.linspace(-4,-2,len(dates))[prev]),"predicted_log_rv":-2,"status":"ok"})
            for i,r in enumerate(records):write_json(folder/f"checkpoints/Persistence/{i}.json",r)
            self.assertEqual(len(load_validated(folder)[0]),2)
            published_hash=digest(folder/"predictions.csv")
            write_json(folder/"report_receipt.json",{"prediction_sha256":published_hash})
            (folder/"checkpoints").rename(folder/"local-checkpoints")
            self.assertEqual(len(load_validated(folder)[0]),2)
            self.assertEqual(digest(folder/"predictions.csv"),published_hash)
            (folder/"local-checkpoints").rename(folder/"checkpoints")
            records[0]["actual_log_rv"]=-1
            write_json(folder/"checkpoints/Persistence/0.json",records[0])
            with self.assertRaises(ValueError):load_validated(folder)
            (folder/"checkpoints/Persistence/0.json").unlink()
            with self.assertRaises(ValueError):load_validated(folder)


if __name__ == "__main__":
    unittest.main()
