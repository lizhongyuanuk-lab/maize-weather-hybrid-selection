import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyze_rolling_stability import (CANDIDATES, HORIZONS, cross_year_stability,
    digest, paired_environment_rows, verify_artifacts, verify_run_inputs, within_year_stability)
import analyze_rolling_stability as analysis
from metrics import evaluate_predictions


class StabilityTests(unittest.TestCase):
    def toy(self):
        # A shared site, unequal year sizes, and two environments under site A.
        return pd.DataFrame({"Year":[2020,2020,2020,2021],"Env":["a","b","c","d"],
            "Experiment_Code":["A","A","B","A"],"n_W0":[10,20,30,5],
            "rmse_W0":[2.,2.,2.,4.],"rmse_weather":[1.,2.,3.,6.],
            "delta_rmse":[-1.,0.,1.,2.]})

    def test_year_weighting_not_environment_or_row_weighting(self):
        r = cross_year_stability(self.toy(),repeats=200)
        self.assertEqual(r["year_equal_delta_rmse"],1.)
        self.assertEqual(r["environment_year_equal_delta_rmse"],.5)
        self.assertEqual(r["years_improved"],0)
        self.assertEqual(r["years_worsened"],1)
        self.assertAlmostEqual(r["relative_gain_of_year_equal_rmse_percent"],-100/3)
        self.assertAlmostEqual(r["mean_annual_relative_gain_percent"],-25.)

    def test_row_counts_do_not_change_cluster_interval(self):
        a = self.toy()
        b = a.assign(n_W0=[10000,1,900,25])
        self.assertEqual(cross_year_stability(a,repeats=200),cross_year_stability(b,repeats=200))

    def test_deterministic_and_shift_equivariant(self):
        a = self.toy()
        r = cross_year_stability(a,repeats=200)
        self.assertEqual(r,cross_year_stability(a,repeats=200))
        b = a.assign(delta_rmse=a.delta_rmse+3,rmse_weather=a.rmse_weather+3)
        s = cross_year_stability(b,repeats=200)
        for key in ("year_equal_delta_rmse","year_block_ci95_low","year_block_ci95_high",
                    "crossed_year_site_ci95_low","crossed_year_site_ci95_high"):
            self.assertAlmostEqual(s[key]-r[key],3.)

    def test_empty_single_and_duplicate_support(self):
        one = self.toy().iloc[:1]
        self.assertTrue(np.isnan(within_year_stability(one)["environment_ci95_low"]))
        self.assertEqual(cross_year_stability(one)["bootstrap_status"],"insufficient_years")
        with self.assertRaises(ValueError):
            cross_year_stability(pd.concat([one,one]))

    def test_sign_and_proportions(self):
        r = within_year_stability(self.toy().iloc[:3],repeats=100)
        self.assertEqual(r["improved_environments"],1)
        self.assertEqual(r["worsened_environments"],1)
        self.assertEqual(r["unchanged_environments"],1)
        self.assertEqual(r["delta_rmse"],0)

    def test_pairing_rejects_changed_members_or_targets(self):
        a = pd.DataFrame({"sample_id":["1","2"],"Env":["E","E"],"Hybrid":["A","B"],
                          "Year":[2020,2020],"Experiment_Code":["X","X"],
                          "y_true":[1.,2.],"prediction":[1.,2.]})
        b = a.assign(prediction=[2.,3.])
        self.assertEqual(paired_environment_rows(a,b).delta_rmse.iloc[0],1.)
        with self.assertRaises(ValueError):
            paired_environment_rows(a,b.assign(sample_id=["3","4"]))
        with self.assertRaises(ValueError):
            paired_environment_rows(a,b.assign(y_true=[2.,3.]))

    def test_analysis_manifest_chain_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(prefix="maize_binding_test_") as temporary:
            root = Path(temporary).resolve()
            directory = root/"results/2020/final"
            features = root/"features/2020/final"
            freeze = root/"results/2020/development/frozen_selection.json"
            for folder in (directory/"predictions",features,freeze.parent):
                folder.mkdir(parents=True)
            protocol = root/"PROTOCOL.md"
            protocol.write_text("fixed synthetic protocol")
            train = features/"train_samples.csv"
            train.write_text("sample_id,Hybrid\na,A\n")
            upstream = root/"source.csv"
            upstream.write_text("immutable synthetic input")
            freeze.write_text("{}")
            feature_manifest = features/"feature_manifest.json"
            feature_manifest.write_text(json.dumps({"input_hashes":{str(upstream):digest(upstream)},
                "outputs":{"train_samples.csv":digest(train)}}))
            run = directory/"run_manifest.json"
            run.write_text(json.dumps({"inputs":{str(p):digest(p) for p in [train,feature_manifest,protocol,freeze]},
                "frozen_selection_sha256":digest(freeze)}))
            files = [run]
            for name in ("summary_metrics.csv","environment_metrics.csv","weather_increment_comparisons.csv","earliest_sufficient_windows.json"):
                p = directory/name
                p.write_text("synthetic")
                files.append(p)
            configs = {(m,h) for m in CANDIDATES for h in HORIZONS}|{("GlobalMean",0),("HistoricalSiteMean",0)}
            for m,h in configs:
                p = directory/"predictions"/f"{m}_W{h}_evaluated.csv"
                p.write_text("synthetic")
                files.append(p)
            completion = {"artifact_path_base":str(root),"run_manifest_sha256":digest(run),
                "frozen_selection_sha256":digest(freeze),
                "artifacts":{str(p.relative_to(root)):digest(p) for p in files}}
            completion_path = directory/"final_evaluation_manifest.json"
            completion_path.write_text(json.dumps(completion))
            self.assertEqual(verify_run_inputs(directory,features,completion,root,2020),freeze)
            self.assertEqual(verify_artifacts(completion_path,root),completion)
            for target in (train,freeze,protocol,upstream):
                old = target.read_text()
                target.write_text(old+"tampered")
                with self.assertRaises(ValueError):
                    verify_run_inputs(directory,features,completion,root,2020)
                target.write_text(old)
            damaged = {**completion,"artifacts":dict(completion["artifacts"])}
            damaged["artifacts"].pop(str((directory/"summary_metrics.csv").relative_to(root)))
            with self.assertRaises(ValueError):
                verify_run_inputs(directory,features,damaged,root,2020)

    def test_complete_five_year_table_assembly_with_empty_subgroups(self):
        # Tiny synthetic data, kept exclusively in a temporary directory. The
        # separate test above covers integrity; this tests complete table wiring.
        with tempfile.TemporaryDirectory(prefix="maize_aggregate_test_") as temporary:
            project = Path(temporary).resolve()
            root = project/"modeling/rolling_v1"
            root.mkdir(parents=True)
            (root/"PROTOCOL.md").write_text("synthetic frozen protocol")
            for name in ("analyze_rolling_stability.py","metrics.py","run_experiments.py"):
                (project/"modeling"/name).write_text("synthetic provenance placeholder")
            configs = sorted({(m,h) for m in CANDIDATES for h in HORIZONS}|{("GlobalMean",0),("HistoricalSiteMean",0)})
            def fake_freeze(directory,feature_directory,completion,root,year):
                return (project/"modeling/results/development/frozen_selection.json" if year==2024
                        else root/f"results/{year}/development/frozen_selection.json")
            for year in analysis.YEARS:
                directory = project/"modeling/results/final" if year==2024 else root/f"results/{year}/final"
                feature = project/"modeling/features/final" if year==2024 else root/f"features/{year}/final"
                freeze = fake_freeze(directory,feature,{},root,year)
                for folder in (directory/"predictions",feature,freeze.parent):
                    folder.mkdir(parents=True,exist_ok=True)
                pd.DataFrame({"sample_id":[f"train{year}"],"Year":[year-1],"Hybrid":["A"],"Experiment_Code":["S"]}).to_csv(feature/"train_samples.csv",index=False)
                freeze.write_text(json.dumps({"recommended_by_validation":{"model":"CatBoost","window":120}}))
                (directory/"final_evaluation_manifest.json").write_text("{}")
                (directory/"earliest_sufficient_windows.json").write_text(json.dumps({"models":{m:{"earliest_sufficient_window":None} for m in CANDIDATES}}))
                summaries,envs = [],[]
                for model,horizon in configs:
                    frame = pd.DataFrame({"sample_id":[f"s{year}_{i}" for i in range(4)],
                        "Env":[f"E{year}a"]*2+[f"E{year}b"]*2,"Hybrid":["A","B","A","C"],
                        "Year":[year]*4,"Experiment_Code":["S"]*4,"y_true":[2.,3.,4.,5.],
                        "prediction":np.array([2.,3.,4.,5.])+1-horizon/240})
                    frame.to_csv(directory/"predictions"/f"{model}_W{horizon}_evaluated.csv",index=False)
                    summary,env = evaluate_predictions(frame)
                    summaries.append({"model":model,"window":horizon,**summary})
                    envs.append(env.assign(model=model,window=horizon))
                pd.DataFrame(summaries).to_csv(directory/"summary_metrics.csv",index=False)
                pd.concat(envs).to_csv(directory/"environment_metrics.csv",index=False)
                pd.DataFrame({"model":list(CANDIDATES),"window":[120]*3}).to_csv(directory/"weather_increment_comparisons.csv",index=False)
            with patch.object(analysis,"PROJECT",project), patch.object(analysis,"verify_artifacts",return_value={}), \
                 patch.object(analysis,"verify_run_inputs",side_effect=fake_freeze), \
                 patch.object(analysis,"within_year_stability",side_effect=lambda f:within_year_stability(f,repeats=25)), \
                 patch.object(analysis,"cross_year_stability",side_effect=lambda f:cross_year_stability(f,repeats=25)):
                manifest = analysis.analyze(root)
            self.assertEqual(manifest["n_scored_observations"],20)
            self.assertEqual(manifest["n_environment_years"],10)
            self.assertEqual(manifest["table_rows"]["yearly_metrics"],85)
            self.assertEqual(manifest["table_rows"]["yearly_subgroup_metrics"],340)
            self.assertEqual(manifest["table_rows"]["cross_year_subgroup_stability"],48)
            cross = pd.read_csv(root/"analysis/cross_year_stability.csv")
            self.assertTrue(np.allclose(cross.loc[cross.window==120,"year_equal_delta_rmse"],-.5))
            subgroup = pd.read_csv(root/"analysis/yearly_subgroup_stability.csv")
            empty = subgroup.loc[(subgroup.group_type=="site_code")&(subgroup.group=="unseen")]
            self.assertEqual(len(empty),60)
            self.assertTrue((empty.n_samples==0).all())
            self.assertTrue(empty.delta_rmse.isna().all())


if __name__ == "__main__":
    unittest.main()
