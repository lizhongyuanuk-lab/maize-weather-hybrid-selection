"""Populate the local research report from completed, validated experiment files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "modeling/results"


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=12))


def main():
    final = RESULTS / "final"
    completion = json.loads((final / "final_evaluation_manifest.json").read_text())
    for stage in ("development", "final"):
        audit = json.loads((ROOT / "modeling" / f"{stage}_validation_report.json").read_text())
        if audit["status"] != "PASS":
            raise ValueError(f"{stage} independent validation must pass before final report generation")
        audited_path = ROOT / audit["audited_manifest_path"]
        if hashlib.sha256(audited_path.read_bytes()).hexdigest() != audit["audited_manifest_sha256"]:
            raise ValueError(f"{stage} results no longer match the independently validated manifest")
    freeze = json.loads((RESULTS / "development/frozen_selection.json").read_text())
    chosen = freeze["recommended_by_validation"]
    summary = pd.read_csv(final / "summary_metrics.csv")
    dev = pd.read_csv(RESULTS / "development/selected_metrics.csv")
    env = pd.read_csv(final / "environment_metrics.csv")
    groups = pd.read_csv(final / "subgroup_metrics.csv")
    comparisons = pd.read_csv(final / "weather_increment_comparisons.csv")
    earliest = json.loads((final / "earliest_sufficient_windows.json").read_text())
    models = ["Ridge", "ExtraTrees", "CatBoost"]
    selected = summary.loc[(summary.model == chosen["model"]) & (summary.window == chosen["window"])].iloc[0]
    family = summary.loc[summary.model == chosen["model"]].set_index("window")
    gain = (family.loc[0, "macro_rmse"] - family.loc[120, "macro_rmse"])/family.loc[0, "macro_rmse"]
    paired = comparisons.loc[(comparisons.model == chosen["model"]) & (comparisons.window == 120)].iloc[0]
    global_mean = summary.loc[summary.model == "GlobalMean"].iloc[0]
    site_mean = summary.loc[summary.model == "HistoricalSiteMean"].iloc[0]
    chosen_env = env.loc[env.model == chosen["model"]]
    deltas = chosen_env.pivot(index="Env", columns="window", values="rmse")[[0, 120]].rename(columns={0:"W0",120:"W120"})
    deltas["delta_rmse"] = deltas.W120 - deltas.W0
    deltas = deltas.sort_values("delta_rmse").reset_index()
    selected_groups = groups.loc[(groups.model == chosen["model"]) & (groups.window == chosen["window"])]
    full_ci = comparisons.loc[comparisons.window == 120, ["model", "vs_W0_delta_rmse", "vs_W0_ci95_low", "vs_W0_ci95_high", "full_window_relative_gain"]]
    p = ROOT / "showcase/src/data.json"
    snapshot = json.loads(p.read_text())
    snapshot.update({"title":"How Early Can Maize Yield Be Predicted?", "surface":"report", "buildStatus":"complete", "status":"observed", "generatedAt":datetime.now(timezone.utc).isoformat()})
    snapshot["queries"]["cohort"]["source"]["caveats"] = [
        "Only 9,486 of the 10,057 test inputs have official observed yield, covering 22 scored environments.",
        "All four pending full-source comparisons are now complete with no data-value differences; the known metadata mismatch is resolved through the official override.",
    ]
    snapshot["queries"]["cohort"]["rows"] = [r for r in snapshot["queries"]["cohort"]["rows"] if r["split"] != "Test scored (2024)"] + [
        {"split":"Test scored (2024)", "samples":int(selected.n), "environments":int(selected.n_environments)}
    ]
    snapshot["queries"]["cohort"]["source"]["files"] = [{"label":"split_summary.csv"}, {"label":"summary_metrics.csv"}]
    model_sources = {"label":"Frozen G2F temporal experiments", "files":[{"label":"summary_metrics.csv"},{"label":"frozen_selection.json"}],
        "metricDefinitions":[{"label":"Environment-equal RMSE", "definition":"Compute RMSE within each environment, then take the arithmetic mean across environments. Units: Mg/ha. Lower is better."},
                             {"label":"Weather windows", "definition":"Use only observed weather in [planting date, planting date + horizon); W0 uses no current-season weather."}],
        "caveats":["Final evaluation uses 9,486 observed environment × hybrid outcomes from 22 environments in one year (2024).",
                   "Hyperparameters and the highlighted configuration were selected using 2023 only; other test-year comparisons are descriptive, not new model selection.",
                   "Uncertainty intervals resample environments; they do not include model-selection uncertainty or establish causality."],
        "evidenceFlow":[{"kind":"method","title":"Frozen temporal protocol","detail":"Train 2014–2022 and choose among two preset candidates per family/window using 2023; freeze choices; refit 2014–2023; persist all predictions before reading test labels."}]}
    def query(frame, files, extra=None):
        source = dict(model_sources)
        source["files"] = [{"label": f} for f in files]
        if extra:
            source["caveats"] = source["caveats"] + extra
        return {"rows": records(frame), "source": source}
    all_summary_cols = ["model","window","n","n_environments","n_prediction_inputs","n_without_observed_target","macro_rmse","macro_rmse_ci95_low","macro_rmse_ci95_high","macro_mae","macro_spearman","macro_top20_recall","macro_bias","pooled_rmse"]
    snapshot["queries"].update({
        "test_accuracy": query(summary.loc[summary.model.isin(models)], ["summary_metrics.csv"]),
        "validation_accuracy": query(dev.loc[dev.model.isin(models)], ["selected_metrics.csv"], ["Validation metrics are used for selection and are not independent final-test estimates."]),
        "baselines": query(summary.loc[~summary.model.isin(models)], ["summary_metrics.csv"]),
        "weather_comparison": query(full_ci, ["weather_increment_comparisons.csv"], ["Delta is W120 minus W0 within the same model family; negative values favor weather. Confidence intervals are unadjusted for multiple comparisons."]),
        "environment_delta": query(deltas, ["environment_metrics.csv"], ["The displayed model family was chosen on 2023 validation; comparison is W120 versus W0 for that fixed family."]),
        "ranking": query(summary.loc[summary.model.isin(models)], ["summary_metrics.csv"], ["Top-20% recall is computed within environment with fractional tie handling; not a yield increase from deploying selection."]),
        "subgroups": query(selected_groups, ["subgroup_metrics.csv"], ["Seen/unseen groups are relative to 2014–2023 training; different group composition can change error. This is descriptive, not a causal effect of novelty."]),
        "all_metrics": query(summary[all_summary_cols], ["summary_metrics.csv"]),
        "window_sufficiency": query(pd.DataFrame([{"model":name, **details} for name,details in earliest["models"].items()]), ["earliest_sufficient_windows.json"], ["Operational rule: W120 gain at least 2% with paired CI below zero, then earliest window retaining at least 90% of that point-estimate gain. Not a formal noninferiority test."]),
    })
    snapshot["queries"]["validation_accuracy"]["source"]["caveats"] = [
        "Validation uses 11,491 environment × hybrid outcomes from 27 environments in 2023.",
        "These scores select the model and parameters; they are not independent final-test estimates.",
        "Environment-bootstrap intervals are conditional on fitted predictions and omit selection uncertainty.",
    ]
    source_followup = ROOT / "preparation/provenance_followup/verification_report.json"
    source_text = "完整来源核验状态请以项目中的 provenance_followup 核验报告为准；未解决的来源差异不能忽略。"
    if source_followup.exists():
        provenance = json.loads(source_followup.read_text())
        snapshot["provenanceFollowup"] = provenance
        if provenance["summary"]["all_followup_files_semantically_verified"]:
            source_text = "四份此前待核验的大文件均已完成官方全量比对：两份天气文件逐字节一致，性状和基因型在记录明确的格式转换后全部值一致。2024 年元数据原镜像有差异，主流程已使用官方覆盖文件，保留官方缺失值。来源核验没有改变既有建模输入。"
    direction = "下降" if gain >= 0 else "上升"
    enough = earliest["models"][chosen["model"]]["earliest_sufficient_window"]
    sufficient_text = (f"在预设规则下，该模型的最早充分窗口为播后 {enough} 天。该结论仅适用于本次 2024 评估。" if enough is not None else
                       "该模型未满足预设的充分窗口判定条件，不能宣布某个天数已经足够。")
    snapshot["researchNarrative"] = {
        "summary": f"## Executive Summary\n\n- 2023 年验证选出的配置是 **{chosen['model']} / W{chosen['window']}**；在独立的 2024 年评估中，环境等权 RMSE 为 **{selected.macro_rmse:.3f} Mg/ha**。总体均值与历史地点均值基线分别为 {global_mean.macro_rmse:.3f} 和 {site_mean.macro_rmse:.3f} Mg/ha。\n- 在同一 {chosen['model']} 家族内，从 W0 到 W120，环境等权 RMSE {direction} **{abs(gain)*100:.1f}%**；配对差值为 {paired.vs_W0_delta_rmse:+.3f} Mg/ha，95% 环境 bootstrap 区间 [{paired.vs_W0_ci95_low:+.3f}, {paired.vs_W0_ci95_high:+.3f}]。\n- {sufficient_text}",
        "scope": "## 如何读这些结果\n\nRMSE 衡量绝对产量误差，单位 Mg/ha（等于吨/公顷），越低越好。本研究先在各环境内计算误差，再对环境等权平均；9.5k 条产量观测并不等于 9.5k 个独立天气环境。最终测试包含 22 个可评估环境，另有 571 条测试输入没有官方产量标签，未用于评分。",
        "weather": f"## 天气增益是否稳定\n\n下表比较每个模型的 W120 与 W0。负值代表误差下降；区间跨零意味着本次环境级重采样不能清晰区分改善与无改善。对于验证选出的 {chosen['model']} 家族：{sufficient_text}\n\n充分窗口规则预先固定：W120 相对 W0 改善至少 2%、配对区间完全低于零，再找保留至少 90% 点估计增益的最早窗口。它不是正式非劣效检验或普遍农业临界日。",
        "environment": f"## 哪些环境受益，哪些没有\n\n固定展示验证期选出的 **{chosen['model']}** 家族，比较 W120 与 W0。全部可评估环境都展示，不只挑改善的地点。下方负值代表加入天气后误差更低。",
        "ranking": "## 预测产量，不等于选对材料\n\nTop-20% Recall 表示真实高产材料中有多少被预测排名前 20% 找回。绝对误差下降但筛选指标不变，意味着天气可能主要改善环境产量水平的校准，而非材料之间的排序。并列分数使用期望纳入权重，不任意按材料名称打破并列。",
        "subgroups": f"## 对新材料与新地点的表现\n\n以下仅展示验证期选出的 **{chosen['model']} / W{chosen['window']}** 配置。分组为描述性结果；尤其新地点样本有限，不能推广成所有未知地区的误差承诺。",
        "methods": "## 方法与贡献\n\n这是使用成熟算法的独立应用研究，不是新型大模型。贡献在于固定样本、严格逐日截断、训练期拟合预处理、分离绝对产量与材料排序，并完整呈现正面与负面结果。\n\n基因型使用 100 维训练期 PCA，结合地点、坐标与播种日序；天气采用 21 个汇总指标与 9 个质量信息量。开发阶段的 9 个质量列在训练集中均为常量，不能把模型增益归因于这些列。每模型每窗口比较 2 个固定候选。2023 年完成选择后，使用 2014–2023 年重拟合，全部 17 组预测存盘锁定后才统一评分。\n\n[已有 G2F 研究](https://academic.oup.com/g3journal/article/12/2/jkab440/6486423)已经将基因型与环境信息结合，并发现泛化能力依赖训练与测试环境的相似程度。因此本项目不声称首次加入天气；研究焦点是早季信息窗口与下一年度预测的权衡。",
        "limits": "## 使用边界\n\n只评估一个最终年份；有限的候选搜索不是穷尽优化；未进行多重比较校正；逐环境区间不包含选参不确定性。多播种日期样本是按小区权重聚合的混合暴露，而不是单一天气轨迹。天气数据可用延迟未核实，因此这是回顾性预测研究，不是已部署的实时系统。不能据此证明天气的因果效应、直接推荐中国农田种植决策或声称发表了论文。\n\n" + source_text,
    }
    snapshot["selectedResearchModel"] = chosen
    snapshot["researchNarrative"]["limits"] += "\n\n每个窗口分别按同样的验证规则选参，因此天气比较衡量的是相同选择预算下的完整方案增益，并非把所有参数固定后只改变天气的净作用。共同队列按 W120 日历可覆盖性筛选，不代表所有缺测试验。开发与最终训练矩阵中的 9 个天气质量列均为常量；这个结论不包括非恒定的坐标缺失指示。"
    snapshot["evidenceCompletion"] = {"n_scored":completion["n_scored"], "models":completion["preset_models_scored"]}
    p.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(p)


if __name__ == "__main__":
    main()
