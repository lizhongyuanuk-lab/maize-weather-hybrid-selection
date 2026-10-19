from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "g2f_data_feasibility_audit.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# MaizeClimate Research：公开多环境玉米数据的天气增强预测研究

这是**项目总称**，不是最终课题标题。本 Notebook 对决定研究方向前的六项数据条件做可复核审计。

核验范围：G2F 2014–2023 训练数据、2024 竞赛测试输入，以及从 CyVerse 官方下载的 2024 真实观测标签。"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
from IPython.display import display, Markdown
from audit_g2f import run_audit

ROOT = Path.cwd()
if not (ROOT / 'audit_g2f.py').exists():
    ROOT = Path('MaizeClimate_Research/audit').resolve()

result = run_audit(ROOT / 'data/raw', ROOT / 'outputs')
summary = result['summary']
checks = result['checks']
display(checks)"""
    ),
    nbf.v4.new_markdown_cell("## 六项核验的关键分母"),
    nbf.v4.new_code_cell(
        """import pandas as pd

key_counts = pd.DataFrame([
    ['训练产量标签', summary['yield']['rows_labeled'], summary['yield']['rows_total'], summary['yield']['label_completeness_pct']],
    ['有标签行的播种日期', summary['planting_dates']['labeled_rows_present'], summary['planting_dates']['labeled_rows_total'], summary['planting_dates']['labeled_rows_present_pct']],
    ['有产量环境的天气匹配', summary['weather']['yield_envs_with_weather'], summary['weather']['yield_envs_total'], summary['weather']['yield_env_weather_coverage_pct']],
    ['跨≥2环境的训练材料', summary['material_repetition']['hybrids_in_at_least_2_environments'], summary['material_repetition']['train_labeled_hybrids'], round(100*summary['material_repetition']['hybrids_in_at_least_2_environments']/summary['material_repetition']['train_labeled_hybrids'],3)],
    ['2024官方真实观测标签', summary['test_2024_labels']['observed_rows_with_numeric_yield'], summary['test_2024_labels']['requested_rows'], summary['test_2024_labels']['label_completeness_pct']],
], columns=['项目','有效数','总数','覆盖率_pct'])
display(key_counts)"""
    ),
    nbf.v4.new_markdown_cell("## 2024 真实标签覆盖与键完整性"),
    nbf.v4.new_code_cell(
        """label_quality = pd.DataFrame([
    ['官方观测行', summary['test_2024_labels']['observed_rows']],
    ['数值产量行', summary['test_2024_labels']['observed_rows_with_numeric_yield']],
    ['提交模板行', summary['test_2024_labels']['requested_rows']],
    ['模板重复 Env-Hybrid 键', summary['test_2024_labels']['template_duplicate_keys']],
    ['观测标签重复 Env-Hybrid 键', summary['test_2024_labels']['observed_duplicate_keys']],
    ['观测键不在模板中', summary['test_2024_labels']['observed_keys_not_in_template']],
    ['非正产量', summary['test_2024_labels']['nonpositive_observed_yields']],
], columns=['检查项','结果'])
display(label_quality)

label_coverage = pd.read_csv(ROOT / 'outputs/2024_observed_label_coverage_by_environment.csv')
display(label_coverage.sort_values(['label_coverage_pct', 'Env']).reset_index(drop=True))"""
    ),
    nbf.v4.new_markdown_cell("## 早期预测窗口的天气可用性"),
    nbf.v4.new_code_cell(
        """early = pd.DataFrame(summary['early_weather_windows'])
display(early)"""
    ),
    nbf.v4.new_markdown_cell("## 覆盖、材料重复与 2024 难度"),
    nbf.v4.new_code_cell(
        """coverage = pd.DataFrame([
    ['训练时间', f\"{summary['coverage']['train_year_min']}–{summary['coverage']['train_year_max']}\"],
    ['训练年份', summary['coverage']['train_years']],
    ['训练环境', summary['coverage']['train_environments']],
    ['训练地点', summary['coverage']['train_field_locations']],
    ['2024测试环境', summary['coverage']['test_environments']],
    ['训练材料跨环境数中位数', summary['material_repetition']['median_environments_per_hybrid']],
    ['2024材料总数', summary['material_repetition']['test_hybrids']],
    ['2024在训练中原样出现的材料', summary['material_repetition']['test_hybrids_seen_in_training']],
    ['2024新材料', summary['material_repetition']['test_hybrids_new']],
], columns=['指标','值'])
display(coverage)"""
    ),
    nbf.v4.new_markdown_cell(
        """## 方向判断（不是最终标题）

1. 播种日期与逐日天气足以支持早期预测：有标签训练行的播种日期为 100%，2024 的 23 个环境都有日期；播后 30/60/90/120 天的日期覆盖均为 100%。
2. 跨年份与跨地点数据也足以评估泛化：训练期覆盖 10 年、272 个环境、45 个地点，同一材料跨环境数中位数为 17。
3. 但 2024 是较强的材料外推测试：1,063 个材料中 959 个未在训练集中原样出现。泛化问题若一次同时包含新年份、新环境、新材料，五天内难以解释清楚。
4. 官方 2024 标签文件与提交模板的 `Env × Hybrid` 键一一匹配：无重复键、无模板外键；9,486/10,057 个请求键有真实产量。缺失的 571 行不能被视作零，也不应插补进最终指标。

因此，最小可回答问题应优先采用“**播后固定窗口 + 天气增量价值 + 2024 独立验证**”的设计；跨地点/年份泛化作为分组验证，而不是同时扩展成另一个大问题。"""
    ),
    nbf.v4.new_markdown_cell(
        """## 数据限制

- 训练天气缺少 `TXH2_2015`、`TXH2_2016`、`TXH2_2017`。
- 2024 天气文件覆盖 1 月 1 日至 11 月 10 日，并非完整到年末；但不影响播后 30–120 天窗口。
- 2024 的核心天气变量在播后 60–120 天窗口约有 2.4%–2.9% 的行存在缺失，集中在 7 月 12–14 日，需要预先规定插补策略。
- CyVerse 官方 `7_Testing_Observed_Values.csv` 已下载并逐行审计。10,057 个模板键中 9,486 个有真实产量，覆盖率 94.322%。
- 571 个无标签键并非全部属于 `SCH1_2024`：该环境完整缺失 385 行，另有 186 行分散在若干部分缺失环境。因此最终指标只在有真实标签的键上计算，并同时报告各环境等权的 macro-RMSE。
- 2024 标签只能在模型、特征窗口和超参数全部由 2014–2023 数据确定后使用一次，避免测试集泄漏。
- 官方标签文件 SHA-256：`8bd354c44bce50abd09a9839272112fdb34bdb96ec6e37ae644eda916aeeabcc`。"""
    ),
    nbf.v4.new_markdown_cell(
        """## 公开来源

- [G2F Resources](https://www.genomes2fields.org/resources/)
- [Official data note](https://pmc.ncbi.nlm.nih.gov/articles/PMC12983533/)
- [CyVerse dataset landing page](https://dc.cyverse.org/dataset/genomes_to_fields_2024_maize_genotype_by_environment_prediction_competition)
- [CyVerse official 2024 observed values](https://datacommons.cyverse.org/download/iplant/home/shared/commons_repo/curated/GenomesToFields_GenotypeByEnvironment_PredictionCompetition_2025/Testing_data/7_Testing_Observed_Values.csv)
- [Public analysis mirror](https://github.com/imbaterry11/UFEED_Maize_GxE)
- [Independent 2024 row-level check](https://pipette.bio/use-cases/maize-future-season-genomic-prediction.html)"""
    ),
]

nbf.write(nb, NOTEBOOK)
print(NOTEBOOK)

