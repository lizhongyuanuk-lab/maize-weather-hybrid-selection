# 官方来源完整性补核验

这是对 Stage 1 已保存数据的独立来源核验，不会重新整理数据，不会改动 `audit/data/raw/` 或 `preparation/outputs/`，也不涉及模型训练或测试集指标。

当前状态以 [verification_report.json](verification_report.json) 为准；`download_complete: false` 不能解释为“内容一致”。

## 最终结果（2026-10-18）

**此前未完成的四份官方来源核验现已全部完成，未发现任何数据单元格或行顺序差异。原始镜像 SHA-256 与 Stage 1 冻结记录仍一致，无需重新跑 Stage 1。**

| 来源 | 数据行 × 列（含标识列） | 官方完整字节数 | 核验结论 |
|---|---:|---:|---|
| 训练天气 | 98,236 × 18 | 10,460,386 | SHA-256 与镜像完全相同 |
| 2024 天气 | 7,245 × 18 | 702,794 | SHA-256 与镜像完全相同 |
| 训练性状 | 173,960 × 26 | 31,590,253 | 全部单元格与顺序键一致；仅 CRLF → LF 后即与镜像字节完全相同（31,416,292 字节） |
| 基因型 | 5,899 × 2,426 | 40,765,753 | 5,899 个基因型、2,425 个 marker 的全部值与顺序一致；以下三项格式转换后即与镜像字节完全相同（40,765,739 字节） |

“来源等价”只证明使用的是相同公开数据，并不证明每个观测都准确，也不替代缺失值、泄漏、划分与模型评估检查。训练/测试元数据和提交模板的先前核验仍见 `../source_verification.json`；其中 2024 镜像元数据确实存在差异，已经通过使用官方覆盖文件解决，并未假装该镜像本身无差异。

## 核验方法

1. 使用此前由官方目录与 `readme.txt` 确认的 CyVerse `de.cyverse.org/anon-files/` 公开文件地址。
2. 官方服务器通过 `HEAD` 返回准确 `Content-Length` 并声明支持 `Accept-Ranges: bytes`。使用正常 HTTPS，不关闭证书校验。
3. 将大文件按 1 MiB 字节块下载，只有响应 `Content-Range` 起止位置、总长度及实际块长度全部正确时才接受该块。
4. 按字节顺序拼接全部块，要求完整长度与官方响应一致，再计算完整文件 SHA-256。
5. CSV 使用标准 `csv.reader` 比较全部表头、行数、顺序键和单元格字符串。CSV 的 CRLF/LF 换行属于表示差异，不属于数据值差异。
6. 官方基因型 TXT 仅允许三种有记录的格式变换：去掉 `<Numeric>` 声明行；将制表符分隔解释为 CSV 列；将 `<Marker>` 标识列名对齐至镜像的 `Geno`。随后逐单元格严格比较，不自动舍入、不任意改写缺失值、不忽略行顺序。
7. 对训练性状和基因型进一步执行上述精确定义的字节转换；转换后完整 SHA-256 均与原镜像相同。报告记录转换前后大小与哈希值。

## 运行

从项目根目录运行：

```bash
python preparation/provenance_followup/verify_official_sources.py --workers 2 --timeout 90
python preparation/provenance_followup/test_verify_official_sources.py
```

不联网重复核验已取得的完整官方文件：

```bash
python preparation/provenance_followup/verify_official_sources.py --compare-only
```

脚本使用 Python 标准库与系统 `curl`，无需额外 Python 包。失败或中断的块不被算作完整文件；下次运行复用经过长度和范围校验的完整块。首轮 6 路下载部分请求收到服务器 HTTP 500，降至 2 路后继续，最后以单连接补齐 3 个失败块。

## 输出

- `official/`：完整取得并拼接成功的官方文件（本地忽略，不纳入 Git）。
- `parts/`：原始响应头及字节块，用于续传与范围验证（本地忽略）。
- `verification_report.json`：四个来源的完整长度、SHA-256、键与值对比，以及当前下载状态。
- `verification_all.json`：下载过程中的中间记录，不作为最终状态。
- `verification_test_weather.json`：首个完成的 2024 天气来源核验记录。
- `test_verify_official_sources.py`：覆盖换行差异、基因型格式转换、真实键/值差异和缺行检测的 4 项测试。

这份记录是现有 `preparation/source_verification.json` 的补充，不改写历史记录。2024 元数据原先发现的镜像补值问题仍成立；主流程已经采用独立保存的官方元数据，不能将该镜像描述为所有字节均与官方相同。
