# AGH 运行证据（真实执行记录）

> 队伍「旭日伴粽同行小队」｜ 编号 U071 ｜ 本科生组
>
> 本文件对应官方提交要求 4.2 中「演示视频：**运行证据**（AGH 执行记录、≥1 条工具/设备调用链、
> Agnes 模型参与核心任务证据、关键配置与专业验证结果）」。
>
> **本文所有内容均来自 AGH 的真实执行记录**，未经修饰。原始会话记录（已脱敏）见
> `docs/submission/evidence/agh-session-record.html`。

---

## 一、会话元信息（AGH 侧记录）

| 项目 | 值 |
|---|---|
| 会话 ID | `agnes:local:local-dev:cli:workspace:6b89c77f6ededb10` |
| 工作目录 | `<项目目录>`（本项目根目录） |
| 创建时间 | 2026-10-14T08:35:30Z |
| 记录序号 | lastSeq = 124 |
| profile / preset | `local-dev` / `standard-windows` |
| 消息构成 | user 3 · assistant 6 · tool 5 |
| 执行底座 | AGH 0.0.0（node 24.21.0，协议 `_agnes/v1`） |

---

## 二、工具调用链（≥1 条工具调用链 ✅）

AGH 侧记录到的工具调用序列如下（状态与耗时为记录原文）：

| 序 | 工具 | 状态 | 说明 |
|---|---|---|---|
| 1 | `read` | **completed** | 读取 `README.md`（前 5 行）——连通性验证 |
| 2 | `shell` | **failed** | 尝试执行本项目验证命令；**被审批策略拒绝**（详见 §五） |
| 3 | `read` | **completed** · 7 ms | 读取 `README.md`（指标表所在段落） |
| 4 | `read` | **completed** · 10 ms | 读取 `evaluations/b0-direct-100.json` |
| 5 | `read` | **completed** · 13 ms | 读取 `evaluations/b1-agent-100.json` |

命令行侧的实时记录（原文）：

```
- tool read
- tool read
- tool read · ok · 7ms
- tool read · ok · 10ms
- tool read · ok · 13ms
```

**上下文与用量**（记录原文，每步均有）：

```
Context breakdown · 4 sections · 1226 tokens (estimated)
$0.000242 (estimated) · title
$0.000252 (estimated) · inference
$0.000266 (estimated) · inference
$0.000419 (estimated) · inference
$0.000248 (estimated) · inference
$0.002467 (estimated) · inference
$0.010766 (estimated) · inference
```

> 说明：金额为 AGH 按账号单价给出的**估算值**，非本项目自行编造。

---

## 三、AGH 在本次执行中的实际输出（原文摘录）

AGH 读取三个文件后给出的报告（**节选，未改一字**）：

> 这三个文件我都读了开头部分，但有一个关键问题必须先说明：**这两个 JSON 文件各约 3000 行，
> 我目前只读到每个文件的第 700 多行（覆盖 G01–G26 左右，100 条中的前 26 条）**。
> 文件里没有现成的汇总指标字段（我看到的顶层字段只有 `created_at`、`baseline`、
> `engine_summary`、`case_count`、`rows`），所以要给出五项指标必须读完全部记录再统计。

其给出的对照表（**并明确声明数据来源**）：

| 指标 | B0 直接问模型 | B1 本项目方法 |
| --- | --- | --- |
| 精确率 | 61.4% | 97.6% |
| 召回率 | 68.6% | 78.4% |
| F1 | 64.8% | 87.0% |
| 误报率 | 44.9% | 2.0% |
| 证据可追溯率 | 0.0%（定义上为 0） | 69.0% |

> **必须声明的三点限制**
> 1. **上面这张表是 README 作者自己写好的汇总数字，不是我从两个 JSON 里算出来的。**
>    我没有对原始 100 条记录做任何统计。
> 2. 我抽样看到的原始记录与 README 的叙述方向一致（例如 B0 在 G10、G25 把正确内容判成
>    "存在矛盾"，即误报；B1 在这些条目上判"已核实"），但这只是前 26 条，不能替代全量核算。
> 3. JSON 里还有一处 README 提到的波动：B0 的六状态一致率由 52.0% 变为 51.0%
>    （100 条中有 1 条不同）——这属于模型判断的随机性，与指标口径有关。

**这一点值得单独指出**：AGH 在未被要求的情况下**主动声明"这张表不是我自己算的"**，
并列出未完成核验的部分。这与本作品"不假装成功、说清边界"的设计原则一致——
它同时说明：AGH 侧确实在真实执行与真实判断，而不是复述提示词。

---

## 四、如何复现这份记录

### 4.1 只读任务（免审批，可立即复现）

AGH 的只读工具（如 `read`）自带只读标记，**无需审批**，可直接复现：

```powershell
# AGH CLI 位置（本机安装于 D:\agnes-harness）
$node = "D:\New Folder\node.exe"
$cli  = "D:\agnes-harness\packages\cli\dist\local\agnes.mjs"

$prompt = "请只读取文件（不要执行命令、不要修改文件）。依次阅读 README.md、" +
          "evaluations/b0-direct-100.json、evaluations/b1-agent-100.json，" +
          "用表格报告 B0 与 B1 在精确率、召回率、F1、误报率、证据可追溯率上的数值。"

& $node $cli -p $prompt --profile local-dev --cwd "<项目目录>" --mode text
```

执行时终端会实时打印 `- tool read · ok · Nms`，即 §二 中的调用链。

### 4.2 执行命令类任务（需在 AGH 端批准一次）

AGH 默认对 `shell` 等写操作要求审批。非交互脚本无法自行批准
（AGH 官方文档原话：*"A script without an available approval interaction cannot approve
dangerous actions on its own."*），因此需二选一：

**方式 A：在 AGH Web 工作台执行（推荐，证据最完整）**

1. 浏览器打开 `http://127.0.0.1:4180`（AGH workbench）
2. 新建会话，工作目录选择本项目根目录
3. 输入以下任务：

   > 请在当前目录依次执行以下两条命令并原样报告结果：
   > （1）`.\.venv\Scripts\python.exe -m pytest -q`
   > （2）`.\.venv\Scripts\python.exe scripts\run_demo.py --id real_doi_wrong_claim`
   > 汇报每条命令的退出状态，以及输出中的关键数字（测试通过数、主张条数、状态分布、检索与模型调用次数）。
   > 不要修改任何文件。

4. 出现审批卡片时，检查命令后选择允许（一次性或本会话）
5. **截图**：会话轨迹页（含工具调用序列、耗时与审批卡片）

**方式 B：命令行 + `--park` 等待批准**

```powershell
& $node $cli -p "<同上任务文本>" --profile local-dev --cwd "<项目目录>" --mode text --park
```

`--park` 会让一次性调用**等待审批**；在 Web 工作台出现审批卡片后确认，命令即继续执行。

> TUI 会话中另有 `/yolo`（完全权限）可跳过本会话的后续审批，
> 但 AGH 官方文档明确"不应把它作为自动化默认"，本队仅在需要时手动使用。

---

## 五、如实说明（三点）

### 5.1 本次 `shell` 调用为什么失败

第 2 次工具调用尝试执行本项目验证命令，**被审批策略拒绝**。AGH 给模型的说明与记录一致：

> The first command was rejected, so I stopped there — I did not run either command
> and made no changes to any files.

原因是：本次通过命令行非交互执行，**没有可用的审批客户端来询问用户**，
因此按 AGH 的安全策略不予执行（AGH 官方文档：无人可询问的调用记为 unavailable，不执行）。
这**不是**本项目或 AGH 的故障，而是 AGH 的默认安全行为；
要取得"AGH 执行命令"的记录，请按 §4.2 在 Web 端批准一次。

### 5.2 AGH 的报告是"部分读取"的结果

AGH 自述只读到两个 JSON 文件的前 700 余行，因此它**没有**找到顶层汇总字段
（`metrics`）。实际上该字段位于文件后部；README 中的指标表即直接取自该字段。
这一点是 AGH 自己主动声明的，本文件照实记录。

### 5.3 关于记录导出方式（安全提示）

AGH 的 `export --raw` 会**在导出时给出明确警告**：

```
warning: --raw exports without redaction (secrets, paths, PII)
```

本队实测确认该警告属实（原始 JSON 中含未脱敏的本地路径与用户名），
因此**未保留、未提交**该文件；随本文件提供的 HTML 记录为 AGH 的**脱敏**导出，
经检查不含密钥、本地路径、用户名或邮箱。

---

## 六、附：其余三类运行证据的取材位置

| 要求项 | 取材位置 |
|---|---|
| **Agnes 模型参与核心任务的证据** | 检测报告页「修正建议与独立复核」——由 `agnes-3.0-flash` 给出的五项检查逐项结论；运行模式标注为「多模型协同」 |
| **关键配置** | 首页顶部横幅（运行模式 / 主模型 / 独立复核模型）＋ 工作台模型信息行 |
| **专业验证结果** | 评测页「核心对比」（B0 误报率 44.9% vs B1 2.0%）、失败案例列表、`pytest -q` 的 `245 passed` |

> 逐项截图步骤见 `docs/submission/运行证据与截图清单.md`。
