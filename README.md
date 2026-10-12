# 智鉴 AI（VeriMind AI）

> 多模型协同的 AI 幻觉检测、证据溯源、错误归因与自动修正系统 —— **Demo 版**

把「相信一个模型」换成「一条可审计的校验流水线」：每条结论都能追问一句 **"你凭什么这么说？"**

---

## 一、当前进度（如实标注）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 阶段 0 | 需求冻结与技术方案（见 `docs/00-stage0-scope-and-design.md`） | ✅ 已完成 |
| 阶段 1 | 开发环境与项目骨架 | ✅ 已完成（**当前**） |
| 阶段 2 | 基础检测流程（离线可跑） | ⏳ 未开始 |
| 阶段 3 | 接入真实大模型 | ⏳ 未开始 |
| 阶段 4 | 文献检索与证据核验 | ⏳ 未开始 |
| 阶段 5 | 修正与独立复核 | ⏳ 未开始 |
| 阶段 6 | 测试与 Demo 完善 | ⏳ 未开始 |
| 阶段 7 | 比赛展示准备 | ⏳ 未开始 |

> **重要**：现在只有工程骨架和数据结构。**检测功能尚未实现**，因此「检测工作台 / 检测报告 / 评测」三页是占位页，
> 不显示任何结果。这是刻意为之：宁可页面空着，也不放看起来像真数据的假内容。

---

## 二、快速开始（Windows / PowerShell）

在项目根目录依次执行：

```powershell
# 1) 一键启动（推荐；会自动切 UTF-8 并检查环境）
powershell -ExecutionPolicy Bypass -File .\run.ps1

# 2) 或者手动启动
.\.venv\Scripts\python.exe -m streamlit run app.py
```

启动后浏览器会自动打开 <http://127.0.0.1:8501>。

**应该看到什么**：一个中文界面，标题是「🔍 智鉴 AI（VeriMind AI）」，
顶部有一条 **"当前运行模式：离线模拟"** 的提示条（因为还没有配置大模型 API Key），
下面是项目介绍、6 种核验状态图例、检测流程、以及左侧三个页面导航。

### 环境自检（遇到问题先跑这个）

```powershell
.\.venv\Scripts\python.exe scripts\check_env.py --network
```

它会检查 Python 版本、依赖版本、配置、数据目录，以及 Crossref / OpenAlex / arXiv / 大模型端点是否连得通。
退出码 0 表示核心检查全部通过。

### 从零重建环境（换电脑或环境损坏时）

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 三、如何测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

**当前状态（已实测）**：`42 passed`，全部离线可跑，不消耗任何 API 额度。

测试覆盖两类内容：

| 测试文件 | 验证什么 |
| --- | --- |
| `tests/test_stage1_data_contract.py` | **数据契约**：没有证据不许判「存在矛盾」；只有元数据的证据不许判断「支持/反驳」；反驳必须引用原文片段；报告里不许引用不存在的证据；密钥脱敏不泄露；无密钥自动降级为离线模式 |
| `tests/test_stage1_ui_renders.py` | **界面能真实渲染**：用 Streamlit 官方 AppTest 执行 4 个页面脚本，断言零异常；并断言占位页必须声明"尚未实现"，防止以后有人偷偷塞假数据 |

> 这些测试不是形式主义：本项目最关键的设计就在数据结构里，用测试钉死它，两个月后也不会被绕过。

---

## 四、目录结构

```
verimind-ai/
├─ app.py                       # Streamlit 入口 = 首页
├─ pages/
│  ├─ 1_检测工作台.py           # 输入文本、启动检测（阶段 2）
│  ├─ 2_检测报告.py             # 主张/证据/问题/修正/复核（阶段 2、4、5）
│  └─ 3_评测.py                 # 真实指标与失败案例（阶段 6）
├─ verimind/                    # 业务代码包（根目录平铺，免安装即可 import）
│  ├─ core/
│  │  ├─ config.py              # 配置：读 .env、脱敏密钥、判定运行模式
│  │  ├─ enums.py               # 6 状态 / 8 问题类型 / 全部枚举（词汇表）
│  │  └─ schemas.py             # 数据契约：Claim/Evidence/VerificationResult/
│  │                            #   Correction/ReviewResult/DetectionReport
│  ├─ ui/theme.py               # 配色、状态徽标、占位页组件
│  ├─ claims/  models/  retrieval/  verification/     # 阶段 2–4 逐步实现
│  ├─ attribution/  correction/  review/              # 阶段 5 实现
│  ├─ evaluation/  storage/                           # 阶段 2（存储）、阶段 6（评测）
├─ scripts/check_env.py         # 环境自检（含外部 API 连通性）
├─ tests/                       # 自动化测试（42 个，离线可跑）
├─ data/                        # 本地数据库/缓存/原始检索响应（不入 Git）
├─ docs/                        # 阶段文档、架构说明
├─ .env.example                 # 配置模板（复制成 .env 后填写）
├─ requirements.txt             # 依赖清单（固定大版本上限）
├─ pyproject.toml               # 项目元信息 + pytest 配置
└─ run.ps1                      # 一键启动脚本
```

### 关键文件的作用

| 文件 | 作用 | 为什么这么设计 |
| --- | --- | --- |
| `verimind/core/enums.py` | 6 种核验状态、8 类问题类型等全部枚举 | 这些词被 6 个模块共用，集中定义才不会出现"同一状态两种写法" |
| `verimind/core/schemas.py` | 所有跨模块数据模型 + 强校验 | **禁止模块间传裸 dict**；把"没有证据不许判错"这类诚实性规则写成代码约束 |
| `verimind/core/config.py` | 从 `.env`/环境变量读配置 | 密钥不进代码；缺密钥自动降级为离线模式而不是崩溃；只暴露脱敏密钥 |
| `verimind/ui/theme.py` | 状态配色与徽标 | 6 种状态的视觉定义只有一处，避免各页面颜色不一致 |

---

## 五、环境要求（本机实测结果）

| 项目 | 实测值 |
| --- | --- |
| 操作系统 | Windows 11 家庭版 中文版 |
| Python（虚拟环境） | **3.13.16**（`D:\python3.13\python.exe` 创建） |
| pip / Git | 26.2.1 / 2.56.0 |
| 主要依赖 | streamlit 1.65.0、pydantic 2.14.1、pandas 3.0.6、httpx 0.28.1、openai 1.109.1、pytest 8.4.2 |
| 大模型 API Key | **尚未配置**（当前为离线模拟模式；阶段 3 才需要） |
| 外部检索服务 | Crossref / OpenAlex / arXiv 均实测 HTTP 200 可达（免密钥） |

---

## 六、配置说明

配置全部通过**环境变量或 `.env` 文件**提供，变量名统一以 `VERIMIND_` 开头。

```powershell
Copy-Item .env.example .env    # 然后编辑 .env
```

- `.env` 已被 `.gitignore` 忽略，**不会提交到 Git**。
- 界面上只显示脱敏后的密钥（如 `sk-****abcd`）；报告中只记录供应商与模型名。
- `VERIMIND_LLM_PROVIDER=mock`（默认）时完全不调用真实模型。
- 填入真实密钥后，系统会自动把"运行模式"从「离线模拟」切换为「单模型多角色模拟」；
  若再为独立复核员配置一个**不同的**模型，则升级为「多模型协同」——模式会在界面顶部如实标注。

---

## 七、常见问题

| 现象 | 原因与处理 |
| --- | --- |
| 命令行里中文显示成乱码 | 终端代码页不是 UTF-8。执行 `chcp 65001`，或用 `.\run.ps1` 启动（已自动处理） |
| `streamlit : 无法将"streamlit"项识别为...` | 直接用模块方式运行：`.\.venv\Scripts\python.exe -m streamlit run app.py` |
| 端口被占用 | 换端口：`.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8600` |
| 提示缺少依赖 | `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| PowerShell 禁止运行脚本 | 用 `powershell -ExecutionPolicy Bypass -File .\run.ps1` |
| Python 3.15 装不上某些库 | 本项目虚拟环境固定用 **3.13** 创建（`py -3.13 -m venv .venv`） |

---

## 八、设计原则（会贯穿整个项目）

1. **证据可追溯**：报告中每条判断都要能点开到真实来源；不在报告里出现无法解释的结论。
2. **模型不下最终结论**：大模型只输出结构化判断，最终状态由 `verification/rules.py` 的可解释规则表裁决。
3. **无证据不判错**："证据不足 / 暂无法核验"是一等公民状态，绝不为了凑结论把"查不到"说成"错误"。
4. **不假装读过全文**：证据必须标注内容层级（仅元数据 / 摘要 / 全文）。
5. **不给假概率**：没有经过校准的概率模型，就不显示貌似精确的"幻觉概率"。
6. **不虚构数据**：不编造文献、DOI、页码、实验数据；页面上的指标必须来自真实运行。
7. **密钥安全**：任何密钥都不写入代码、不进 Git、不出现在日志与报告中。

---

## 九、免责声明

本项目为 Demo，用于方法验证与竞赛演示。系统输出是基于可检索公开证据的**辅助判断**，
不构成学术结论、法律意见或事实认定的最终依据。系统无法确知其他模型内部真实的出错原因；
报告中的问题类型是系统依据**可观察证据**作出的归因。
