# 市场监管投诉智能处理系统

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

> 本地优先、人机协同的多智能体市场监管投诉处理系统。
> 规则引擎（YAML 配置化）+ 轻量 ML + ChromaDB RAG + 注入检测 + 答案校验 + 人工复核闭环。

> **项目背景**：基于真实 12315 投诉工单开发（抽样 200 条分析发现约 70% 可由规则覆盖、20% 为明确职责外事项、10% 为灰色地带），据此设计"规则引擎 → ML → RAG → LLM"四层分级调度——**确定性越高的方案越靠前，MVP 阶段 0 LLM Token 消耗**。系统自动完成投诉分类、职责判定、法规检索、机构分派与回复草稿生成，低置信度/敏感场景自动转人工复核。

## 🎯 项目演示

**一句话**：面向 12315 投诉场景、从 0 独立搭建的人机协同多智能体系统——自动完成受理判断、机构分派、法规检索与回复草稿，低置信 / 敏感场景自动转人工，MVP 阶段 **0 LLM Token**。

- ✅ **完整工程闭环，不是 Demo 脚本**：LangGraph 编排 + 人工复核 + 数据回流 + 全链路 Trace + **202 个测试全通过**，可一键运行、可回放、可评估。
- ✅ **为真实业务约束设计**：秉持"宁可转人工、不可误拒乱答"，规则→ML→RAG→LLM 四层分级，端到端准确率 **96.19%**、误拒率 **<2%**。
- ✅ **生产级可靠性**：注入检测 / PII 脱敏 / 答案校验三道护栏，LLM、向量库、单个 Agent 故障均可降级不中断，Docker Compose 一键部署。

### 界面演示（人工复核工作台）

**① 工作台首页** —— 左侧为智能问答助手与运行状态，右侧"是否受理分析"支持粘贴投诉原文、一键分析与人工分派机构。

<img src="docs/images/01-workbench-overview.png" width="760" alt="人工复核工作台首页">

**② 输入投诉原文** —— 例：消费者购买玩具后发现损坏，商家以"已拆封"为由拒绝退款，粘贴原文后点击"分析"。

<img src="docs/images/02-complaint-input.png" width="760" alt="输入投诉原文">

**③ 自动分析结果** —— 系统判定"建议受理"、置信度 **81%（命中 RULE 规则）**，自动分派至对应市场监管所（规则命中 100%），并生成 trace_id 全程可追溯；当 ML 模型文件缺失时，**自动降级为关键词规则引擎**（黄色提示），服务不中断。

<img src="docs/images/03-analysis-result.png" width="760" alt="自动分析结果：受理判定/置信度/分派/降级">

> 完整 REST 接口见运行后的 `/docs`（Swagger）；架构与工程细节见下文"核心能力"与"关键技术决策（FAQ）"。

## 核心指标

| 指标 | 数值 | 说明 |
|---|---|---|
| 分类准确率 | **96.19%** | 100 条手工标注 Golden Dataset（6 类场景）离线评估 |
| 误拒率 | < 2% | 低置信度拒答控制 |
| 分派正确率 | > 90% | 机构分派评估 |
| 测试用例 | **202 个全通过** | 单元 + 集成测试（`python -m pytest tests/`） |
| 注入检测 | 6 类 32 用例 | 指令覆盖 / 角色劫持 / 提示词窃取 / 强制受理 / 跳过法规 / 代码注入 |
| 部署方式 | Docker Compose 一键部署 | 支持纯规则（0 Token）与 LLM 增强双模式 |

## 架构

```mermaid
flowchart TD
    U(投诉输入) --> PG[Pre-Guard: PII脱敏 + 注入检测]
    PG --> CLS[Classifier: RuleEngine YAML + ML]
    CLS -->|ACCEPT| DSP[Dispatch + Retrieve 并行]
    CLS -->|REVIEW| RVW{人工复核}
    DSP --> RPL[Reply + AnswerVerifier]
    RPL --> OG[Post-Guard: 输出合规]
    OG --> TRC[Trace 审计 JSONL]
    RVW --> FB[(复核数据回流)]
    FB --> TD[(训练数据)]
    TD --> CLS
```

## 快速启动

### Docker（推荐）

```bash
docker compose up -d                  # 规则模式
docker compose --profile llm up -d    # 含 Ollama LLM
```

接口文档：`http://localhost:8000/docs`

### 本地开发

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## 核心能力

| 模块 | 说明 |
|------|------|
| **规则引擎配置化** | YAML 规则文件（accept/reject/dispatch/sensitive），RuleLoader + RuleEngine 替代硬编码关键词 |
| **双模式编排** | LangGraph DAG（规则基线）+ ReAct Agent（LLM tool-calling）|
| **并行 Agent 执行** | 受理路径下 dispatch 与 retrieve 并行 fan-out，降低延迟 |
| **6 个专家 Agent** | Classifier / RejectReason / Dispatch / Retrieval / Reply / Guardrails |
| **RAG 可信度** | RAG 评估集 50 条 + 评估器（Recall@K/MRR）+ AnswerVerifier 法规引用真实性校验 |
| **注入检测** | PromptInjectionDetector：6 类注入攻击检测（指令覆盖/角色劫持/提示词窃取/强制受理/跳过法规/代码注入）|
| **人机协同闭环** | 复核队列 API（pending/detail/reject/confirm）+ 训练数据冲突检测（label/decision/model 三类）|
| **全链路追踪** | AgentStep.decision_source + SSE 节点进度事件 + JSONL + `/metrics` Prometheus |
| **安全护栏** | GuardrailsAgent：输入 PII + 注入检测 + 输出过度承诺拦截 + 法规引用校验 |
| **评估框架** | Golden Dataset 100 条（6 类覆盖）+ 误拒率/复核率/回复合规率指标 |

## 关键技术决策（FAQ）

<details>
<summary><b>为什么规则优先，而不是直接上大模型？</b></summary>

抽样 200 条真实工单发现：约 70% 可由明确规则覆盖、20% 为职责外事项、仅 10% 是灰色地带。规则与轻量模型确定性高、可解释、可在内网离线运行且零成本；大模型只用于最不确定的生成环节，并始终受规则与答案校验约束，兼顾准确率、合规与成本。
</details>

<details>
<summary><b>如何防止大模型幻觉、错误受理或过度承诺？</b></summary>

三道关：① RAG 回复必须带法规出处；② AnswerVerifier 做引用真实性、降级检测、冲突检测、绝对化措辞、幻觉风险 5 项校验，不过则替换为兜底话术；③ 低置信度、职责外、敏感场景一律转人工复核。
</details>

<details>
<summary><b>LangGraph 如何编排？为什么要并行 fan-out？</b></summary>

用 StateGraph 定义共享状态与 preprocess→classify→(fan-out)→reply→validate→audit_log 节点，条件边按受理 / 不受理 / 复核路由；受理路径用线程池让"机构分派"与"法规检索"并行执行以降低延迟，最后汇聚。每个节点均有异常 try/except 与降级，单点故障不会拖垮全链路。
</details>

<details>
<summary><b>效果如何评估？怎么证明 96.19% 可信？</b></summary>

分层评估：202 个 pytest 保障功能正确性；100 条手工标注 Golden Dataset（覆盖 6 类场景）计算端到端准确率、误拒率、复核率、回复合规率；RAG 另设 50 条评估集计算 Recall@K / MRR；并统计接口延迟 p50/p95。
</details>

<details>
<summary><b>依赖服务故障时怎么办？</b></summary>

逐级降级：LLM 不可用 → 规则引擎；ChromaDB 不可用 → 关键词匹配；分类 / 分派 / 回复任一 Agent 异常 → 返回对应兜底结果并转人工。所有降级都通过 decision_source、Trace 与复核原因显式记录，做到"降级可见、可追踪"。
</details>

## 接口

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/api/v1/complaints/analyze` | 规则模式分析 |
| `POST` | `/api/v1/complaints/analyze-llm` | LLM ReAct Agent 分析 |
| `POST` | `/api/v1/complaints/analyze-llm/stream` | LLM Agent SSE 流式 |
| `POST` | `/api/v1/complaints/debate` | 双 Agent 交叉验证 |
| `GET` | `/api/v1/reviews/pending` | 待复核列表 |
| `GET` | `/api/v1/reviews/{trace_id}` | 复核详情 |
| `POST` | `/api/v1/reviews/{trace_id}/confirm` | 复核确认 |
| `POST` | `/api/v1/reviews/{trace_id}/reject` | 驳回复核 |
| `GET` | `/api/v1/reviews/stats` | 复核统计 |
| `GET` | `/api/v1/traces/{trace_id}` | 全链路追踪 |
| `GET` | `/health` `/readyz` | 健康检查 |
| `GET` | `/metrics` | Prometheus 指标 |
| `POST` | `/api/v1/reviews/export-training` | 导出训练数据 |

## 评估

```bash
# 快速抽查
python -m eval.run_evaluation --quick

# 完整评估 + 导出
python -m eval.run_evaluation --output eval/results.json

# RAG 专项评估
python -m eval.run_rag_eval
```

评估维度：分类准确率 / 分派正确率 / RAG 召回率 / 端到端 / 误拒率 / 复核率 / 回复合规率 / 延迟 p50 p95

## 安全

- **Pre-Guard**：PII 脱敏 + 注入攻击检测（6 类 32 个测试用例）
- **Post-Guard**：过度承诺拦截 + 法规引用校验 + 绝对化措辞检测
- **Rate Limit**：IP 滑动窗口限流（30 req/min），`/health` `/readyz` `/metrics` 白名单豁免
- **降级策略**：LLM 不可用 → 规则引擎；ChromaDB 不可用 → 关键词匹配

## 目录

```
app/
├── agents/          # Agent（含 RuleEngine/AnswerVerifier/PromptInjection）
├── api/routes.py    # REST 端点（含复核队列 API）
├── core/            # 配置/模型/日志/指标/RuleLoader/RateLimit
├── tools/           # 训练/评估/冲突检测/数据导入
├── static/          # 人工复核工作台 (HTML)
eval/                # 评估框架 + golden dataset 100 条 + RAG 评估器
data/
├── rules/           # YAML 规则文件（accept/reject/dispatch/sensitive）
├── security/        # 注入检测测试用例
├── dispatch/        # 分派规则映射表
├── knowledge/       # RAG 法规知识库
├── samples/         # 脱敏后的投诉工单示例（真实工单数据不公开）
tests/               # 202 个测试用例
```

## 数据与隐私合规

本项目基于**真实 12315 投诉工单**开发（青铜峡市市场监管局见习期间，抽样 200 条分析数据分布后设计分层调度架构），但：

- **真实工单数据不随仓库分发**：含个人信息（投诉人地址、消费记录、联系方式等）的原始工单、训练集、向量库已从本仓库及其 **git 全部历史** 中清除，本地保留于私有备份。
- `data/samples/complaint_samples.csv` 提供 10 条**脱敏示例**（保留真实工单的结构与表达习惯，姓名/电话/地址/店铺已替换为占位符），供复现数据形态与流程演示。
- 系统内置的 PII 脱敏（Pre-Guard）在运行时对输入投诉同样生效。
- 如需使用真实数据，请通过合法渠道获取并自行完成脱敏后接入。

## 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ORCHESTRATOR_BACKEND` | `langgraph` | 编排器（langgraph / simple）|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | LLM 服务地址 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM 模型名 |
| `USE_CHROMA_RETRIEVAL` | `true` | 启用向量检索 |
| `EMBEDDING_PROVIDER` | `hash` | 嵌入方案（hash / bge）|

## License

MIT
