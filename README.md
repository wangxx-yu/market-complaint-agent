# 市场监管投诉智能处理系统 MVP

本项目实现一个本地优先、人机协同的投诉智能处理系统首版。它提供 FastAPI 接口、规则基线 Agent、历史 Excel 导入、Trace 追踪和人工复核回写能力。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m app.tools.ingest_excel --data-dir . --out data/runtime
uvicorn app.main:app --reload
```

接口文档：`http://127.0.0.1:8000/docs`

## 训练是否受理模型

默认训练样本路径配置在 `app/core/training_config.py`：

```python
ACCEPT_TRAINING_CSV = Path("C:/Users/wangxinwx/Desktop/模型构建训练项目/training_data_balanced.csv")
ACCEPT_MODEL_DIR = Path("models/accept_v4")
```

### 当前稳定模型：TF-IDF + LogisticRegression

这是当前系统默认使用的轻量模型，训练快，适合样本量还不大的阶段。可以直接运行：

```powershell
python -m app.tools.train_accept_model
```

如果以后换训练集，改 `ACCEPT_TRAINING_CSV`，或者临时用 `--csv` 覆盖。

训练完成后，`ClassifierAgent` 会优先使用 `ACCEPT_MODEL_DIR/accept_model.joblib` 做是否受理判断。接口返回里的 `"decision_source": "MODEL"` 表示这次分类来自训练模型；如果模型概率处于阈值中间区间，会返回 `REVIEW` 并要求人工复核。

### 实验模型：BERT

BERT 用来提升语义理解能力，适合后续和当前稳定模型做效果对比。它不会自动替换当前线上模型，默认保存到：

```text
models/accept_bert_v1
```

首次运行需要下载 `bert-base-chinese`，如果网络慢会等待较久。CPU 也能训练，但会比 GPU 慢。

```powershell
python -m app.tools.train_accept_bert `
  --csv "C:/Users/wangxinwx/Desktop/模型构建训练项目/training_data_balanced.csv" `
  --out-dir models/accept_bert_v1 `
  --epochs 3 `
  --batch-size 8
```

训练完成后，会生成：

- `models/accept_bert_v1/metrics.json`：准确率、混淆矩阵、各类别指标。
- `models/accept_bert_v1/test_predictions.csv`：测试集每条预测结果。
- `models/accept_bert_v1/test_errors.csv`：预测错误样本，后续重点复核。
- `models/accept_bert_v1/metadata.json`：模型说明和阈值。

预测一条投诉：

```powershell
python -m app.tools.predict_accept_bert `
  --model-dir models/accept_bert_v1 `
  --text "市民在饭店就餐发现食品变质，要求退款赔偿"
```

判断是否值得切换到 BERT，重点看 `metrics.json` 里的 `REJECT` 召回率和 `test_errors.csv`。如果不受理样本仍然很少，BERT 不一定比轻量模型稳定。

## 主要接口

- `POST /api/v1/complaints/analyze`：分析投诉，返回受理、分派、回复建议和执行链路。
- `POST /api/v1/reviews/{trace_id}/confirm`：人工确认或修正结果，并写入复核样本。
- `GET /api/v1/traces/{trace_id}`：查看全链路执行记录。
- `GET /health`：健康检查。

## 数据说明

`app.tools.ingest_excel` 会读取当前目录下的业务信息 Excel，生成：

- `data/runtime/complaints.jsonl`：标准化历史投诉样本。
- `data/runtime/address_aliases.json`：从历史处理机构和地址中挖掘的候选映射。
- `data/runtime/reply_templates.json`：从反馈内容中抽取的常见模板候选。

首版不会直接替代人工决定。低置信度、敏感词、默认分派、职责外、LLM 降级等场景都会触发人工复核。

## 人工维护分派规则

人工地址分派规则配置在：

```text
data/dispatch/manual_dispatch_rules.json
```

格式：

```json
{
  "某某小区": {
    "office_code": "QTX_YUMIN",
    "office_name": "裕民市场监管所",
    "confidence": 1.0,
    "support": "manual"
  }
}
```

这份文件优先级最高，适合手工补充或纠正地址分派。修改后重启服务生效。
