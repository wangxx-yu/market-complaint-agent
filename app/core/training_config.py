from __future__ import annotations

import os
from pathlib import Path


# 训练"是否受理模型"的默认样本文件。
# 可通过环境变量 ACCEPT_TRAINING_CSV 覆盖，默认从项目 data/ 目录读取。
ACCEPT_TRAINING_CSV = Path(
    os.getenv(
        "ACCEPT_TRAINING_CSV",
        str(Path("data/training/training_data_balanced.csv")),
    )
)

# 训练好的模型默认保存目录。
# 可通过环境变量 ACCEPT_MODEL_DIR 覆盖。
ACCEPT_MODEL_DIR = Path(
    os.getenv("ACCEPT_MODEL_DIR", str(Path("models/accept_v4")))
)
