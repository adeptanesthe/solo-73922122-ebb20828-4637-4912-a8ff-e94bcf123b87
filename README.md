# 共享词嵌入续写训练 · SGD 单步对照校验

面向文本模型训练工程师的**内存节省**实现与**一致性对照**工具，同一训练页支持两种模式：

* **硬标签续写**：损失为全部**有效**目标（目标号 `-100` 忽略）上的**平均交叉熵**。
* **教师分布蒸馏**：教师只给出 1–8 个续写词的概率和**其余词的总概率**；把未列出的
  词合为一类，按教师和学生的类别概率计算**样本平均交叉熵**。

共同设定：

* 同一个 float64 权重矩阵 `W`（种子 → 标准正态 → ×倍率）既作**输入嵌入**（查表），
  又作**输出投影**（`logits = h · Wᵀ`），即 weight tying；学生按**全词表**归一化。
* 只做**一次** lr = 0.01 的 SGD。
* 自定义 `torch.autograd.Function` 计算损失：
  * 投影临时张量的**词表轴 ≤ 128**（`MAX_VOCAB_CHUNK`）；
  * 前向**不保存**任何分块分数（logits）或概率，只保存共享 W、隐状态与目标信息；
    反向按块**重算** logits / logsumexp 再求梯度；
  * 硬标签模式下微批损失按各自有效 token 数加权后相加，等价于全局有效 token 平均。
* 独立的**稠密参考**实现不受限（完整 `[tokens, vocab]` logits + `log_softmax` /
  `logsumexp`）。
* 网页显示**损失 / 共享 W 梯度 / 更新后权重**三项对照误差，全部 ≤1e-8 才显示“通过”；
  硬标签模式额外验证把同一 token 流重新拆成 1–4 个微批时三项结果不变。
* 输入无效或计算失败时，网页**撤下旧结论**（清空判定与表格，只保留错误信息）。
* 内置样例：硬标签三个（重复 token、不等有效长度、大分数），蒸馏四个（混合列出词、
  边界质量、大分数其余类下溢、满 32 条样本）。

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

服务提供：

| 路径 | 说明 |
| --- | --- |
| `GET /` | 录入界面（硬标签 / 蒸馏两个标签页） |
| `GET /healthz` | 健康检查（Compose healthcheck 使用） |
| `GET /api/samples` | 三个硬标签内置样例 |
| `POST /api/run` | 运行硬标签对照，返回 JSON 报告 |
| `GET /api/distill/samples` | 四个蒸馏内置样例 |
| `POST /api/distill/run` | 运行蒸馏对照，返回 JSON 报告 |

### 本地运行

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 运行环境为 CPU 版 PyTorch（镜像中显式安装 `--index-url .../whl/cpu`）。
> 本项目用到的都是 float64 CPU 算子，但若本机默认装的是 CUDA 版 PyTorch，
> 数值结果一致。

## 输入格式

```json
{
  "seed": 20240517,
  "scale": 1.0,
  "vocab_size": 256,
  "microbatches": [
    {"name": "dup-A", "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]]},
    {"name": "dup-B", "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]]}
  ]
}
```

* `seed`：0 … 2⁶³−1 的整数；`torch.Generator` 固定种子生成
  `randn(vocab_size, 16, dtype=float64)`。
* `scale`：0.1 … 100（权重倍率）。
* `vocab_size`：256 … 4096；嵌入维度固定 16。
* `microbatches`：1 … 4 个；每项 `tokens` 为 `[输入号, 目标号]` 列表。
  输入号必须落在词表内；目标号为 `-100` 或词表内整数。
* 全部微批的 token 总数（含 `-100` 项）≤ 128。
* 若**所有**目标都是 `-100`，请求返回 422 且不做任何参数更新
  （损失无定义，拒绝运行）。
* 单微批简写：顶层直接给 `"tokens": [...]` 也可。

## 蒸馏输入格式

```json
{
  "seed": 20240517,
  "scale": 1.0,
  "vocab_size": 256,
  "samples": [
    {"prefix": 5, "listed": [[5, 0.6], [9, 0.1]], "rest": 0.3},
    {"prefix": 200, "listed": [[0, 0.5]], "rest": 0.5}
  ]
}
```

* `seed` / `scale` / `vocab_size`：与硬标签模式完全相同（沿用种子、词表大小与
  权重倍率，嵌入维度固定 16）。
* `samples`：1 … 32 条。每条包含：
  * `prefix`：前缀词号，落在词表内；
  * `listed`：1 … 8 个 `[词号, 教师概率]`，同一样本内词号**互不重复**、
    落在词表内，概率非负；
  * `rest`：其余词的总概率，非负。
* 每条样本的列出词概率之和 + `rest` 必须等于 1（容差 1e-9）。
* 学生分布：`hidden = W[prefix]`，`logits = hidden · Wᵀ`，按全词表 softmax。
  未列出的词合为一类，类别交叉熵按样本平均：

  ```
  loss = mean_s [ Σ_i p_i · (lse_s − z_{s,i})  +  rest_s · (lse_s − lse_rest_s) ]
  ```

  其中 `lse_s` 是全词表 logsumexp，`lse_rest_s` 是**未列出词**上的 logsumexp。
* 任一条目非法 → 422，不做任何参数更新；网页同时撤下旧结论。

### 为什么其余类在 scale=100 时仍然算得准

`scale=100` 时 logits 可达 ±1.8×10⁵，其余类的学生概率 `s_rest` 可能下溢为 0
（例如内置样例中 `exp(-35474)`）。实现不做 `log(1 − Σs_listed)`（会发生
`log(0)` 或灾难性抵消），而是：

* 损失：直接以 `lse_rest − lse` 得到 `log s_rest`（有限值，约 −3.5×10⁴）；
* 梯度：未列出词 `j` 的分量写成 `s_j − rest · q_j`，其中
  `q_j = exp(z_j − lse_rest)` 是其余类**内部**归一化的概率，避免 `0/0`。

稠密参考用同样的 masked-logsumexp 公式（完整词表、无分块），两者对照误差
≤1e-8（实测 ≤1e-14 量级）。

## 对照报告

`POST /api/run` 返回（节选）：

```jsonc
{
  "ok": true,
  "tolerance": 1e-8,
  "chunked": { "loss": 25.70, "grad_norm": 6.84, "weight_norm": 63.94 },
  "dense":   { "loss": 25.70, "grad_norm": 6.84, "weight_norm": 63.94 },
  "errors":  { "loss_abs_diff": 3.6e-15, "grad_max_abs_diff": 2.6e-15,
               "weight_max_abs_diff": 2.2e-16 },
  "split":   { "max_loss_diff": 3.6e-15, "max_grad_diff": 2.2e-16,
               "max_weight_diff": 0.0, "runs": [ ... ] },
  "checks":  { "loss": true, "grad": true, "updated_weight": true,
               "split_loss": true, "split_grad": true, "split_weight": true },
  "memory":  { "max_vocab_chunk": 128, "scores_or_probabilities_saved": false,
               "recompute_in_backward": true, ... }
}
```

* 三项主对照：`loss_abs_diff`、`grad_max_abs_diff`、`weight_max_abs_diff`
  （分块实现 vs 稠密参考）。
* 拆分不变量：把同一 token 流（保持顺序）重新均分为 1–4 个微批，各自跑一遍，
  与用户录入的微批结果比较，再取最大值。
* 只有六个 `checks` 全为 true，`ok` 才为 true，网页才显示“通过”。

`POST /api/distill/run` 的报告结构相同，只是没有 `split` 一节（蒸馏按样本平均、
不涉及微批拆分），`checks` 为 `loss` / `grad` / `updated_weight` 三项；
`memory.saved_shapes` 记录跨前向/反向保存的五个张量形状
（`W`、`hidden`、`listed_ids`、`listed_probs`、`rest_probs`）。

## 实现要点

```
app/engine.py       硬标签引擎：校验、W 生成、自定义 autograd、稠密参考、对照报告
app/distill.py      蒸馏引擎：top-k+其余类校验、分块重算 autograd、稠密参考、对照报告
app/main.py         FastAPI 路由
app/static/index.html  录入与结果界面（无外部依赖，两个标签页）
app/samples.py      硬标签三个 + 蒸馏四个内置样例
tests/test_engine.py   硬标签数值/约束/内存行为测试
tests/test_distill.py  蒸馏数值/边界/内存行为测试
tests/test_api.py      HTTP 层测试（两种模式）
```

自定义损失的核心：

```python
class ChunkedSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, hidden, target):
        # 1) 逐词表块算 logits = h @ W[chunk].T（词表轴 ≤128），
        #    logaddexp 累积每个 token 的全局 logsumexp；
        # 2) 只按块取目标 token 的 logit；
        # 3) loss = mean(lse - target_logit)；
        # 4) 只保存 weight / hidden / target / valid_idx。
    @staticmethod
    def backward(ctx, grad_output):
        # 用保存的 hidden 与 W 重算每块 logits，再 exp(logits - lse)，
        # 就地减去 one-hot 后按块累加 dW[start:end] += probs.T @ h，
        # dH += probs @ W[start:end]；不缓存前向的分数或概率。
```

* 重复输入号通过 `index_add_` 累加 `dH`，与 autograd 的 `index_select` 语义一致。
* `logaddexp` 分块累积避免“先拼完整分数矩阵”，对 scale=100 的大分数仍稳定。
* 微批按“有效 token 数 / 全局有效 token 数”加权，因此拆分数不改变结果。

蒸馏自定义损失的核心（`app/distill.py`）：

```python
class ChunkedDistillLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, hidden, listed_ids, listed_probs, rest_probs):
        # 1) 分块算全词表 lse 与”未列出词”的 masked lse_rest（词表轴 ≤128）；
        # 2) 按块取列出词 logit；loss = mean(Σ p·(lse−z) + rest·(lse−lse_rest))；
        # 3) 只保存 W / hidden / 列出词号 / 列出概率 / 其余类概率（[N,8] 填充）。
    @staticmethod
    def backward(ctx, grad_output):
        # 重算 lse 与 lse_rest；每块重构 s = exp(z−lse) 与 q = exp(z−lse_rest)，
        # 列出词位置 q 置 0、s 减 p，得 dL/dz = s − rest·q（列出位为 s−p），
        # 再按块累加 dW[start:end] += Gᵀ @ h，dH += G @ W[start:end]。
```

## 测试

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python3 -m pip install -r requirements.txt      # 运行服务
python3 -m pip install -r requirements-dev.txt  # 运行测试（含 pytest/httpx）
python3 -m pytest tests -q
```

覆盖：种子可复现、权重共享（与手工 tying 图对比）、`-100` 忽略、全忽略拒绝、
边界与超限校验、128 token 上限、微批拆分不变量、重复 token 梯度累加、
有限差分梯度校验、大分数数值精度、`save_for_backward` 不含分数/概率、词表分块 ≤128、
HTTP 路由与样例端到端通过；蒸馏侧另有：样本/列出词数量与去重校验、概率非负与
合计为 1（容差 1e-9）、其余类为 0 / 为 1、重复前缀梯度累加、其余类概率下溢时
（scale=100）损失与梯度仍与稠密参考一致、满 32 条样本、蒸馏 HTTP 路由与四个
内置样例端到端通过。

## 已知限制

* 仅 CPU、float64、单步 SGD；没有优化器状态、学习率调度或梯度裁剪。
* `scale` 在 W 生成时一次性乘上，等价于把整体 logits 放大；不做损失缩放。
* 网页不做并发限流，单次请求为同步计算（≤128 token、词表 ≤4096 时耗时很短）。
* 对照误差阈值固定 1e-8（float64 下实测约 1e-11 量级或更小）。
