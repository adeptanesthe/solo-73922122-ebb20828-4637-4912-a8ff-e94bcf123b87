"""Ready-made payloads demonstrating the tricky cases required by the spec."""

SAMPLES = [
    {
        "id": "duplicate-tokens",
        "title": "重复 token（同一输入/目标多次出现，且跨微批重复）",
        "note": (
            "输入 4 号与目标 4 号在多处重复，测试 index_add 的梯度累加与"
            "共享 W 的输入/输出双重角色。"
        ),
        "payload": {
            "seed": 20240517,
            "scale": 1.0,
            "vocab_size": 256,
            "microbatches": [
                {
                    "name": "dup-A",
                    "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]],
                },
                {
                    "name": "dup-B",
                    "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]],
                },
            ],
        },
    },
    {
        "id": "uneven-valid",
        "title": "不等有效长度（含整批被忽略的微批）",
        "note": (
            "第一个微批仅 1 个有效目标，第二个全部 -100（应当跳过且不参与损失），"
            "第三个有 6 个有效目标；全局均值必须按“全部有效 token”加权。"
        ),
        "payload": {
            "seed": 991,
            "scale": 0.5,
            "vocab_size": 300,
            "microbatches": [
                {
                    "name": "short",
                    "tokens": [[1, 2]],
                },
                {
                    "name": "all-ignored",
                    "tokens": [[3, -100], [4, -100], [5, -100]],
                },
                {
                    "name": "long",
                    "tokens": [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, -100]],
                },
            ],
        },
    },
    {
        "id": "large-scores",
        "title": "大分数（scale=100，词表 4096）",
        "note": (
            "W 乘以 100 后 logits 可达数万，验证分块 logsumexp / 重算后向的"
            "数值稳定性和 float64 精度。"
        ),
        "payload": {
            "seed": 424242,
            "scale": 100.0,
            "vocab_size": 4096,
            "microbatches": [
                {
                    "name": "big-A",
                    "tokens": [[1, 4095], [2048, 0], [4095, 4095], [17, 33]],
                },
                {
                    "name": "big-B",
                    "tokens": [[255, 256], [256, 255], [1024, -100]],
                },
                {
                    "name": "big-C",
                    "tokens": [[0, 0], [4095, 1], [2047, 2048]],
                },
            ],
        },
    },
]


def _full_32_samples():
    """32 deterministic samples (max allowed), distinct listed ids per sample."""
    out = []
    for i in range(32):
        out.append(
            {
                "prefix": (13 * i + 1) % 512,
                "listed": [
                    [(29 * i + 3) % 512, 0.5],
                    [(53 * i + 7) % 512, 0.25],
                ],
                "rest": 0.25,
            }
        )
    return out


DISTILL_SAMPLES = [
    {
        "id": "distill-basic-mix",
        "title": "蒸馏·混合列出词（k=1..8、重复前缀、列出词含前缀自身）",
        "note": (
            "4 条样本：前缀 5 出现两次（梯度按行累加），列出词数从 1 到 8 不等，"
            "第一条的列出词包含前缀自身；教师概率均非负且合计为 1。"
        ),
        "payload": {
            "seed": 20240517,
            "scale": 1.0,
            "vocab_size": 256,
            "samples": [
                {"prefix": 5, "listed": [[5, 0.6], [9, 0.1]], "rest": 0.3},
                {"prefix": 5, "listed": [[7, 0.25], [8, 0.25], [9, 0.25]], "rest": 0.25},
                {"prefix": 200, "listed": [[0, 0.5]], "rest": 0.5},
                {"prefix": 17,
                 "listed": [[1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1],
                            [5, 0.1], [6, 0.1], [7, 0.1], [8, 0.1]],
                 "rest": 0.2},
            ],
        },
    },
    {
        "id": "distill-edge-masses",
        "title": "蒸馏·边界质量（其余类为 0、其余类为 1）",
        "note": (
            "第一条其余词总概率为 0（教师质量全部列出）；第二条列出词概率全为 0，"
            "其余类承担全部质量；第三条前缀与第二条相同。"
        ),
        "payload": {
            "seed": 991,
            "scale": 0.5,
            "vocab_size": 300,
            "samples": [
                {"prefix": 3, "listed": [[3, 0.7], [4, 0.3]], "rest": 0.0},
                {"prefix": 9, "listed": [[10, 0.0], [11, 0.0]], "rest": 1.0},
                {"prefix": 9, "listed": [[250, 0.45]], "rest": 0.55},
            ],
        },
    },
    {
        "id": "distill-large-scores-rest",
        "title": "蒸馏·大分数（scale=100，其余类概率下溢仍须算准）",
        "note": (
            "seed=424242、前缀 7 时学生 logits 的 argmax 正是 7 号词（logit≈182156，"
            "与次大值相差≈35474）。第一条把 7 号列入教师分布：其余类学生概率 "
            "exp(-35474) 在 float64 中下溢为 0，朴素 1-Σs 会得到 log(0)=-inf，"
            "而分块实现用 masked logsumexp 直接算 log s_rest，损失与梯度仍与"
            "稠密参考一致到 1e-8。第二、三条对照：其余类概率≈1、前缀重复。"
        ),
        "payload": {
            "seed": 424242,
            "scale": 100.0,
            "vocab_size": 4096,
            "samples": [
                {"prefix": 7, "listed": [[7, 0.9], [123, 0.05]], "rest": 0.05},
                {"prefix": 3000, "listed": [[1, 0.3], [2, 0.3]], "rest": 0.4},
                {"prefix": 7, "listed": [[759, 0.5], [3228, 0.2]], "rest": 0.3},
            ],
        },
    },
    {
        "id": "distill-full-32",
        "title": "蒸馏·满 32 条样本（上限）",
        "note": (
            "32 条样本达到条数上限，每条 2 个列出词（词号由确定性公式生成、"
            "样本内互不重复），其余类概率 0.25。"
        ),
        "payload": {
            "seed": 7,
            "scale": 2.0,
            "vocab_size": 512,
            "samples": _full_32_samples(),
        },
    },
]
