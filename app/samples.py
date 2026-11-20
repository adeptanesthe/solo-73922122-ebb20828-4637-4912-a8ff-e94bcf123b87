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
