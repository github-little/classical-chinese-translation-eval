import json
from pathlib import Path

# ========== 1. 文件路径 ==========
INPUT_PATH = Path("deepseek_scores.json")
OUTPUT_PATH = Path("deepseek_scores_weighted631.json")

# ========== 2. 设置权重 ==========
WEIGHTS = {
    "信": 0.6,
    "达": 0.3,
    "雅": 0.1
}

# ========== 3. 读取原始数据 ==========
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

valid_count = 0
missing_count = 0

# ========== 4. 计算综合可信度 ==========
for record in data:

    scores = record.get("评分")

    # 跳过无评分记录，保留原始数据
    if not isinstance(scores, dict):
        missing_count += 1
        continue

    # 遍历所有模型
    for model_name, model_scores in scores.items():

        if not isinstance(model_scores, dict):
            missing_count += 1
            continue

        # 检查三个维度是否完整
        if not all(k in model_scores for k in WEIGHTS):
            missing_count += 1
            continue

        # 加权计算
        total_score = sum(
            float(model_scores[k]) * weight
            for k, weight in WEIGHTS.items()
        )

        # 新增综合可信度字段
        model_scores["综合可信度"] = round(total_score, 4)

        valid_count += 1

# ========== 5. 保存新文件 ==========
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        ensure_ascii=False,
        indent=2
    )

# ========== 6. 输出统计信息 ==========
print("加权评分计算完成！")
print(f"原始记录数：{len(data)}")
print(f"成功计算评分数：{valid_count}")
print(f"跳过的记录或评分数：{missing_count}")
print(f"输出文件：{OUTPUT_PATH.resolve()}")

# ========== 7. 计算各模型平均综合可信度 ==========
print("\n各模型平均综合可信度：")

model_results = {}

for record in data:
    for model, scores in (record.get("评分") or {}).items():

        if "综合可信度" in scores:
            model_results.setdefault(model, []).append(
                scores["综合可信度"]
            )

for model, values in model_results.items():

    average = sum(values) / len(values)

    print(
        f"{model}: {average:.4f} "
        f"(样本量={len(values)})"
    )