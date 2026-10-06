import json

# 文件路径（请根据实际情况修改）
SAMPLE_FILE = "sample60_scores.json"    # 包含60条样本（有评分，但只作为句子列表）
SOURCE_FILE = "qwen_scores.json"         # 包含所有数据（有人工评分）
OUTPUT_FILE = "qwen_matched_60_ordered.json" # 输出按样本顺序匹配的数据

# 1. 读取样本文件
with open(SAMPLE_FILE, 'r', encoding='utf-8') as f:
    sample_data = json.load(f)   # 假设是列表

# 2. 读取源文件
with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
    source_data = json.load(f)   # 假设是列表

# 3. 为了高效查找，建立源数据中 sentence -> item 的映射（如果有重复句子，只保留第一个）
source_dict = {}
for item in source_data:
    sentence = item.get('sentence')
    if sentence and sentence not in source_dict:
        source_dict[sentence] = item

# 4. 按样本顺序匹配
matched = []
missing = []
for sample_item in sample_data:
    sent = sample_item.get('sentence')
    if sent in source_dict:
        matched.append(source_dict[sent])
    else:
        missing.append(sent)
        # 如果没有匹配，可以选择占位或跳过，这里先加一个空对象？最好跳过
        # 但为了顺序，可以加None或跳过；我们选择跳过，但必须保证最终长度一致。
        # 但题目要求提取相同的60条，应该都能匹配到。
        # 若确实缺失，则说明数据不一致，可打印警告。

print(f"成功匹配 {len(matched)} 条，缺失 {len(missing)} 条。")
if missing:
    print("缺失的句子：", missing)

# 5. 保存结果（只保存匹配到的，顺序与样本一致）
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(matched, f, ensure_ascii=False, indent=2)

print(f"已按样本顺序输出到 {OUTPUT_FILE}")