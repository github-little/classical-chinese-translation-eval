import json
import random

# 设置随机种子，使得每次运行结果一致（可选，注释掉则每次不同）
random.seed(42)  # 可改为任意整数

# 输入/输出文件路径
INPUT_FILE = "merged_translate.json"          # 请替换为您的实际文件路径
OUTPUT_FILE = "sample_60.json"    # 抽样结果保存路径

# 需要抽取的样本数量
SAMPLE_SIZE = 60

# 1. 读取原始数据
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查数据条数
total = len(data)
print(f"原始数据共 {total} 条记录。")

if total < SAMPLE_SIZE:
    print(f"警告：原始数据不足 {SAMPLE_SIZE} 条，将抽取全部数据。")
    sample = data
else:
    # 2. 随机抽取（无放回）
    sample = random.sample(data, SAMPLE_SIZE)
    print(f"成功抽取 {len(sample)} 条记录。")

# 3. 保存为新的 JSON 文件
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

print(f"抽样结果已保存至：{OUTPUT_FILE}")