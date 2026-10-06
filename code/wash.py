import json

from sympy.parsing.sympy_parser import null


def add_blank_scores(input_file, output_file):
    # 读取原始 JSON 数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历每条数据
    for item in data:
        # 从 "模型翻译" 的键获取所有模型名称（保证一致性）
        models = list(item.get("模型翻译", {}).keys())
        # 构造评分字典，每个模型下的信达雅均为空字符串
        score_dict = {}
        for model in models:
            score_dict[model] = {
                "信": "",
                "达": "",
                "雅": ""
            }
        # 添加 "人工评分" 字段
        item["人工评分"] = score_dict

    # 写入新文件（保留中文字符）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    add_blank_scores("sample_60.json", "person_eval.json")
    print("处理完成，已生成 person_eval.json")