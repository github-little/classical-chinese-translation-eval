import json
import os


def convert_to_standard_json(input_file, output_file):
    # 读取原始 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    formatted_data = []

    for item in data:
        # 构建新的标准 JSON 对象
        new_item = {
            "sentence": item.get("sentence", ""),
            "翻译内容": item.get("翻译内容", ""),
            "模型翻译与评分": {}
        }

        translations = item.get("模型翻译", {})
        scores = item.get("人工评分", {})

        # 遍历每个模型的翻译，并将评分合并进去
        for model_name, translation in translations.items():
            model_scores = scores.get(model_name, {})

            new_item["模型翻译与评分"][model_name] = {
                "翻译": translation,
                "信": model_scores.get("信", ""),
                "达": model_scores.get("达", ""),
                "雅": model_scores.get("雅", "")
            }

        formatted_data.append(new_item)

    # 写入标准 JSON 文件（ensure_ascii=False 保证中文正常显示，indent=4 保证格式化缩进）
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    print(f"转换完成！已保存到 {output_file}")


if __name__ == "__main__":
    input_json_path = "person_eval.json"
    output_json_path = "person_eval_standard.json"

    if os.path.exists(input_json_path):
        convert_to_standard_json(input_json_path, output_json_path)
    else:
        print(f"找不到文件 {input_json_path}，请确保文件在同目录下。")