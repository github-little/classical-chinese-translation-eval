import json

def merge_hy3_to_file(source_file, target_file, output_file=None):
    """
    将源文件中每个句子的 Hy3 翻译数据添加到目标文件中
    
    Args:
        source_file: 源文件路径 (包含 Hy3 数据)
        target_file: 目标文件路径 (需要添加 Hy3 数据)
        output_file: 输出文件路径 (如果为 None，则覆盖 target_file)
    """
    # 读取源文件
    with open(source_file, 'r', encoding='utf-8') as f:
        source_data = json.load(f)
    
    # 读取目标文件
    with open(target_file, 'r', encoding='utf-8') as f:
        target_data = json.load(f)
    
    # 构建源数据的查找字典 {sentence: hy3_translation}
    hy3_dict = {}
    for item in source_data:
        sentence = item.get('sentence', '').strip()
        if sentence:
            # 从模型翻译中提取 Hy3 的数据
            model_translations = item.get('模型翻译', {})
            hy3_value = model_translations.get('Hy3', '')
            hy3_dict[sentence] = hy3_value
    
    # 遍历目标数据，添加 Hy3 字段
    added_count = 0
    missing_count = 0
    
    for item in target_data:
        sentence = item.get('sentence', '').strip()
        if sentence in hy3_dict:
            # 如果已有模型翻译字段，在其中添加 Hy3
            if '模型翻译' not in item:
                item['模型翻译'] = {}
            item['模型翻译']['Hy3'] = hy3_dict[sentence]
            added_count += 1
        else:
            # 如果源文件中没有找到对应的 Hy3 翻译
            missing_count += 1
            print(f"警告: 未找到句子 '{sentence[:30]}...' 对应的 Hy3 翻译")
    
    # 确定输出文件路径
    if output_file is None:
        output_file = target_file
    
    # 写入目标文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(target_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成!")
    print(f"- 成功添加 Hy3 翻译: {added_count} 条")
    print(f"- 未找到对应翻译: {missing_count} 条")
    print(f"- 输出文件: {output_file}")

if __name__ == "__main__":
    source_file = "gsw_translate.json"  # 包含 Hy3 数据的源文件
    target_file = "Final_gsw_chatgpt-5.6-sol.json"  # 需要添加 Hy3 数据的目标文件
    
    # 方式1: 覆盖原文件
    # merge_hy3_to_file(source_file, target_file)
    
    # 方式2: 输出到新文件
    output_file = "merged_translate.json"
    merge_hy3_to_file(source_file, target_file, output_file)