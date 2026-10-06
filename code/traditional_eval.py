import json
import jieba
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import os
from comet import download_model, load_from_checkpoint


# -------------------- 工具函数 --------------------
def tokenize(text):
    """使用 jieba 分词，返回词列表"""
    return list(jieba.cut(text))


# 在 compute_bleu 函数中（替换原函数）
def compute_bleu(ref, cand):
    ref_tokens = tokenize(ref)
    cand_tokens = tokenize(cand)
    smooth = SmoothingFunction().method1
    bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    return {'BLEU-1': bleu1, 'BLEU-2': bleu2}


from rouge import Rouge

import re
def clean_text(text):
    return re.sub(r'[，。、；：！？\.,;:!?]', '', text)

def compute_rouge(ref, cand):
    # 去除标点（可选，但建议保留中文标点？实际去除使比较更干净）
    ref_clean = clean_text(ref)   # 已去除标点
    cand_clean = clean_text(cand)

    # 字符级：直接拆成单个字符，并用空格连接
    ref_chars = ' '.join(list(ref_clean))
    cand_chars = ' '.join(list(cand_clean))

    rouge = Rouge()
    scores = rouge.get_scores(cand_chars, ref_chars, avg=True)
    return {
        'rouge1': scores['rouge-1']['f'],
        'rouge2': scores['rouge-2']['f'],
        'rougeL': scores['rouge-l']['f']
    }

def compute_bert_score(ref, cand):
    """计算 BERTScore-F1（语言设为 zh）"""
    _, _, F1 = bert_score([cand], [ref], lang='zh', verbose=False)
    return F1.item()


def compute_comet(ref, src, cand, model):
    """计算 COMET 分数"""
    data_for_comet = [{"src": src, "ref": ref, "mt": cand}]
    scores = model.predict(data_for_comet, batch_size=1, progress_bar=False)
    return scores.scores[0]


# -------------------- 主处理函数 --------------------
def process_dataset(input_file, output_file, skip_comet=False):
    """
    读取 input_file，计算每条数据中所有模型的评分，
    将评分添加到 '评分' 字段，并写入 output_file。
    """
    # 1. 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 加载 COMET 模型（如果可用且不跳过）
    comet_model = None
    if not skip_comet:
        try:
            model_path = download_model("Unbabel/wmt20-comet-da")
            comet_model = load_from_checkpoint(model_path)
            print("COMET 模型加载成功。")
        except Exception as e:
            print(f"COMET 模型加载失败（{e}），将跳过 COMET 评分。")
            comet_model = None
    else:
        print("根据设置跳过 COMET 评分。")

    # 3. 遍历每条数据
    total = len(data)
    for idx, item in enumerate(data, 1):
        print(f"\n处理第 {idx}/{total} 条: {item.get('sentence', '')[:30]}...")
        ref = item['翻译内容']
        src = item['sentence']

        # 初始化该条数据的评分字典
        item['传统评分'] = {}

        # 对每个模型翻译计算评分
        for model_name, cand in item['模型翻译'].items():
            print(f"  计算模型 {model_name} ...")
            scores = {}
            try:
                bleu_scores = compute_bleu(ref, cand)
                scores['BLEU-1'] = bleu_scores['BLEU-1']
                scores['BLEU-2'] = bleu_scores['BLEU-2']
            except Exception as e:
                print(f"    BLEU 计算失败: {e}")
                scores['BLEU-1'] = None
                scores['BLEU-2'] = None

            try:
                rouge_scores = compute_rouge(ref, cand)
                scores['ROUGE-1'] = rouge_scores['rouge1']
                scores['ROUGE-2'] = rouge_scores['rouge2']
                scores['ROUGE-L'] = rouge_scores['rougeL']
            except Exception as e:
                print(f"    ROUGE 计算失败: {e}")
                for k in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']:
                    scores[k] = None

            try:
                scores['BERTScore-F1'] = compute_bert_score(ref, cand)
            except Exception as e:
                print(f"    BERTScore 计算失败: {e}")
                scores['BERTScore-F1'] = None

            if comet_model is not None:
                try:
                    scores['COMET'] = compute_comet(ref, src, cand, comet_model)
                except Exception as e:
                    print(f"    COMET 计算失败: {e}")
                    scores['COMET'] = None
            else:
                scores['COMET'] = None

            item['传统评分'][model_name] = scores

    # 4. 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！结果已保存至 {output_file}")


# -------------------- 命令行入口 --------------------
if __name__ == "__main__":
    # 这里请修改为你的输入文件路径和输出文件路径
    INPUT_JSON = "sample_60.json"  # 你的输入 JSON 文件
    OUTPUT_JSON = "sample60_scores.json"  # 输出 JSON 文件

    # 如果不想计算 COMET（因为模型大且慢），可设置 skip_comet=True
    process_dataset(INPUT_JSON, OUTPUT_JSON, skip_comet=False)