#!/usr/bin/env python3
"""
批量调用 DeepSeek API 对古诗文模型译文进行“信达雅”评分。
支持断点续传，中断后可继续，不会重复打分。

断点重置方式：
- RESET_CHECKPOINT = True: 备份并删除输出文件，完全重新开始
- RESET_SCORES_ONLY = True: 保留输出文件，但清空所有已有评分
"""
import os
import json
import time
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ======================== 配置区域 ========================
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', "")
DEEPSEEK_MODEL = "deepseek-v4-flash"  # 或 "deepseek-reasoner"

INPUT_JSON = "merged_translate.json"
OUTPUT_JSON = "deepseek_scores.json"

# ========== 调试参数 ==========
DEBUG_LIMIT = None          # 设为数字（如 3）则只处理前 N 条；设为 None 处理全部
DEBUG_OVERWRITE = False     # True 强制覆盖已有评分，False 跳过已有评分

# ========== 断点重置参数 ==========
RESET_CHECKPOINT = False    # True: 备份并删除输出文件，完全重新开始
RESET_SCORES_ONLY = False   # True: 保留输出文件，但清空所有已有评分
# =============================================

MAX_RETRIES = 5
REQUEST_TIMEOUT = 300
REQUEST_INTERVAL = 1.0
MAX_COMPLETION_TOKENS = 4000
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# ======================== 配置结束 ========================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / INPUT_JSON
OUTPUT_PATH = SCRIPT_DIR / OUTPUT_JSON
CHAT_COMPLETIONS_URL = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"


def load_json(path: Path) -> List[Dict[str, Any]]:
    """加载 JSON 文件"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是数组")
    return data


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """原子保存 JSON，防止中断损坏"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def make_record_key(record: Dict[str, Any]) -> str:
    """
    生成记录唯一键，用于断点恢复。
    这里使用：原文 + 参考译文 + 模型译文内容，避免输入顺序变化导致评分错位。
    """
    sentence = str(record.get("sentence", "")).strip()
    reference = str(record.get("翻译内容", "")).strip()
    models = record.get("模型翻译", {})

    if isinstance(models, dict):
        model_pairs = sorted(
            (str(k).strip(), str(v).strip()) for k, v in models.items()
        )
    else:
        model_pairs = []

    if not sentence:
        return ""

    return json.dumps(
        {
            "sentence": sentence,
            "reference": reference,
            "models": model_pairs,
        },
        ensure_ascii=False,
    )


def clear_scores(records: List[Dict[str, Any]]) -> None:
    """清空所有记录中的评分字段"""
    for record in records:
        if isinstance(record, dict):
            record.pop("评分", None)


def reset_checkpoint() -> None:
    """
    重置断点：
    - 备份原输出文件
    - 删除原输出文件
    """
    if not OUTPUT_PATH.exists():
        print("无需重置断点：输出文件不存在")
        return

    backup_path = OUTPUT_PATH.parent / (OUTPUT_PATH.name + ".bak")
    if backup_path.exists():
        backup_path = OUTPUT_PATH.parent / (
            OUTPUT_PATH.name + f".{int(time.time())}.bak"
        )

    OUTPUT_PATH.replace(backup_path)
    print(f"已重置断点：原输出文件备份至 {backup_path}")


def build_prompt(record: Dict[str, Any], idx: int) -> str:
    """构建精简 Prompt，减少 Token 消耗"""
    original = record.get("sentence", "").strip()
    reference = record.get("翻译内容", "").strip()
    model_translations = record.get("模型翻译", {})

    if not original or not reference or not model_translations:
        raise ValueError("记录缺少必要字段（sentence / 翻译内容 / 模型翻译）")

    models_text = "\n".join(
        [f"- {name}: {text}" for name, text in model_translations.items()]
    )

    prompt = f"""请对以下古诗文译文的"信""达""雅"进行评分（1-5分）：

原文：{original}
参考译文：{reference}

模型译文：
{models_text}

评分标准：
信：1=严重错误，3=基本准确，5=完全忠实
达：1=流畅性差，3=基本流畅，5=极其规范自然
雅：1=意境保留差，3=适当渲染，5=强烈保留意境

只返回JSON：{{"模型名":{{"信":分,"达":分,"雅":分}}, ...}}"""

    return prompt


def call_deepseek(prompt: str) -> str:
    """调用 DeepSeek API，带重试机制"""
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是严谨的评分助手，只返回JSON。"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0.3,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(
                CHAT_COMPLETIONS_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")

            resp_data = json.loads(raw)

            # 检查错误
            if "error" in resp_data:
                err_msg = resp_data["error"].get("message", str(resp_data["error"]))
                if "rate limit" in err_msg.lower() or "overload" in err_msg.lower():
                    raise Exception(f"临时错误: {err_msg}")
                else:
                    raise RuntimeError(f"API 错误: {err_msg}")

            choices = resp_data.get("choices")
            if not choices or not isinstance(choices, list):
                raise ValueError("响应中无 choices")

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("响应 content 为空")

            return content.strip()

        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break

            wait = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5)
            print(f"  请求出错 (尝试 {attempt}/{MAX_RETRIES}): {e}，等待 {wait:.1f}s 重试")
            time.sleep(wait)

    raise RuntimeError(f"连续 {MAX_RETRIES} 次请求失败: {last_error}")


def parse_scores(raw: str, expected_models: List[str]) -> Dict[str, Dict[str, int]]:
    """解析 DeepSeek 返回的 JSON 评分"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试提取 JSON 片段
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as e:
                raise ValueError(f"无法解析 JSON: {raw[:200]}") from e
        else:
            raise ValueError(f"无法解析 JSON: {raw[:200]}")

    if not isinstance(data, dict):
        raise ValueError("返回 JSON 顶层不是对象")

    scores: Dict[str, Dict[str, int]] = {}

    for model in expected_models:
        model_clean = model.strip()
        found = None

        for key, val in data.items():
            if key.strip() == model_clean:
                found = val
                break

        if found is None:
            raise ValueError(f"模型 {model} 的评分缺失")

        if not isinstance(found, dict):
            raise ValueError(f"模型 {model} 的评分不是对象")

        score_dict: Dict[str, int] = {}

        for dim in ["信", "达", "雅"]:
            if dim in found:
                val = found[dim]
            else:
                # 尝试英文键名
                alt_keys = {
                    "信": "faithfulness",
                    "达": "fluency",
                    "雅": "elegance",
                }
                alt = alt_keys.get(dim)
                if alt and alt in found:
                    val = found[alt]
                else:
                    raise ValueError(f"模型 {model} 缺少 {dim} 维度")

            if isinstance(val, bool):
                raise ValueError(f"模型 {model} 的 {dim} 分数无效: {val}")

            if isinstance(val, (int, float)):
                int_val = int(round(val))
            elif isinstance(val, str) and val.isdigit():
                int_val = int(val)
            else:
                raise ValueError(f"模型 {model} 的 {dim} 分数无效: {val}")

            if 1 <= int_val <= 5:
                score_dict[dim] = int_val
            else:
                raise ValueError(
                    f"模型 {model} 的 {dim} 分数 {int_val} 不在 1-5"
                )

        scores[model_clean] = score_dict

    return scores


def get_processed_indices(
    records: List[Dict[str, Any]],
    overwrite: bool = False,
) -> Set[int]:
    """获取已处理的记录索引（已有评分且不覆盖）"""
    processed: Set[int] = set()

    if overwrite:
        return processed

    for i, record in enumerate(records):
        score = record.get("评分")
        if isinstance(score, dict) and score:
            processed.add(i)

    return processed


def main():
    if not INPUT_PATH.exists():
        print(f"错误：输入文件 {INPUT_PATH} 不存在")
        return

    # 加载输入数据
    input_records = load_json(INPUT_PATH)
    total_all = len(input_records)

    # 调试模式截取
    if DEBUG_LIMIT is not None:
        records = input_records[:DEBUG_LIMIT]
        print(f"调试模式：只处理前 {DEBUG_LIMIT} 条（共 {total_all} 条）")
    else:
        records = input_records
        print(f"共 {total_all} 条记录待处理")

    overwrite = DEBUG_OVERWRITE

    # 如果两个重置开关同时开启，优先执行完全重置
    if RESET_CHECKPOINT and RESET_SCORES_ONLY:
        print("警告：RESET_CHECKPOINT 与 RESET_SCORES_ONLY 同时开启，优先执行 RESET_CHECKPOINT")
        RESET_SCORES_ONLY = False

    # ========================
    # 初始化输出记录
    # ========================
    if RESET_CHECKPOINT:
        reset_checkpoint()
        output_records = records.copy()

    elif RESET_SCORES_ONLY:
        output_records = records.copy()

    elif OUTPUT_PATH.exists():
        try:
            old_records = load_json(OUTPUT_PATH)
            print(f"加载已有输出文件，共 {len(old_records)} 条记录")
        except Exception as e:
            print(f"警告：无法读取输出文件 ({e})，将重新生成")
            old_records = []

        # 建立旧评分索引
        old_scores: Dict[str, Dict[str, int]] = {}
        for rec in old_records:
            key = make_record_key(rec)
            if key and isinstance(rec.get("评分"), dict) and rec["评分"]:
                old_scores[key] = rec["评分"]

        # 按当前输入记录重建输出记录，并尽量恢复旧评分
        output_records = []
        recovered = 0

        for rec in records:
            new_rec = rec.copy()
            key = make_record_key(new_rec)

            if key and key in old_scores and not overwrite:
                new_rec["评分"] = old_scores[key]
                recovered += 1

            output_records.append(new_rec)

        if recovered:
            print(f"从旧输出文件中恢复了 {recovered} 条评分")

    else:
        output_records = records.copy()

    # ========================
    # 处理重置/覆盖/调试状态
    # ========================
    if RESET_CHECKPOINT or RESET_SCORES_ONLY or overwrite:
        clear_scores(output_records)
        save_json(OUTPUT_PATH, output_records)
        print("已重置评分状态，将重新评分")

    elif DEBUG_LIMIT is not None:
        save_json(OUTPUT_PATH, output_records)
        print("调试模式：输出文件已截断为当前调试记录")

    # ========================
    # 计算待处理记录
    # ========================
    processed_indices = get_processed_indices(output_records, overwrite=False)
    pending_indices = [
        i for i in range(len(output_records)) if i not in processed_indices
    ]
    total_pending = len(pending_indices)

    if total_pending == 0:
        print("所有记录已处理完成，无需评分")
        return

    print(f"已有 {len(processed_indices)} 条已评分，待处理 {total_pending} 条")

    # ========================
    # 开始评分
    # ========================
    processed_count = 0
    failed_indices: List[int] = []

    for pending_pos, idx in enumerate(pending_indices):
        record = output_records[idx]

        # 检查必要字段
        if (
            "sentence" not in record
            or "翻译内容" not in record
            or "模型翻译" not in record
        ):
            print(f"第 {idx + 1} 条记录缺少必要字段，跳过")
            failed_indices.append(idx + 1)
            continue

        model_translations = record.get("模型翻译", {})

        if not isinstance(model_translations, dict) or not model_translations:
            print(f"第 {idx + 1} 条记录没有有效模型译文，跳过")
            failed_indices.append(idx + 1)
            continue

        model_names = list(model_translations.keys())

        print(
            f"处理第 {idx + 1}/{len(output_records)} 条，"
            f"模型: {', '.join(model_names)}"
        )

        try:
            prompt = build_prompt(record, idx)
            raw_reply = call_deepseek(prompt)
            scores = parse_scores(raw_reply, model_names)

            record["评分"] = scores
            processed_count += 1

            # 每处理一条立即保存
            save_json(OUTPUT_PATH, output_records)
            print(f"  ✓ 第 {idx + 1} 条评分完成")

            # 请求间隔
            if pending_pos < len(pending_indices) - 1:
                time.sleep(REQUEST_INTERVAL)

        except Exception as e:
            print(f"  ✗ 第 {idx + 1} 条处理失败: {e}")
            failed_indices.append(idx + 1)

            # 即使失败也保存，避免丢失已处理的数据
            save_json(OUTPUT_PATH, output_records)
            continue

    # ========================
    # 总结
    # ========================
    print("\n" + "=" * 50)
    print(f"处理完成：新增评分 {processed_count} 条")
    print(f"跳过已评分 {len(processed_indices)} 条")

    if failed_indices:
        print(f"失败记录索引: {failed_indices}")

    print(f"结果已保存至 {OUTPUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()