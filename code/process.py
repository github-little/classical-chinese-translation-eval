#!/usr/bin/env python3
"""
批量调用 AGICTO（Claude Fable 5）对古诗文模型译文进行“信达雅”评分。
通过脚本内变量 DEBUG_LIMIT 和 DEBUG_OVERWRITE 控制调试行为。
"""
import os
import json
import time
import random
import re
from functools import total_ordering
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ======================== 配置区域 ========================
AGICTO_API_KEY = os.environ.get('AGICTO_API_KEY', "")
AGICTO_MODEL = "doubao-seed-2-0-pro-260215"
INPUT_JSON = "sample_60.json"
OUTPUT_JSON = "doubao_match_60_scores.json"
OVERWRITE_EXISTING = False       # 默认是否覆盖已有评分

# ========== 调试参数（直接修改这里） ==========
DEBUG_LIMIT = None   # 设为数字（如 3）则只处理前 N 条；设为 None 处理全部
DEBUG_OVERWRITE = None      # 设为 True 强制覆盖已有评分，设为 None 则使用 OVERWRITE_EXISTING
# =============================================

USE_JSON_MODE = True
MAX_RETRIES = 5
REQUEST_TIMEOUT = 300
REQUEST_INTERVAL = 0.5
MAX_COMPLETION_TOKENS = 4000
AGICTO_BASE_URL = "https://api.agicto.cn/v1/"
# ======================== 配置结束 ========================

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH = SCRIPT_DIR / INPUT_JSON
OUTPUT_PATH = SCRIPT_DIR / OUTPUT_JSON
CHAT_COMPLETIONS_URL = f"{AGICTO_BASE_URL.rstrip('/')}/chat/completions"


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("输入 JSON 顶层必须是数组")
    return data


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def build_prompt(record: Dict[str, Any]) -> str:
    original = record.get("sentence", "").strip()
    reference = record.get("翻译内容", "").strip()
    model_translations = record.get("模型翻译", {})

    if not original or not reference or not model_translations:
        raise ValueError("记录缺少必要字段（sentence / 翻译内容 / 模型翻译）")

    models_text = "\n".join([f"- {name}: {text}" for name, text in model_translations.items()])

    prompt = f"""你是古诗文翻译领域的专家，现在需要你根据参考译文对以下模型译文进行严格打分。
原始文本：{original}
参考译文：{reference}
以下是各模型译文：
{models_text}
请按照以下标准对每个模型译文进行"信""达""雅"三个维度的打分，分数为1-5的整数。
- 信：1分表示存在严重事实性或逻辑性错误，3分表示整体语义基本准确，5分表示严格忠实于原文。
- 达：1分表示语言流畅性差，3分表示基本流畅自然，5分表示极其规范自然。
- 雅：1分表示情感意境保留差，3分表示适当渲染，5分表示强烈保留氛围意境。
请只返回一个JSON对象，结构为 {{"模型名": {{"信": 分数, "达": 分数, "雅": 分数}}, ...}}，不要包含任何解释或其他文字。"""
    return prompt


def call_agicto(prompt: str) -> str:
    payload = {
        "model": AGICTO_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的评分助手，只返回JSON。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": False,
    }
    if USE_JSON_MODE:
        payload["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(
                CHAT_COMPLETIONS_URL,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {AGICTO_API_KEY}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            resp_data = json.loads(raw)
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

        except (HTTPError, URLError, TimeoutError, ValueError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5)
            print(f"  请求出错 (尝试 {attempt}/{MAX_RETRIES}): {e}，等待 {wait:.1f}s 重试")
            time.sleep(wait)

    raise RuntimeError(f"连续 {MAX_RETRIES} 次请求失败: {last_error}")


def parse_scores(raw: str, expected_models: List[str]) -> Dict[str, Dict[str, int]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except:
                raise ValueError(f"无法解析 JSON: {raw[:200]}")
        else:
            raise ValueError(f"无法解析 JSON: {raw[:200]}")

    if not isinstance(data, dict):
        raise ValueError("返回 JSON 顶层不是对象")

    scores = {}
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
        score_dict = {}
        for dim in ["信", "达", "雅"]:
            if dim not in found:
                raise ValueError(f"模型 {model} 缺少 {dim} 维度")
            val = found[dim]
            if isinstance(val, (int, float)):
                int_val = int(round(val))
                if 1 <= int_val <= 5:
                    score_dict[dim] = int_val
                else:
                    raise ValueError(f"模型 {model} 的 {dim} 分数 {int_val} 不在1-5")
            elif isinstance(val, str) and val.isdigit():
                int_val = int(val)
                if 1 <= int_val <= 5:
                    score_dict[dim] = int_val
                else:
                    raise ValueError(f"模型 {model} 的 {dim} 分数 {int_val} 不在1-5")
            else:
                raise ValueError(f"模型 {model} 的 {dim} 分数无效: {val}")
        scores[model_clean] = score_dict
    return scores










def main():
    # ... 前面加载 input 和 output 的代码不变 ...
    # 加载输入数据
    if not INPUT_PATH.exists():
        print(f"错误：输入文件 {INPUT_PATH} 不存在")
        return
    all_records = load_json(INPUT_PATH)
    total = len(all_records)
    print(f"输入文件共 {total} 条记录")

    # 加载或初始化输出数据
    if OUTPUT_PATH.exists():
        output_records = load_json(OUTPUT_PATH)

    else:
        output_records = all_records.copy()
        print("未找到输出文件，将新建")
    # 决定是否覆盖
    if DEBUG_OVERWRITE is not None:
        overwrite = DEBUG_OVERWRITE
    else:
        overwrite = OVERWRITE_EXISTING

    # ❌ 删除这段：
    # if DEBUG_LIMIT is not None:
    #     output_records = output_records[:DEBUG_LIMIT]

    processed = 0          # 实际调用 API 的次数
    skipped = 0
    failed_indices = []
    max_to_process = DEBUG_LIMIT   # None 表示无限制

    for idx, record in enumerate(output_records):
        # 如果已经达到调试限制，则停止处理（后续记录保持不变）
        if max_to_process is not None and processed >= max_to_process:
            print(f"已达到调试限制 {max_to_process}，停止处理后续记录")
            break
        # 判断是否需要跳过（已有完整评分且不覆盖）
        if not overwrite and "评分" in record:
            # 这里假设“评分”字段为字典，长度等于模型个数（可根据实际调整）
            # 原代码用 len == 11，可根据实际情况修改，这里保留原逻辑
            if len(record["评分"]) == 11:
                print(f"第 {idx+1} 条已有评分，跳过")
                skipped += 1
                processed += 1
                continue
            else:
                print(f"⚠️ 第 {idx+1} 条评分不完整，将重新处理")

        # 检查必要字段
        if "sentence" not in record or "翻译内容" not in record or "模型翻译" not in record:
            print(f"⚠️ 第 {idx+1} 条记录缺少必要字段，跳过")
            processed += 1
            continue



        model_names = list(record["模型翻译"].keys())
        if not model_names:
            print(f"⚠️ 第 {idx+1} 条记录没有模型译文，跳过")
            continue

        print(f"🔄 处理第 {idx+1}/{len(output_records)} 条：{record['sentence']}")

        try:
            prompt = build_prompt(record)
            raw_reply = call_agicto(prompt)
            scores = parse_scores(raw_reply, model_names)
            record["评分"] = scores
            processed += 1
            save_json(OUTPUT_PATH, output_records)   # 保存全量数据
            if idx < len(output_records) - 1:
                time.sleep(REQUEST_INTERVAL)
        except Exception as e:
            print(f"❌ 第 {idx+1} 条处理失败: {e}")
            failed_indices.append(idx+1)

    # ... 结尾统计和保存 ...


if __name__ == "__main__":
    main()