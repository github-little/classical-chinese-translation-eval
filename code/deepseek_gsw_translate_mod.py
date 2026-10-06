#!/usr/bin/env python3
"""在 VS Code 中直接运行：按作品调用 DeepSeek，拆分译文并写回句级 JSON。"""

from __future__ import annotations
import os

import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ======================== 请在这里填写配置 ========================

# API 密钥从环境变量读取。
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', "")

# DeepSeek 官方模型：deepseek-v4-pro 或 deepseek-v4-flash。
DEEPSEEK_MODEL = "deepseek-v4-pro"

# 模型生成参数。用于学术评测时，建议保持较低温度以提高可复现性。
TEMPERATURE = 0.0
TOP_P = 1.0

# 这里是单篇作品允许生成的最大 token 数。
# 数据中最长篇目包含较多句段，因此默认值高于逐句翻译脚本。
MAX_TOKENS = 16000

# 生成译文在“模型翻译”字段下使用的键名。
# 建议每次实验使用一个新键名，既保留旧结果，也便于断点续跑。
OUTPUT_MODEL_KEY = "deepseek-v4-pro"

# 把本脚本和下面两个 JSON 文件放在同一个文件夹内。
SOURCE_FILE_NAME = "Final_gsw_new.json"
TARGET_FILE_NAME = "Final_gsw_chatgpt-5.6-sol.json"

# 按“作品”控制处理范围，而不是按句段控制。
# END_WORK_INDEX = None 表示处理到最后一篇作品。
START_WORK_INDEX = 0
END_WORK_INDEX = None

# 每完成多少篇作品保存一次。设为 1 最稳妥，异常退出后可直接续跑。
SAVE_EVERY_WORKS = 1

# 相邻作品请求之间的等待时间，可根据 DeepSeek 的限流情况调整。
REQUEST_INTERVAL_SECONDS = 0.3

# 长篇作品默认采用流式接收，避免等待完整响应时发生读取超时。
USE_STREAMING = True

# DeepSeek 默认开启思考模式。翻译任务建议关闭，以减少 token 消耗并稳定JSON格式。
THINKING_MODE = False

# 请求失败后的最大重试次数和单次“等待下一段数据”的超时秒数。
MAX_RETRIES = 6
REQUEST_TIMEOUT_SECONDS = 900

# False：某篇作品在 OUTPUT_MODEL_KEY 下已有完整结果时跳过。
# True：重新生成并覆盖该键下的已有结果。
OVERWRITE_EXISTING = False

# 一般不需要修改。
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# ======================== 配置结束 ========================


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / SOURCE_FILE_NAME
TARGET_PATH = SCRIPT_DIR / TARGET_FILE_NAME
CHAT_COMPLETIONS_URL = f"{DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
EMPTY_SENTENCE_RESULT = "（原句为空，无可译内容）"


def check_config() -> None:
    """在发送请求前检查必须填写的配置。"""
    if not DEEPSEEK_API_KEY.strip() or "请在这里填写" in DEEPSEEK_API_KEY:
        raise RuntimeError("请先设置 DEEPSEEK_API_KEY 环境变量。")
    if not DEEPSEEK_MODEL.strip():
        raise RuntimeError("DEEPSEEK_MODEL 不能为空。")
    if not OUTPUT_MODEL_KEY.strip():
        raise RuntimeError("OUTPUT_MODEL_KEY 不能为空。")
    if not 0.0 <= TEMPERATURE <= 2.0:
        raise RuntimeError("TEMPERATURE 必须在 0.0 到 2.0 之间。")
    if not 0.0 < TOP_P <= 1.0:
        raise RuntimeError("TOP_P 必须大于 0.0 且不超过 1.0。")
    if MAX_TOKENS <= 0:
        raise RuntimeError("MAX_TOKENS 必须大于 0。")
    if START_WORK_INDEX < 0:
        raise RuntimeError("START_WORK_INDEX 不能小于 0。")
    if END_WORK_INDEX is not None and END_WORK_INDEX < START_WORK_INDEX:
        raise RuntimeError("END_WORK_INDEX 不能小于 START_WORK_INDEX。")
    if SAVE_EVERY_WORKS <= 0:
        raise RuntimeError("SAVE_EVERY_WORKS 必须大于 0。")
    if MAX_RETRIES <= 0:
        raise RuntimeError("MAX_RETRIES 必须大于 0。")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """读取顶层为数组的 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path.name} 的 JSON 顶层必须是数组。")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path.name} 的数组元素必须全部是对象。")
    return data


def validate_and_align(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> None:
    """确认源文件与目标文件逐条对应，避免译文写错位置。"""
    if len(source_rows) != len(target_rows):
        raise ValueError(
            "源文件与目标文件的记录数不一致："
            f"{len(source_rows)} != {len(target_rows)}"
        )

    for row_index, (source, target) in enumerate(zip(source_rows, target_rows)):
        if not str(source.get("poem_id", "")).strip():
            raise ValueError(f"源文件第 {row_index} 条记录缺少 poem_id。")

        source_sentence = str(source.get("sentence", ""))
        target_sentence = str(target.get("sentence", ""))
        if source_sentence != target_sentence:
            raise ValueError(
                f"第 {row_index} 条记录的 sentence 不一致，已停止以防写错位置。"
            )

        source_translation = str(source.get("translation", ""))
        target_translation = str(target.get("翻译内容", ""))
        if source_translation != target_translation:
            raise ValueError(
                f"第 {row_index} 条记录的参考译文不一致，已停止以防数据错位。"
            )

        model_translations = target.get("模型翻译")
        if model_translations is not None and not isinstance(model_translations, dict):
            raise ValueError(f"目标文件第 {row_index} 条记录的“模型翻译”不是对象。")


def build_work_groups(
    source_rows: list[dict[str, Any]],
) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    """按 poem_id 分组，并按 sentence_index 排列每篇作品的句段。"""
    groups: OrderedDict[str, list[tuple[int, dict[str, Any]]]] = OrderedDict()
    for row_index, row in enumerate(source_rows):
        poem_id = str(row["poem_id"])
        groups.setdefault(poem_id, []).append((row_index, row))

    metadata_fields = ("poem_title", "author", "dynasty", "category", "grade")
    result: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []

    for poem_id, items in groups.items():
        indices = [row.get("sentence_index") for _, row in items]
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(f"作品 {poem_id} 存在非整数 sentence_index。")
        if len(indices) != len(set(indices)):
            raise ValueError(f"作品 {poem_id} 存在重复 sentence_index。")

        items.sort(key=lambda item: item[1]["sentence_index"])
        first_row = items[0][1]
        title = str(first_row.get("poem_title", ""))

        for field in metadata_fields:
            values = {str(row.get(field, "")) for _, row in items}
            if len(values) != 1:
                raise ValueError(f"作品《{title}》的 {field} 字段不一致。")

        for _, row in items:
            segment_total = row.get("segment_total")
            if segment_total is not None and segment_total != len(items):
                raise ValueError(
                    f"作品《{title}》的 segment_total={segment_total}，"
                    f"但实际句段数为 {len(items)}。"
                )

        result.append((poem_id, items))

    return result


def build_work_prompt(items: list[tuple[int, dict[str, Any]]]) -> str:
    """构造整篇翻译提示词；绝不向模型发送参考译文。"""
    first_row = items[0][1]
    sentences = [str(row.get("sentence", "")).strip() for _, row in items]
    sentence_map = {str(index): sentence for index, sentence in enumerate(sentences)}
    sentence_json = json.dumps(
        sentence_map,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    last_id = len(sentences) - 1

    return (
        "你是一名古诗文翻译专家。请结合篇名、作者及全文语境，将下面的古诗文全文"
        "准确翻译为现代汉语。原文是一个JSON对象，每个键都是不可更改的句段ID。"
        "请逐个ID翻译，不要合并、拆分、遗漏或调换句段。\n"
        f"输入共有{len(sentences)}项，ID从\"0\"到\"{last_id}\"。输出必须是严格合法的"
        "JSON对象，其中translations也必须是JSON对象；它必须有且仅有输入中的全部ID，"
        "每个ID的值必须是对应原文的现代汉语译文。即使两个相邻句段语义连续，也必须"
        "分别保留两个ID并分别填写译文。"
        "输出格式示例：{\"translations\":{\"0\":\"第0段译文\",\"1\":\"第1段译文\"}}。"
        "只输出这个JSON对象，不要输出Markdown代码块、序号、说明或其他内容。"
        "不要参考任何现成译文。\n"
        f"篇名：{str(first_row.get('poem_title', '')).strip()}\n"
        f"作者：{str(first_row.get('author', '')).strip()}\n"
        f"古诗文全文：{sentence_json}"
    )


def build_repair_prompt(
    items: list[tuple[int, dict[str, Any]]],
    missing_indices: list[int],
) -> str:
    """构造仅补译空缺ID的提示词，同时保留整篇语境。"""
    first_row = items[0][1]
    sentences = [str(row.get("sentence", "")).strip() for _, row in items]
    sentence_map = {str(index): sentence for index, sentence in enumerate(sentences)}
    missing_map = {str(index): sentences[index] for index in missing_indices}
    missing_id_text = "、".join(f'"{index}"' for index in missing_indices)
    example_output = {
        "translations": {
            str(index): f"第{index}段译文" for index in missing_indices
        }
    }
    sentence_json = json.dumps(
        sentence_map,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    missing_json = json.dumps(
        missing_map,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    example_json = json.dumps(
        example_output,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return (
        "你是一名古诗文翻译专家。上一轮整篇翻译中，下面列出的句段ID译文为空或缺失。"
        "请结合篇名、作者和完整原文语境，只补译这些缺失句段，不要重新输出其他ID。\n"
        f"必须补译的ID：{missing_id_text}。输出必须是严格合法的JSON对象，translations"
        "必须有且仅有这些ID，所有值都必须是非空的现代汉语译文。"
        f"输出格式示例：{example_json}。"
        "只输出JSON对象，不要输出Markdown代码块、说明或其他内容。"
        "不要参考任何现成译文。\n"
        f"篇名：{str(first_row.get('poem_title', '')).strip()}\n"
        f"作者：{str(first_row.get('author', '')).strip()}\n"
        f"完整古诗文语境：{sentence_json}\n"
        f"本次只需补译：{missing_json}"
    )


def extract_message_text(response_data: dict[str, Any]) -> str:
    """从 OpenAI 兼容响应中提取模型回复文本。"""
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "DeepSeek 返回格式异常：缺少 choices[0].message.content。"
        ) from error

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    else:
        text = str(content).strip()

    if not text:
        raise ValueError("DeepSeek 返回了空内容。")
    return text


def content_block_to_text(content: Any) -> str:
    """将流式响应中的字符串或文本块转换为文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def read_streaming_response(response: Any) -> str:
    """读取 OpenAI 兼容的 SSE 流，并拼接全部 content 增量。"""
    content_parts: list[str] = []
    fallback_lines: list[str] = []
    saw_sse_data = False
    reasoning_character_count = 0
    finish_reasons: list[str] = []

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue

        if not line.startswith("data:"):
            fallback_lines.append(line)
            continue

        saw_sse_data = True
        event_data = line[5:].strip()
        if event_data == "[DONE]":
            break

        try:
            chunk = json.loads(event_data)
        except json.JSONDecodeError as error:
            raise ValueError("DeepSeek 流式响应中出现无效 JSON 数据块。") from error

        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            error_text = json.dumps(chunk["error"], ensure_ascii=False)[:500]
            raise ValueError(f"DeepSeek 流式响应返回错误：{error_text}")

        try:
            choice = chunk["choices"][0]
        except (KeyError, IndexError, TypeError):
            continue

        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            finish_reasons.append(finish_reason)

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content_parts.append(content_block_to_text(delta.get("content")))
            reasoning_text = content_block_to_text(delta.get("reasoning_content"))
            reasoning_character_count += len(reasoning_text)
        elif isinstance(choice.get("message"), dict):
            # 兼容少数接口在流中返回完整 message 的情况。
            content_parts.append(
                content_block_to_text(choice["message"].get("content"))
            )
            reasoning_text = content_block_to_text(
                choice["message"].get("reasoning_content")
            )
            reasoning_character_count += len(reasoning_text)

    if saw_sse_data:
        text = "".join(content_parts).strip()
        if not text:
            details = []
            if reasoning_character_count:
                details.append(f"仅收到 {reasoning_character_count} 个思考字符")
            if finish_reasons:
                details.append(f"finish_reason={finish_reasons[-1]}")
            detail_text = "；".join(details) if details else "未收到content字段"
            raise ValueError(f"DeepSeek 流式响应结束但没有正文：{detail_text}。")
        return text

    # 兼容服务端忽略 stream 参数、仍返回普通 JSON 的情况。
    raw_response = "\n".join(fallback_lines).strip()
    if not raw_response:
        raise ValueError("DeepSeek 返回了空响应。")
    try:
        response_data = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError("DeepSeek 返回的内容不是有效 JSON。") from error
    if not isinstance(response_data, dict):
        raise ValueError("DeepSeek 返回的 JSON 顶层不是对象。")
    return extract_message_text(response_data)


def call_deepseek_once(prompt: str) -> str:
    """向 DeepSeek 官方接口发送一次整篇翻译请求。"""
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": USE_STREAMING,
        "thinking": {"type": "enabled" if THINKING_MODE else "disabled"},
        "response_format": {"type": "json_object"},
    }
    request = Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        if USE_STREAMING:
            return read_streaming_response(response)
        raw_response = response.read().decode("utf-8")

    try:
        response_data = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError("DeepSeek 返回的内容不是有效 JSON。") from error

    if not isinstance(response_data, dict):
        raise ValueError("DeepSeek 返回的 JSON 顶层不是对象。")
    return extract_message_text(response_data)


def remove_markdown_fence(text: str) -> str:
    """移除模型偶尔附加的 Markdown 代码块标记。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class IncompleteTranslationsError(ValueError):
    """回复包含部分有效译文，但仍有可单独补译的ID。"""

    def __init__(
        self,
        message: str,
        partial_translations: dict[int, str],
        missing_indices: list[int],
    ) -> None:
        super().__init__(message)
        self.partial_translations = partial_translations
        self.missing_indices = missing_indices


def normalize_translation_item(item: Any) -> str:
    """将模型返回的单项译文规范化为字符串。"""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        value = item.get("translation", item.get("text"))
        return value.strip() if isinstance(value, str) else ""
    return ""


def parse_translation_array(raw_text: str, expected_count: int) -> list[str]:
    """从回复中提取译文，优先校验固定ID映射，并兼容旧版数组。"""
    cleaned = remove_markdown_fence(raw_text)
    if not cleaned:
        raise ValueError("模型回复为空。")

    def validate_candidate(parsed: Any) -> list[str]:
        # 新格式为 {"translations": {"0": "...", ...}}。
        # 同时兼容旧版 {"translations": [...]} 和顶层数组，避免接口偶尔偏离格式。
        missing_ids: list[str] = []
        if isinstance(parsed, dict):
            parsed = parsed.get("translations")

        if isinstance(parsed, dict):
            expected_ids = {str(index) for index in range(expected_count)}
            actual_ids = {str(key) for key in parsed}
            missing_ids = sorted(expected_ids - actual_ids, key=int)
            extra_ids = sorted(actual_ids - expected_ids)
            if extra_ids:
                raise ValueError(
                    "translations对象多出ID " + ", ".join(extra_ids) + "。"
                )
            ordered_items = [
                parsed.get(str(index)) for index in range(expected_count)
            ]
        elif isinstance(parsed, list):
            if len(parsed) != expected_count:
                raise ValueError(
                    f"模型返回 {len(parsed)} 项译文，但本篇需要 {expected_count} 项。"
                )
            ordered_items = parsed
        else:
            raise ValueError("JSON中没有translations对象或数组。")

        if len(ordered_items) != expected_count:
            raise ValueError(
                f"模型返回 {len(ordered_items)} 项译文，但本篇需要 {expected_count} 项。"
            )

        partial_translations: dict[int, str] = {}
        invalid_indices = {int(item_id) for item_id in missing_ids}
        for item_index, item in enumerate(ordered_items):
            translation = normalize_translation_item(item)
            if translation:
                partial_translations[item_index] = translation
            else:
                invalid_indices.add(item_index)

        if invalid_indices:
            missing_list = sorted(invalid_indices)
            missing_text = "、".join(str(index) for index in missing_list)
            raise IncompleteTranslationsError(
                f"模型返回的ID {missing_text} 译文为空或缺失。",
                partial_translations=partial_translations,
                missing_indices=missing_list,
            )

        return [partial_translations[index] for index in range(expected_count)]

    decoder = json.JSONDecoder()

    # 若整个回复已经是合法JSON，直接校验最外层结构。这样能保留“缺少ID”等精确信息，
    # 不会被嵌套对象产生的次要错误覆盖。
    try:
        complete_value = json.loads(cleaned)
    except json.JSONDecodeError:
        complete_value = None
    else:
        return validate_candidate(complete_value)

    # 若JSON前后附加了文字或多个JSON值，逐个寻找第一个完整对象或数组。
    candidate_errors: list[str] = []
    incomplete_errors: list[IncompleteTranslationsError] = []
    start_positions = [
        position for position, character in enumerate(cleaned) if character in "[{"
    ]
    for position in start_positions:
        try:
            parsed, _ = decoder.raw_decode(cleaned, position)
        except json.JSONDecodeError:
            continue
        try:
            return validate_candidate(parsed)
        except IncompleteTranslationsError as error:
            incomplete_errors.append(error)
        except ValueError as error:
            candidate_errors.append(str(error))

    if incomplete_errors:
        raise incomplete_errors[0]
    best_error = candidate_errors[0] if candidate_errors else "未找到完整JSON对象或数组"
    raise ValueError(f"无法提取符合要求的译文JSON：{best_error}")


def parse_repair_translations(
    raw_text: str,
    required_indices: list[int],
) -> dict[int, str]:
    """解析短补译回复，只接受指定ID且要求每项非空。"""
    cleaned = remove_markdown_fence(raw_text)
    if not cleaned:
        raise ValueError("补译回复为空。")

    required_ids = {str(index) for index in required_indices}

    def validate_candidate(parsed: Any) -> dict[int, str]:
        if isinstance(parsed, dict) and "translations" in parsed:
            parsed = parsed["translations"]

        if isinstance(parsed, dict):
            actual_ids = {str(key) for key in parsed}
            missing_ids = sorted(required_ids - actual_ids, key=int)
            extra_ids = sorted(actual_ids - required_ids)
            if missing_ids or extra_ids:
                details: list[str] = []
                if missing_ids:
                    details.append("缺少ID " + ", ".join(missing_ids))
                if extra_ids:
                    details.append("多出ID " + ", ".join(extra_ids))
                raise ValueError("补译ID不符合要求：" + "；".join(details) + "。")
            ordered_items = [parsed[str(index)] for index in required_indices]
        elif isinstance(parsed, list):
            if len(parsed) != len(required_indices):
                raise ValueError(
                    f"补译返回 {len(parsed)} 项，但需要 {len(required_indices)} 项。"
                )
            ordered_items = parsed
        else:
            raise ValueError("补译JSON中没有translations对象或数组。")

        repaired: dict[int, str] = {}
        for index, item in zip(required_indices, ordered_items):
            translation = normalize_translation_item(item)
            if not translation:
                raise ValueError(f"补译返回的ID {index} 仍为空或格式错误。")
            repaired[index] = translation
        return repaired

    try:
        complete_value = json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    else:
        return validate_candidate(complete_value)

    decoder = json.JSONDecoder()
    candidate_errors: list[str] = []
    for position, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned, position)
        except json.JSONDecodeError:
            continue
        try:
            return validate_candidate(parsed)
        except ValueError as error:
            candidate_errors.append(str(error))

    best_error = candidate_errors[0] if candidate_errors else "未找到完整JSON对象或数组"
    raise ValueError(f"无法提取符合要求的补译JSON：{best_error}")


def request_missing_translations(
    items: list[tuple[int, dict[str, Any]]],
    missing_indices: list[int],
) -> dict[int, str]:
    """仅请求空缺句段，避免因一项为空而重复生成整篇译文。"""
    prompt = build_repair_prompt(items, missing_indices)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            retry_prompt = prompt
            if attempt > 1:
                required_text = "、".join(str(index) for index in missing_indices)
                retry_prompt += (
                    f"\n再次强调：只补译ID {required_text}，每个值必须是非空字符串；"
                    "不要返回其他ID。"
                )
            raw_text = call_deepseek_once(retry_prompt)
            return parse_repair_translations(raw_text, missing_indices)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"HTTP {error.code}: {detail[:500]}"
            if error.code in {400, 401, 403, 404}:
                raise RuntimeError(
                    f"DeepSeek补译请求失败，且不会重试：{message}"
                ) from error
            last_error = RuntimeError(message)
        except (URLError, TimeoutError, ValueError) as error:
            last_error = error

        if attempt < MAX_RETRIES:
            wait_seconds = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5)
            print(
                f"  缺失句段补译失败（第 {attempt}/{MAX_RETRIES} 次）：{last_error}；"
                f"{wait_seconds:.1f} 秒后重试。"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"连续 {MAX_RETRIES} 次补译失败：{last_error}")


def request_work_translations(
    items: list[tuple[int, dict[str, Any]]],
) -> list[str]:
    """请求整篇译文；若仅有少量空缺，则保留成功项并进行短补译。"""
    prompt = build_work_prompt(items)
    expected_count = len(items)
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            retry_prompt = prompt
            if attempt > 1:
                retry_prompt += (
                    "\n再次强调：只返回{\"translations\":{\"0\":\"...\",...}}格式的"
                    f"合法JSON对象。translations必须有且仅有\"0\"到\"{expected_count - 1}\""
                    "这些ID；每个ID都不可遗漏，不要把相邻句段合并到同一个ID，也不要添加说明。"
                )
            raw_text = call_deepseek_once(retry_prompt)
            try:
                return parse_translation_array(raw_text, expected_count)
            except IncompleteTranslationsError as error:
                missing_text = "、".join(
                    str(index) for index in error.missing_indices
                )
                print(
                    f"  整篇回复中ID {missing_text} 为空或缺失；"
                    f"已保留其余 {len(error.partial_translations)} 项，"
                    "现在仅补译缺失句段。"
                )
                repaired = request_missing_translations(
                    items,
                    error.missing_indices,
                )
                merged = dict(error.partial_translations)
                merged.update(repaired)
                if len(merged) != expected_count:
                    raise ValueError(
                        f"合并补译后得到 {len(merged)} 项，但需要 {expected_count} 项。"
                    )
                return [merged[index] for index in range(expected_count)]
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"HTTP {error.code}: {detail[:500]}"

            # 这些错误通常无法通过重试解决，应立即提示检查配置或模型参数。
            if error.code in {400, 401, 403, 404}:
                raise RuntimeError(f"DeepSeek 请求失败，且不会重试：{message}") from error
            last_error = RuntimeError(message)
        except (URLError, TimeoutError, ValueError) as error:
            last_error = error

        if attempt < MAX_RETRIES:
            wait_seconds = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5)
            print(
                f"  请求或译文拆分失败（第 {attempt}/{MAX_RETRIES} 次）：{last_error}；"
                f"{wait_seconds:.1f} 秒后重试。"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"连续 {MAX_RETRIES} 次处理失败：{last_error}")


def work_is_complete(
    target_rows: list[dict[str, Any]],
    items: list[tuple[int, dict[str, Any]]],
) -> bool:
    """检查一篇作品的所有句段是否已有指定模型结果。"""
    for row_index, _ in items:
        model_translations = target_rows[row_index].get("模型翻译", {})
        value = model_translations.get(OUTPUT_MODEL_KEY) if isinstance(model_translations, dict) else None
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def atomic_save(path: Path, data: list[dict[str, Any]]) -> None:
    """先写临时文件再替换目标文件，避免保存中断导致文件损坏。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def main() -> None:
    check_config()
    source_rows = load_json_array(SOURCE_PATH)
    target_rows = load_json_array(TARGET_PATH)
    validate_and_align(source_rows, target_rows)
    work_groups = build_work_groups(source_rows)

    total_work_count = len(work_groups)
    stop_work_index = (
        total_work_count
        if END_WORK_INDEX is None
        else min(END_WORK_INDEX, total_work_count)
    )
    if START_WORK_INDEX >= stop_work_index:
        print("没有需要处理的作品，请检查 START_WORK_INDEX 和 END_WORK_INDEX。")
        return

    selected_groups = work_groups[START_WORK_INDEX:stop_work_index]
    selected_segment_count = sum(len(items) for _, items in selected_groups)
    saved_request_count = selected_segment_count - len(selected_groups)
    saved_percentage = (
        saved_request_count / selected_segment_count * 100
        if selected_segment_count
        else 0.0
    )

    print(f"源文件：{SOURCE_PATH}")
    print(f"目标文件：{TARGET_PATH}")
    print(
        f"处理范围：第 {START_WORK_INDEX} 至 {stop_work_index - 1} 篇，"
        f"共 {len(selected_groups)} 篇、{selected_segment_count} 个句段"
    )
    print(
        f"按篇调用预计请求 {len(selected_groups)} 次；相比逐句调用减少 "
        f"{saved_request_count} 次（{saved_percentage:.1f}%）。"
    )
    print(f"写入字段：模型翻译 -> {OUTPUT_MODEL_KEY}")

    generated_work_count = 0
    generated_segment_count = 0
    skipped_work_count = 0

    try:
        for relative_index, (poem_id, items) in enumerate(selected_groups):
            work_index = START_WORK_INDEX + relative_index
            first_row = items[0][1]
            title = str(first_row.get("poem_title", ""))

            if not OVERWRITE_EXISTING and work_is_complete(target_rows, items):
                skipped_work_count += 1
                print(
                    f"[{work_index + 1}/{stop_work_index}] 《{title}》已有完整结果，跳过。"
                )
                continue

            translatable_items = [
                item for item in items if str(item[1].get("sentence", "")).strip()
            ]
            print(
                f"[{work_index + 1}/{stop_work_index}] 正在翻译《{title}》："
                f"{len(items)} 个句段，1 次整篇请求（空缺时仅补译缺失句段）。"
            )

            if translatable_items:
                translations = request_work_translations(translatable_items)
            else:
                translations = []

            translation_iterator = iter(translations)
            for row_index, source_row in items:
                model_translations = target_rows[row_index].setdefault("模型翻译", {})
                if not isinstance(model_translations, dict):
                    raise ValueError(f"目标文件第 {row_index} 条记录的“模型翻译”不是对象。")

                sentence = str(source_row.get("sentence", "")).strip()
                if sentence:
                    model_translations[OUTPUT_MODEL_KEY] = next(translation_iterator)
                else:
                    model_translations[OUTPUT_MODEL_KEY] = EMPTY_SENTENCE_RESULT

            # 确保模型译文数组已被完整使用，没有数量错位。
            try:
                next(translation_iterator)
                raise RuntimeError(f"作品《{title}》存在未写入的多余译文。")
            except StopIteration:
                pass

            generated_work_count += 1
            generated_segment_count += len(items)

            if generated_work_count % SAVE_EVERY_WORKS == 0:
                atomic_save(TARGET_PATH, target_rows)
                print(
                    f"  已保存进度：本次完成 {generated_work_count} 篇、"
                    f"{generated_segment_count} 个句段。"
                )

            if REQUEST_INTERVAL_SECONDS > 0 and translatable_items:
                time.sleep(REQUEST_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n检测到手动停止，正在保存已完成的作品……")
        atomic_save(TARGET_PATH, target_rows)
        print("进度已保存，下次直接运行即可按作品续跑。")
        return
    except Exception:
        atomic_save(TARGET_PATH, target_rows)
        print("发生错误，已保存此前完成的作品。")
        raise

    atomic_save(TARGET_PATH, target_rows)
    print(
        "全部完成！"
        f"本次生成 {generated_work_count} 篇、{generated_segment_count} 个句段；"
        f"跳过已有完整结果 {skipped_work_count} 篇。"
    )


if __name__ == "__main__":
    main()
