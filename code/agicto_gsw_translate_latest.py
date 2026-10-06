#!/usr/bin/env python3
"""在 VS Code 中直接运行：按作品调用 AGICTO，拆分译文并写回句级 JSON。"""

from __future__ import annotations
import os

import json
import random
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# ======================== 请在这里填写配置 ========================

# API 密钥从环境变量读取。
AGICTO_API_KEY = os.environ.get('AGICTO_API_KEY', "")

# 当前实验使用 GLM-5.3。
AGICTO_MODEL = "kimi-k3"

# 模型生成参数。用于学术评测时，建议保持较低温度以提高可复现性。
TEMPERATURE = 0.0
TOP_P = 1.0

# 这里是单篇作品允许生成的最大 token 数。
# 数据中最长篇目包含较多句段，因此默认值高于逐句翻译脚本。
MAX_TOKENS = 16000

# 生成译文在“模型翻译”字段下使用的键名。
# 建议每次实验使用一个新键名，既保留旧结果，也便于断点续跑。
OUTPUT_MODEL_KEY = "kimi-k3"

# 把本脚本和下面两个 JSON 文件放在同一个文件夹内。
SOURCE_FILE_NAME = "Final_gsw_new.json"
TARGET_FILE_NAME = "Final_gsw_chatgpt-5.6-sol.json"

# 长篇降级分批时的断点缓存文件，脚本会自动创建和读取。
CHUNK_CACHE_FILE_NAME = "agicto_gsw_chunk_cache.json"

# 按“作品”控制处理范围，而不是按句段控制。
# END_WORK_INDEX = None 表示处理到最后一篇作品。
START_WORK_INDEX = 0
END_WORK_INDEX = None

# 每完成多少篇作品保存一次。设为 1 最稳妥，异常退出后可直接续跑。
SAVE_EVERY_WORKS = 1

# 相邻作品请求之间的等待时间，可根据 AGICTO 的限流情况调整。
REQUEST_INTERVAL_SECONDS = 1.0

# 长篇作品默认采用流式接收，避免等待完整响应时发生读取超时。
USE_STREAMING = True

# None：不向上游模型发送厂商专用的 enable_thinking 参数，兼容性最好。
# 只有在 AGICTO 模型页面明确说明支持该参数时，才改为 True 或 False。
ENABLE_THINKING = None

# 请求失败后的最大重试次数和单次“等待下一段数据”的超时秒数。
MAX_RETRIES = 8
REQUEST_TIMEOUT_SECONDS = 900

# 非JSON回复只进行少量“格式整理”请求，不重复执行完整翻译。
FORMAT_REPAIR_MAX_RETRIES = 3

# 超过该句段数的作品不再先请求整篇，而是固定均分为前后两次请求。
# 例如《陈涉世家》189段会拆为95段和94段。
TWO_REQUEST_THRESHOLD_SEGMENTS = 150

# 长篇连续过载后自动分批。普通篇目仍保持一次整篇请求。
LONG_WORK_THRESHOLD_SEGMENTS = 80
LONG_WORK_CHUNK_SIZE = 40
OVERLOAD_ATTEMPTS_BEFORE_CHUNKING = 3

# 服务过载、限流等临时错误使用更长退避，避免一两秒内连续撞到同一拥塞窗口。
OVERLOAD_RETRY_BASE_SECONDS = 5.0
MAX_RETRY_WAIT_SECONDS = 120.0

# False：某篇作品在 OUTPUT_MODEL_KEY 下已有完整结果时跳过。
# True：重新生成并覆盖该键下的已有结果。
OVERWRITE_EXISTING = False

# 一般不需要修改。
AGICTO_BASE_URL = "https://api.agicto.cn/v1/"

# ======================== 配置结束 ========================


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / SOURCE_FILE_NAME
TARGET_PATH = SCRIPT_DIR / TARGET_FILE_NAME
CHUNK_CACHE_PATH = SCRIPT_DIR / CHUNK_CACHE_FILE_NAME
CHAT_COMPLETIONS_URL = f"{AGICTO_BASE_URL.rstrip('/')}/chat/completions"
EMPTY_SENTENCE_RESULT = "（原句为空，无可译内容）"


def check_config() -> None:
    """在发送请求前检查必须填写的配置。"""
    if not AGICTO_API_KEY.strip() or "请在这里填写" in AGICTO_API_KEY:
        raise RuntimeError("请先设置 AGICTO_API_KEY 环境变量。")
    if not AGICTO_MODEL.strip() or "请在这里填写" in AGICTO_MODEL:
        raise RuntimeError("请先在脚本顶部填写 AGICTO_MODEL。")
    if not OUTPUT_MODEL_KEY.strip():
        raise RuntimeError("OUTPUT_MODEL_KEY 不能为空。")
    if OUTPUT_MODEL_KEY.strip() != AGICTO_MODEL.strip():
        raise RuntimeError(
            "OUTPUT_MODEL_KEY 必须与 AGICTO_MODEL 完全一致，"
            "避免将一个模型的译文误标为另一个模型。"
        )
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
    if FORMAT_REPAIR_MAX_RETRIES <= 0:
        raise RuntimeError("FORMAT_REPAIR_MAX_RETRIES 必须大于 0。")
    if TWO_REQUEST_THRESHOLD_SEGMENTS <= 0:
        raise RuntimeError("TWO_REQUEST_THRESHOLD_SEGMENTS 必须大于 0。")
    if LONG_WORK_THRESHOLD_SEGMENTS <= 0:
        raise RuntimeError("LONG_WORK_THRESHOLD_SEGMENTS 必须大于 0。")
    if LONG_WORK_CHUNK_SIZE <= 0:
        raise RuntimeError("LONG_WORK_CHUNK_SIZE 必须大于 0。")
    if OVERLOAD_ATTEMPTS_BEFORE_CHUNKING <= 0:
        raise RuntimeError("OVERLOAD_ATTEMPTS_BEFORE_CHUNKING 必须大于 0。")
    if OVERLOAD_RETRY_BASE_SECONDS <= 0:
        raise RuntimeError("OVERLOAD_RETRY_BASE_SECONDS 必须大于 0。")
    if MAX_RETRY_WAIT_SECONDS <= 0:
        raise RuntimeError("MAX_RETRY_WAIT_SECONDS 必须大于 0。")
    if ENABLE_THINKING is not None and not isinstance(ENABLE_THINKING, bool):
        raise RuntimeError("ENABLE_THINKING 只能是 None、True 或 False。")


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
        "回复的第一个字符必须是{，最后一个字符必须是}。只输出这个JSON对象，"
        "不要输出Markdown代码块、序号、说明或其他内容。"
        "不要参考任何现成译文。\n"
        f"篇名：{str(first_row.get('poem_title', '')).strip()}\n"
        f"作者：{str(first_row.get('author', '')).strip()}\n"
        f"古诗文全文：{sentence_json}"
    )


def build_chunk_prompt(
    all_items: list[tuple[int, dict[str, Any]]],
    chunk_start: int,
    chunk_end: int,
) -> str:
    """为长篇作品构造分批提示词；每批仍携带完整全文语境。"""
    first_row = all_items[0][1]
    all_sentences = [
        str(row.get("sentence", "")).strip() for _, row in all_items
    ]
    chunk_sentences = all_sentences[chunk_start:chunk_end]
    chunk_map = {
        str(local_index): sentence
        for local_index, sentence in enumerate(chunk_sentences)
    }
    full_context = json.dumps(
        all_sentences,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    chunk_json = json.dumps(
        chunk_map,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    last_local_id = len(chunk_sentences) - 1

    return (
        "你是一名古诗文翻译专家。下面是一篇较长的古诗文，系统为降低单次输出长度而"
        "分批请求。请结合提供的完整全文语境，只翻译本批句段。不要翻译语境中的其他"
        "句段，也不要参考任何现成译文。\n"
        f"本批对应全篇第{chunk_start}至{chunk_end - 1}个句段，共{len(chunk_sentences)}项；"
        f"本批局部ID从\"0\"到\"{last_local_id}\"。输出必须是严格合法的JSON对象，"
        "translations必须有且仅有本批全部局部ID，每个值为对应原文的现代汉语译文。"
        "不得合并、拆分、遗漏或调换句段。"
        "输出格式：{\"translations\":{\"0\":\"第0段译文\",\"1\":\"第1段译文\"}}。"
        "回复的第一个字符必须是{，最后一个字符必须是}，不要添加说明或代码块。\n"
        f"篇名：{str(first_row.get('poem_title', '')).strip()}\n"
        f"作者：{str(first_row.get('author', '')).strip()}\n"
        f"完整全文语境：{full_context}\n"
        f"本批需要翻译：{chunk_json}"
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


def build_format_repair_prompt(
    items: list[tuple[int, dict[str, Any]]],
    raw_reply: str,
) -> str:
    """要求模型只整理上一轮译文格式，尽量避免再次执行整篇翻译。"""
    first_row = items[0][1]
    sentences = [str(row.get("sentence", "")).strip() for _, row in items]
    sentence_map = {str(index): sentence for index, sentence in enumerate(sentences)}
    source_json = json.dumps(
        sentence_map,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    last_id = len(sentences) - 1

    return (
        "你只负责整理格式，不要重新翻译或润色。上一轮已经生成了古诗文现代汉语译文，"
        "但回复不是可解析的JSON。请根据句段ID和原文，将上一轮回复整理为严格合法的"
        "JSON对象。若上一轮确有某个ID缺失，才根据原文语境补齐该ID。\n"
        f"必须输出且仅输出ID从\"0\"到\"{last_id}\"的{len(sentences)}项译文，格式为："
        "{\"translations\":{\"0\":\"第0段译文\",\"1\":\"第1段译文\"}}。"
        "回复的第一个字符必须是{，最后一个字符必须是}，不要添加说明或代码块。\n"
        f"篇名：{str(first_row.get('poem_title', '')).strip()}\n"
        f"作者：{str(first_row.get('author', '')).strip()}\n"
        f"带ID原文：{source_json}\n"
        f"上一轮模型回复：\n{raw_reply.strip()}"
    )


class AGICTOResponseError(RuntimeError):
    """AGICTO返回业务错误或无法识别的响应结构，不应盲目重复计费请求。"""


class AGICTORetryableError(ValueError):
    """服务过载、限流或超时等适合等待后重试的临时错误。"""


def content_block_to_text(content: Any) -> str:
    """兼容 OpenAI、Anthropic 和 Responses 风格的文本块。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and str(part.get("type", "")).lower() in {
                "thinking",
                "reasoning",
                "reasoning_content",
            }:
                continue
            text_parts.append(content_block_to_text(part))
        return "".join(text_parts)
    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            if key in content:
                text = content_block_to_text(content[key])
                if text:
                    return text
    return ""


def describe_response_shape(value: Any, depth: int = 0) -> str:
    """只描述字段名和数据类型，不记录模型正文或用户提示词。"""
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, dict):
        parts = [
            f"{key}:{describe_response_shape(item, depth + 1)}"
            for key, item in list(value.items())[:12]
        ]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, list):
        item_shape = describe_response_shape(value[0], depth + 1) if value else "empty"
        return f"list[{len(value)}]({item_shape})"
    return type(value).__name__


def extract_api_error(response_data: dict[str, Any]) -> str | None:
    """提取HTTP 200响应中夹带的业务错误信息。"""
    if response_data.get("error") not in (None, "", False):
        error_value = response_data["error"]
        if isinstance(error_value, dict):
            message = error_value.get("message", error_value.get("detail"))
            code = error_value.get("code", error_value.get("type"))
            if message:
                return f"{code}: {message}" if code else str(message)
        return json.dumps(error_value, ensure_ascii=False)[:500]

    status = str(response_data.get("status", "")).lower()
    success = response_data.get("success")
    code = response_data.get("code")
    failed_code = isinstance(code, int) and code not in {0, 200}
    failed_code = failed_code or (
        isinstance(code, str)
        and code.strip().lower() not in {"", "0", "200", "ok", "success"}
    )
    if success is False or status in {"error", "failed", "failure"} or failed_code:
        message = response_data.get("message", response_data.get("msg", "未知错误"))
        return f"{code}: {message}" if code is not None else str(message)
    return None


def is_retryable_api_error(error_text: str) -> bool:
    """根据错误语义判断是否属于临时服务故障。"""
    normalized = error_text.casefold()
    retryable_markers = (
        "service load is too high",
        "try again later",
        "overloaded",
        "temporarily unavailable",
        "service unavailable",
        "server is busy",
        "rate limit",
        "too many requests",
        "request timeout",
        "gateway timeout",
        "服务负载过高",
        "服务繁忙",
        "稍后重试",
        "请求超时",
        "请求过多",
        "限流",
    )
    return any(marker in normalized for marker in retryable_markers)


def is_service_overload_error(error: Exception | None) -> bool:
    """判断临时错误是否属于可通过缩短单次输出缓解的服务过载。"""
    if not isinstance(error, AGICTORetryableError):
        return False
    normalized = str(error).casefold()
    overload_markers = (
        "service load is too high",
        "overloaded",
        "server is busy",
        "service unavailable",
        "capacity",
        "服务负载过高",
        "服务繁忙",
        "容量不足",
    )
    return any(marker in normalized for marker in overload_markers)


def retry_wait_seconds(attempt: int, error: Exception | None) -> float:
    """临时服务故障采用较长指数退避，其他可重试错误保持短退避。"""
    if isinstance(error, AGICTORetryableError):
        base_seconds = OVERLOAD_RETRY_BASE_SECONDS * (2 ** (attempt - 1))
        capped_seconds = min(base_seconds, MAX_RETRY_WAIT_SECONDS)
    else:
        capped_seconds = min(2 ** (attempt - 1), 30)
    return capped_seconds + random.uniform(0, 1.0)


def extract_message_text(response_data: dict[str, Any]) -> str:
    """从多种常见兼容响应结构中提取模型正文。"""
    api_error = extract_api_error(response_data)
    if api_error:
        if is_retryable_api_error(api_error):
            raise AGICTORetryableError(f"AGICTO临时服务错误：{api_error}")
        raise AGICTOResponseError(f"AGICTO返回业务错误：{api_error}")

    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                text = content_block_to_text(message.get("content")).strip()
                if text:
                    return text
                # 某些推理模型错误地只返回 reasoning_content；保留为最后兜底。
                reasoning_text = content_block_to_text(
                    message.get("reasoning_content")
                ).strip()
                if reasoning_text:
                    return reasoning_text
            text = content_block_to_text(choice.get("text")).strip()
            if text:
                return text
            delta = choice.get("delta")
            if isinstance(delta, dict):
                text = content_block_to_text(delta.get("content")).strip()
                if text:
                    return text

    # 兼容 Anthropic Messages、Responses API 以及网关增加data/result的包装。
    for key in ("output_text", "content", "output"):
        if key in response_data:
            text = content_block_to_text(response_data[key]).strip()
            if text:
                return text

    for key in ("data", "response"):
        nested = response_data.get(key)
        if isinstance(nested, dict):
            try:
                return extract_message_text(nested)
            except AGICTOResponseError:
                raise
        elif isinstance(nested, str) and nested.strip():
            return nested.strip()

    for key in ("result", "message"):
        if key in response_data:
            text = content_block_to_text(response_data[key]).strip()
            if text:
                return text

    shape = describe_response_shape(response_data)
    raise AGICTOResponseError(
        "AGICTO返回了无法识别的响应结构。"
        f"响应字段结构：{shape}。请核对模型名是否为Chat Completions模型。"
    )


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
            raise ValueError("AGICTO 流式响应中出现无效 JSON 数据块。") from error

        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            error_text = json.dumps(chunk["error"], ensure_ascii=False)[:500]
            raise ValueError(f"AGICTO 流式响应返回错误：{error_text}")

        try:
            choice = chunk["choices"][0]
        except (KeyError, IndexError, TypeError):
            # 兼容 Anthropic 风格 SSE：delta.text 或 content_block.text。
            top_delta = chunk.get("delta")
            if isinstance(top_delta, dict):
                content_parts.append(
                    content_block_to_text(
                        top_delta.get("text", top_delta.get("content"))
                    )
                )
            content_parts.append(content_block_to_text(chunk.get("content_block")))
            continue

        if not isinstance(choice, dict):
            continue
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            finish_reasons.append(finish_reason)

        delta = choice.get("delta")
        if isinstance(delta, dict):
            content_parts.append(
                content_block_to_text(
                    delta.get("content", delta.get("text"))
                )
            )
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
        else:
            content_parts.append(content_block_to_text(choice.get("text")))

    if saw_sse_data:
        text = "".join(content_parts).strip()
        if not text:
            details = []
            if reasoning_character_count:
                details.append(f"仅收到 {reasoning_character_count} 个思考字符")
            if finish_reasons:
                details.append(f"finish_reason={finish_reasons[-1]}")
            detail_text = "；".join(details) if details else "未收到content字段"
            raise ValueError(f"AGICTO 流式响应结束但没有正文：{detail_text}。")
        return text

    # 兼容服务端忽略 stream 参数、仍返回普通 JSON 的情况。
    raw_response = "\n".join(fallback_lines).strip()
    if not raw_response:
        raise ValueError("AGICTO 返回了空响应。")
    try:
        response_data = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError("AGICTO 返回的内容不是有效 JSON。") from error
    if not isinstance(response_data, dict):
        raise ValueError("AGICTO 返回的 JSON 顶层不是对象。")
    return extract_message_text(response_data)


def call_agicto_once(prompt: str) -> str:
    """向 AGICTO 的 OpenAI 兼容接口发送一次整篇翻译请求。"""
    payload = {
        "model": AGICTO_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": USE_STREAMING,
    }
    if ENABLE_THINKING is not None:
        payload["enable_thinking"] = ENABLE_THINKING
    request = Request(
        CHAT_COMPLETIONS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {AGICTO_API_KEY}",
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
        raise ValueError("AGICTO 返回的内容不是有效 JSON。") from error

    if not isinstance(response_data, dict):
        raise ValueError("AGICTO 返回的 JSON 顶层不是对象。")
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


class UnparseableTranslationsError(ValueError):
    """回复没有可直接解析的JSON，保留原文供格式修复请求使用。"""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def normalize_translation_item(item: Any) -> str:
    """将模型返回的单项译文规范化为字符串。"""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        value = item.get("translation", item.get("text"))
        return value.strip() if isinstance(value, str) else ""
    return ""


NUMBERED_TRANSLATION_PATTERN = re.compile(
    r"^\s*(?:[-+*]\s*)?(?:\*{1,2})?"
    r"(?:(?:ID|译文)\s*)?(?:第\s*)?"
    r"[\[\(（【{「『\"'“]?\s*(\d+)\s*"
    r"[\]\)）】}」』\"'”]?\s*(?:项|段|句|条)?"
    r"(?:\*{1,2})?\s*(?:[:：=、.．]|->|→)\s*(.+?)\s*$",
    re.IGNORECASE,
)


def clean_numbered_translation(value: str) -> str:
    """清理编号列表中包裹译文的引号、逗号和Markdown标记。"""
    cleaned = value.strip().rstrip(",，").strip()
    if cleaned.startswith("**") and cleaned.endswith("**") and len(cleaned) > 4:
        cleaned = cleaned[2:-2].strip()
    quote_pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))
    for left_quote, right_quote in quote_pairs:
        if (
            cleaned.startswith(left_quote)
            and cleaned.endswith(right_quote)
            and len(cleaned) > len(left_quote) + len(right_quote)
        ):
            cleaned = cleaned[len(left_quote) : -len(right_quote)].strip()
            break
    return cleaned


def parse_numbered_translations(
    raw_text: str,
    expected_count: int,
) -> list[str] | None:
    """兼容模型偶尔返回的“0：译文”或“1. 译文”编号文本。"""
    # 兼容模型把所有ID放在同一行、以分号分隔的情况。
    expanded = re.sub(
        r"[;；]\s*(?=(?:(?:ID|译文)\s*)?(?:第\s*)?\d+\s*(?:项|段|句|条)?\s*[:：=])",
        "\n",
        raw_text,
        flags=re.IGNORECASE,
    )
    entries: dict[int, str] = {}
    current_id: int | None = None

    for original_line in expanded.splitlines():
        line = original_line.strip()
        if not line or line.startswith("```"):
            continue
        match = NUMBERED_TRANSLATION_PATTERN.match(line)
        if match:
            item_id = int(match.group(1))
            if item_id in entries:
                return None
            translation = clean_numbered_translation(match.group(2))
            if not translation:
                return None
            entries[item_id] = translation
            current_id = item_id
        elif current_id is not None:
            # 允许同一译文自然换行；下一条编号出现时会自动切换ID。
            continuation = clean_numbered_translation(line)
            if continuation:
                entries[current_id] = f"{entries[current_id]} {continuation}".strip()

    zero_based_ids = set(range(expected_count))
    one_based_ids = set(range(1, expected_count + 1))
    actual_ids = set(entries)
    if actual_ids == zero_based_ids:
        translations = [entries[index].strip() for index in range(expected_count)]
    elif actual_ids == one_based_ids:
        translations = [entries[index].strip() for index in range(1, expected_count + 1)]
    else:
        return None

    return translations if all(translations) else None


def parse_translation_array(raw_text: str, expected_count: int) -> list[str]:
    """从回复中提取译文，优先校验固定ID映射，并兼容旧版数组。"""
    cleaned = remove_markdown_fence(raw_text)
    if not cleaned:
        raise ValueError("模型回复为空。")

    def validate_candidate(parsed: Any) -> list[str]:
        missing_ids: list[str] = []
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                pass
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
    try:
        complete_value = json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    else:
        return validate_candidate(complete_value)

    candidate_errors: list[str] = []
    incomplete_errors: list[IncompleteTranslationsError] = []
    for position, character in enumerate(cleaned):
        if character not in "[{":
            continue
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

    numbered_translations = parse_numbered_translations(cleaned, expected_count)
    if numbered_translations is not None:
        return numbered_translations

    if candidate_errors:
        raise ValueError(f"无法提取符合要求的译文JSON：{candidate_errors[0]}")

    preview = " ".join(cleaned.split())[:240]
    raise UnparseableTranslationsError(
        "模型回复中没有完整JSON，也不能按编号文本安全拆分。"
        f"回复开头：{preview!r}",
        raw_text=cleaned,
    )


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


def request_format_repair(
    items: list[tuple[int, dict[str, Any]]],
    raw_reply: str,
) -> list[str]:
    """只整理非JSON回复的格式，不重复进行整篇翻译。"""
    prompt = build_format_repair_prompt(items, raw_reply)
    expected_count = len(items)
    last_error: Exception | None = None

    for attempt in range(1, FORMAT_REPAIR_MAX_RETRIES + 1):
        try:
            retry_prompt = prompt
            if attempt > 1:
                retry_prompt += (
                    "\n再次强调：不要重新翻译，只把已有译文整理为规定的JSON对象。"
                )
            repaired_text = call_agicto_once(retry_prompt)
            try:
                return parse_translation_array(repaired_text, expected_count)
            except IncompleteTranslationsError as error:
                missing_text = "、".join(
                    str(index) for index in error.missing_indices
                )
                print(
                    f"  格式整理后ID {missing_text} 仍为空或缺失；"
                    "现在只补译这些ID。"
                )
                missing_translations = request_missing_translations(
                    items,
                    error.missing_indices,
                )
                merged = dict(error.partial_translations)
                merged.update(missing_translations)
                if len(merged) != expected_count:
                    raise ValueError(
                        f"格式整理及补译后得到 {len(merged)} 项，"
                        f"但需要 {expected_count} 项。"
                    )
                return [merged[index] for index in range(expected_count)]
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"HTTP {error.code}: {detail[:500]}"
            if is_retryable_api_error(message):
                last_error = AGICTORetryableError(message)
            elif error.code in {400, 401, 403, 404}:
                raise RuntimeError(
                    f"AGICTO格式整理请求失败，且不会重试：{message}"
                ) from error
            elif error.code in {408, 409, 425, 429} or error.code >= 500:
                last_error = AGICTORetryableError(message)
            else:
                last_error = RuntimeError(message)
        except (URLError, TimeoutError, ValueError) as error:
            last_error = error

        if attempt < FORMAT_REPAIR_MAX_RETRIES:
            wait_seconds = retry_wait_seconds(attempt, last_error)
            print(
                "  译文格式整理失败"
                f"（第 {attempt}/{FORMAT_REPAIR_MAX_RETRIES} 次）：{last_error}；"
                f"{wait_seconds:.1f} 秒后重试。"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"连续 {FORMAT_REPAIR_MAX_RETRIES} 次整理译文格式失败：{last_error}"
    )


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
            raw_text = call_agicto_once(retry_prompt)
            return parse_repair_translations(raw_text, missing_indices)
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"HTTP {error.code}: {detail[:500]}"
            if is_retryable_api_error(message):
                last_error = AGICTORetryableError(message)
            elif error.code in {400, 401, 403, 404}:
                raise RuntimeError(
                    f"AGICTO补译请求失败，且不会重试：{message}"
                ) from error
            elif error.code in {408, 409, 425, 429} or error.code >= 500:
                last_error = AGICTORetryableError(message)
            else:
                last_error = RuntimeError(message)
        except (URLError, TimeoutError, ValueError) as error:
            last_error = error

        if attempt < MAX_RETRIES:
            wait_seconds = retry_wait_seconds(attempt, last_error)
            print(
                f"  缺失句段补译失败（第 {attempt}/{MAX_RETRIES} 次）：{last_error}；"
                f"{wait_seconds:.1f} 秒后重试。"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"连续 {MAX_RETRIES} 次补译失败：{last_error}")


def load_chunk_cache() -> dict[str, Any]:
    """读取长篇分批缓存；文件不存在时返回空缓存。"""
    if not CHUNK_CACHE_PATH.exists():
        return {}
    with CHUNK_CACHE_PATH.open("r", encoding="utf-8") as file:
        cache = json.load(file)
    if not isinstance(cache, dict):
        raise ValueError(f"{CHUNK_CACHE_PATH.name} 的JSON顶层必须是对象。")
    return cache


def save_chunk_cache(cache: dict[str, Any]) -> None:
    """原子保存长篇分批缓存，避免中断造成缓存损坏。"""
    temporary_path = CHUNK_CACHE_PATH.with_suffix(CHUNK_CACHE_PATH.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(CHUNK_CACHE_PATH)


def request_work_in_chunks(
    items: list[tuple[int, dict[str, Any]]],
) -> list[str]:
    """将过载长篇分批翻译，每批保留全文语境并持久化完成结果。"""
    first_row = items[0][1]
    poem_id = str(first_row.get("poem_id", ""))
    title = str(first_row.get("poem_title", ""))
    sentences = [str(row.get("sentence", "")).strip() for _, row in items]
    total_count = len(items)
    total_chunks = (total_count + LONG_WORK_CHUNK_SIZE - 1) // LONG_WORK_CHUNK_SIZE
    cache_key = f"{OUTPUT_MODEL_KEY}::{poem_id}"
    cache = load_chunk_cache()

    cached_entry = cache.get(cache_key)
    if (
        OVERWRITE_EXISTING
        or not isinstance(cached_entry, dict)
        or cached_entry.get("sentences") != sentences
        or cached_entry.get("segment_total") != total_count
    ):
        cached_entry = {
            "model": OUTPUT_MODEL_KEY,
            "poem_id": poem_id,
            "poem_title": title,
            "segment_total": total_count,
            "sentences": sentences,
            "translations": {},
        }
        cache[cache_key] = cached_entry
        save_chunk_cache(cache)

    cached_translations = cached_entry.get("translations")
    if not isinstance(cached_translations, dict):
        cached_translations = {}
        cached_entry["translations"] = cached_translations

    print(
        f"  《{title}》自动分为 {total_chunks} 批，每批最多 "
        f"{LONG_WORK_CHUNK_SIZE} 个句段；每批均携带完整全文语境。"
    )

    for chunk_number, chunk_start in enumerate(
        range(0, total_count, LONG_WORK_CHUNK_SIZE),
        start=1,
    ):
        chunk_end = min(chunk_start + LONG_WORK_CHUNK_SIZE, total_count)
        required_keys = [str(index) for index in range(chunk_start, chunk_end)]
        chunk_is_cached = all(
            isinstance(cached_translations.get(key), str)
            and cached_translations[key].strip()
            for key in required_keys
        )
        if chunk_is_cached:
            print(
                f"  [长篇分批 {chunk_number}/{total_chunks}] "
                f"第 {chunk_start} 至 {chunk_end - 1} 段已有缓存，跳过。"
            )
            continue

        print(
            f"  [长篇分批 {chunk_number}/{total_chunks}] 正在翻译"
            f"第 {chunk_start} 至 {chunk_end - 1} 段。"
        )
        chunk_items = items[chunk_start:chunk_end]
        chunk_prompt = build_chunk_prompt(items, chunk_start, chunk_end)
        chunk_results = request_work_translations(
            chunk_items,
            prompt_override=chunk_prompt,
            allow_chunk_fallback=False,
        )
        if len(chunk_results) != chunk_end - chunk_start:
            raise RuntimeError(
                f"《{title}》第 {chunk_number} 批返回数量异常："
                f"{len(chunk_results)} != {chunk_end - chunk_start}"
            )

        for local_index, translation in enumerate(chunk_results):
            cached_translations[str(chunk_start + local_index)] = translation
        save_chunk_cache(cache)
        print(
            f"    第 {chunk_number}/{total_chunks} 批已保存到长篇缓存。"
        )

        if REQUEST_INTERVAL_SECONDS > 0 and chunk_number < total_chunks:
            time.sleep(REQUEST_INTERVAL_SECONDS)

    missing_indices = [
        index
        for index in range(total_count)
        if not isinstance(cached_translations.get(str(index)), str)
        or not cached_translations[str(index)].strip()
    ]
    if missing_indices:
        missing_text = "、".join(str(index) for index in missing_indices[:20])
        raise RuntimeError(f"《{title}》长篇缓存仍缺少句段：{missing_text}")

    return [cached_translations[str(index)].strip() for index in range(total_count)]


def request_work_in_two_parts(
    items: list[tuple[int, dict[str, Any]]],
) -> list[str]:
    """将超长作品固定均分为两次请求，并按原始句段顺序合并结果。"""
    first_row = items[0][1]
    poem_id = str(first_row.get("poem_id", ""))
    title = str(first_row.get("poem_title", ""))
    sentences = [str(row.get("sentence", "")).strip() for _, row in items]
    total_count = len(items)
    split_index = (total_count + 1) // 2
    part_ranges = [(0, split_index), (split_index, total_count)]
    cache_key = f"{OUTPUT_MODEL_KEY}::{poem_id}"
    cache = load_chunk_cache()

    cached_entry = cache.get(cache_key)
    if (
        OVERWRITE_EXISTING
        or not isinstance(cached_entry, dict)
        or cached_entry.get("sentences") != sentences
        or cached_entry.get("segment_total") != total_count
    ):
        cached_entry = {
            "model": OUTPUT_MODEL_KEY,
            "poem_id": poem_id,
            "poem_title": title,
            "segment_total": total_count,
            "sentences": sentences,
            "translations": {},
        }
        cache[cache_key] = cached_entry
        save_chunk_cache(cache)

    cached_translations = cached_entry.get("translations")
    if not isinstance(cached_translations, dict):
        cached_translations = {}
        cached_entry["translations"] = cached_translations

    first_count = part_ranges[0][1] - part_ranges[0][0]
    second_count = part_ranges[1][1] - part_ranges[1][0]
    print(
        f"  《{title}》超过 {TWO_REQUEST_THRESHOLD_SEGMENTS} 段，"
        f"固定拆为2次请求：前半 {first_count} 段，后半 {second_count} 段；"
        "两次请求均携带完整全文语境。"
    )

    for part_number, (part_start, part_end) in enumerate(part_ranges, start=1):
        required_keys = [str(index) for index in range(part_start, part_end)]
        part_is_cached = all(
            isinstance(cached_translations.get(key), str)
            and cached_translations[key].strip()
            for key in required_keys
        )
        if part_is_cached:
            print(
                f"  [固定两段 {part_number}/2] 第 {part_start} 至 "
                f"{part_end - 1} 段已有缓存，跳过。"
            )
            continue

        print(
            f"  [固定两段 {part_number}/2] 正在翻译第 {part_start} 至 "
            f"{part_end - 1} 段。"
        )
        part_items = items[part_start:part_end]
        part_prompt = build_chunk_prompt(items, part_start, part_end)
        part_results = request_work_translations(
            part_items,
            prompt_override=part_prompt,
            allow_chunk_fallback=False,
        )
        expected_part_count = part_end - part_start
        if len(part_results) != expected_part_count:
            raise RuntimeError(
                f"《{title}》第 {part_number} 次请求返回数量异常："
                f"{len(part_results)} != {expected_part_count}"
            )

        for local_index, translation in enumerate(part_results):
            cached_translations[str(part_start + local_index)] = translation
        save_chunk_cache(cache)
        print(f"    第 {part_number}/2 次请求结果已保存到长篇缓存。")

        if REQUEST_INTERVAL_SECONDS > 0 and part_number == 1:
            time.sleep(REQUEST_INTERVAL_SECONDS)

    missing_indices = [
        index
        for index in range(total_count)
        if not isinstance(cached_translations.get(str(index)), str)
        or not cached_translations[str(index)].strip()
    ]
    if missing_indices:
        missing_text = "、".join(str(index) for index in missing_indices[:20])
        raise RuntimeError(f"《{title}》两次请求完成后仍缺少句段：{missing_text}")

    return [cached_translations[str(index)].strip() for index in range(total_count)]


def request_work_translations(
    items: list[tuple[int, dict[str, Any]]],
    prompt_override: str | None = None,
    allow_chunk_fallback: bool = True,
) -> list[str]:
    """请求整篇译文；若仅有少量空缺，则保留成功项并进行短补译。"""
    prompt = prompt_override if prompt_override is not None else build_work_prompt(items)
    expected_count = len(items)
    last_error: Exception | None = None
    consecutive_overload_count = 0

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            retry_prompt = prompt
            if attempt > 1:
                retry_prompt += (
                    "\n再次强调：只返回{\"translations\":{\"0\":\"...\",...}}格式的"
                    f"合法JSON对象。translations必须有且仅有\"0\"到\"{expected_count - 1}\""
                    "这些ID；每个ID都不可遗漏，不要把相邻句段合并到同一个ID，也不要添加说明。"
                )
            raw_text = call_agicto_once(retry_prompt)
            try:
                return parse_translation_array(raw_text, expected_count)
            except UnparseableTranslationsError as error:
                print(
                    "  整篇译文未按JSON或编号格式返回；"
                    "现在只整理已有译文的格式，不重新翻译。"
                )
                return request_format_repair(items, error.raw_text)
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
            if is_retryable_api_error(message):
                last_error = AGICTORetryableError(message)
            elif error.code in {400, 401, 403, 404}:
                raise RuntimeError(f"AGICTO 请求失败，且不会重试：{message}") from error
            elif error.code in {408, 409, 425, 429} or error.code >= 500:
                last_error = AGICTORetryableError(message)
            else:
                last_error = RuntimeError(message)
        except (URLError, TimeoutError, ValueError) as error:
            last_error = error

        if is_service_overload_error(last_error):
            consecutive_overload_count += 1
        else:
            consecutive_overload_count = 0

        if (
            allow_chunk_fallback
            and expected_count >= LONG_WORK_THRESHOLD_SEGMENTS
            and consecutive_overload_count >= OVERLOAD_ATTEMPTS_BEFORE_CHUNKING
        ):
            print(
                f"  长篇连续 {consecutive_overload_count} 次发生服务过载，"
                "停止重复整篇请求，改用带全文语境的分批模式。"
            )
            return request_work_in_chunks(items)

        if attempt < MAX_RETRIES:
            wait_seconds = retry_wait_seconds(attempt, last_error)
            print(
                f"  请求或译文拆分失败（第 {attempt}/{MAX_RETRIES} 次）：{last_error}；"
                f"{wait_seconds:.1f} 秒后重试。"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"连续 {MAX_RETRIES} 次处理失败：{last_error}")


def request_work_with_length_policy(
    items: list[tuple[int, dict[str, Any]]],
) -> list[str]:
    """按句段数量选择单次整篇请求或固定两次请求。"""
    if len(items) > TWO_REQUEST_THRESHOLD_SEGMENTS:
        return request_work_in_two_parts(items)
    return request_work_translations(items)


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
    fixed_two_request_work_count = sum(
        1
        for _, items in selected_groups
        if sum(
            1
            for _, row in items
            if str(row.get("sentence", "")).strip()
        ) > TWO_REQUEST_THRESHOLD_SEGMENTS
    )
    planned_request_count = len(selected_groups) + fixed_two_request_work_count
    saved_request_count = selected_segment_count - planned_request_count
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
        f"正常情况下预计请求 {planned_request_count} 次；相比逐句调用减少 "
        f"{saved_request_count} 次（{saved_percentage:.1f}%）。"
    )
    print(f"写入字段：模型翻译 -> {OUTPUT_MODEL_KEY}")
    print(
        f"固定两次策略：大于 {TWO_REQUEST_THRESHOLD_SEGMENTS} 段的作品"
        "直接均分为前后两次请求，并按原顺序合并写回。"
    )
    print(
        f"长篇策略：不少于 {LONG_WORK_THRESHOLD_SEGMENTS} 段的作品连续过载 "
        f"{OVERLOAD_ATTEMPTS_BEFORE_CHUNKING} 次后，按每批最多 "
        f"{LONG_WORK_CHUNK_SIZE} 段自动续译。"
    )

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
            if len(translatable_items) > TWO_REQUEST_THRESHOLD_SEGMENTS:
                request_description = "固定分为前后2次请求"
            else:
                request_description = (
                    "先进行1次整篇请求（空缺时仅补译，长篇过载时自动分批）"
                )
            print(
                f"[{work_index + 1}/{stop_work_index}] 正在翻译《{title}》："
                f"{len(items)} 个句段，{request_description}。"
            )

            if translatable_items:
                translations = request_work_with_length_policy(translatable_items)
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
