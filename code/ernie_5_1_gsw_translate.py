#!/usr/bin/env python3
"""在 VS Code 中直接运行：通过百度千帆调用 ERNIE 5.1 批量翻译古诗文。"""

from __future__ import annotations

import json
import os
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
# 千帆 API Key 通常以 bce-v3/ 开头。
QIANFAN_API_KEY = os.environ.get('QIANFAN_API_KEY', "")

# 百度千帆官方模型列表中的 ERNIE 5.1 模型 ID。
QIANFAN_MODEL = "ernie-5.1"

# 写入“模型翻译”字段时使用的键名。
OUTPUT_MODEL_KEY = "ernie-5.1"

# 把脚本与源文件放在同一文件夹内。目标文件不存在时会自动创建。
SOURCE_FILE_NAME = "Final_gsw_new.json"
TARGET_FILE_NAME = "Final_gsw_chatgpt-5.6-sol.json"

# 按作品控制运行范围。END_WORK_INDEX=None 表示处理到最后一篇。
START_WORK_INDEX = 0
END_WORK_INDEX = None

# 翻译属于规则明确的任务，关闭思考可节省 reasoning token。
# 千帆文档说明 thinking 默认也是 disabled；仍显式发送，若服务端不接受则自动移除。
THINKING_TYPE = "disabled"  # 可选："disabled"、"enabled"
USE_THINKING_CONTROL = True

# 每批同时受句段数和原文字符数限制；任一达到上限就自动切下一批。
# 根据当前数据分布，100 段的《鸿门宴》一次请求，189 段的《陈涉世家》自动切分。
MAX_BATCH_SEGMENTS = 150
MAX_BATCH_SOURCE_CHARS = 12000

# 长篇分批时，每个待译句段最多携带前后各几个邻近句段作为语境。
# 不再为每一批重复发送整篇全文，可显著减少输入 token。
CONTEXT_SEGMENTS = 2

# ERNIE 5.1 官方 max_tokens 上限为 65536。thinking=disabled 时这里只限制正文输出；
# 它只是上限，不会预先消耗这些 token。
MAX_COMPLETION_TOKENS = 65536
MIN_COMPLETION_TOKENS = 4000

# 千帆支持 response_format=json_object；若接口版本不接受，脚本会自动移除，
# 并继续使用提示词约束和本地容错解析。
USE_JSON_MODE = True
USE_STREAMING = True

# 网络或限流错误才会重试；格式错误不会重复翻译整批。
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 900
REQUEST_INTERVAL_SECONDS = 0.8

# 本地解析完全失败时，只允许额外请求一次“整理格式”，不重新翻译。
ALLOW_ONE_FORMAT_REPAIR = True
MAX_FORMAT_REPAIR_INPUT_CHARS = 40000

# 模型漏项时只补缺失 ID，不重译已经取得的内容。补译首轮合并发送以节省输入
# token；若整轮没有取得任何新 ID，下一轮自动缩为逐句请求。
MAX_MISSING_REPAIR_ROUNDS = 4
MISSING_REPAIR_BATCH_SEGMENTS = 20
MISSING_REPAIR_BATCH_SOURCE_CHARS = 3000

# False：已有 ernie-5.1 结果时跳过；True：覆盖它，保留其他模型结果。
OVERWRITE_EXISTING = False

# 百度千帆文本生成接口，一般无需修改。
QIANFAN_CHAT_COMPLETIONS_URL = "https://qianfan.baidubce.com/v2/chat/completions"

# ======================== 配置结束 ========================


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / SOURCE_FILE_NAME
TARGET_PATH = SCRIPT_DIR / TARGET_FILE_NAME
EMPTY_SENTENCE_RESULT = "（原句为空，无可译内容）"
ERNIE_SYSTEM_PROMPT = (
    "你是严谨的古诗文翻译专家。只完成用户指定的翻译或格式整理任务，"
    "严格保持输入 ID，不输出解释、Markdown 或任务外内容。"
)


class TransientAPIError(ValueError):
    """适合等待后重试的临时 API 错误。"""


class ContentParseError(ValueError):
    """模型正文中没有可安全提取的译文。"""


def check_config() -> None:
    """检查运行配置，避免请求发出后才发现参数错误。"""
    if not QIANFAN_API_KEY.strip() or "请在这里填写" in QIANFAN_API_KEY:
        raise RuntimeError(
            "请先设置环境变量 QIANFAN_API_KEY，或在脚本顶部填写 QIANFAN_API_KEY。"
        )
    if not QIANFAN_MODEL.strip():
        raise RuntimeError("QIANFAN_MODEL 不能为空。")
    if not OUTPUT_MODEL_KEY.strip():
        raise RuntimeError("OUTPUT_MODEL_KEY 不能为空。")
    if THINKING_TYPE not in {"disabled", "enabled"}:
        raise RuntimeError('THINKING_TYPE 只能是 "disabled" 或 "enabled"。')
    if START_WORK_INDEX < 0:
        raise RuntimeError("START_WORK_INDEX 不能小于 0。")
    if END_WORK_INDEX is not None and END_WORK_INDEX < START_WORK_INDEX:
        raise RuntimeError("END_WORK_INDEX 不能小于 START_WORK_INDEX。")
    if MAX_BATCH_SEGMENTS <= 0 or MAX_BATCH_SOURCE_CHARS <= 0:
        raise RuntimeError("分批句段数和字符数必须大于 0。")
    if CONTEXT_SEGMENTS < 0:
        raise RuntimeError("CONTEXT_SEGMENTS 不能小于 0。")
    if MIN_COMPLETION_TOKENS <= 0:
        raise RuntimeError("MIN_COMPLETION_TOKENS 必须大于 0。")
    if MAX_COMPLETION_TOKENS < MIN_COMPLETION_TOKENS:
        raise RuntimeError("MAX_COMPLETION_TOKENS 不能小于 MIN_COMPLETION_TOKENS。")
    if MAX_RETRIES <= 0:
        raise RuntimeError("MAX_RETRIES 必须大于 0。")
    if MAX_MISSING_REPAIR_ROUNDS <= 0:
        raise RuntimeError("MAX_MISSING_REPAIR_ROUNDS 必须大于 0。")
    if MISSING_REPAIR_BATCH_SEGMENTS <= 0 or MISSING_REPAIR_BATCH_SOURCE_CHARS <= 0:
        raise RuntimeError("补译分批句段数和字符数必须大于 0。")


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """读取顶层为对象数组的 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path.name} 的 JSON 顶层必须是对象数组。")
    return data


def atomic_save(path: Path, data: list[dict[str, Any]]) -> None:
    """先保存临时文件再替换目标，避免中断造成 JSON 损坏。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def create_target_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按既定三字段结构创建新的模型译文文件。"""
    return [
        {
            "sentence": str(source.get("sentence", "")),
            "翻译内容": str(source.get("translation", "")),
            "模型翻译": {},
        }
        for source in source_rows
    ]


def load_or_create_target(
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """读取目标文件；首次运行时根据源文件自动创建。"""
    if TARGET_PATH.exists():
        target_rows = load_json_array(TARGET_PATH)
    else:
        target_rows = create_target_rows(source_rows)
        atomic_save(TARGET_PATH, target_rows)
        print(f"目标文件不存在，已自动创建：{TARGET_PATH.name}")
    validate_alignment(source_rows, target_rows)
    return target_rows


def validate_alignment(
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> None:
    """确认源文件与目标文件逐条对应，避免译文写错位置。"""
    if len(source_rows) != len(target_rows):
        raise ValueError(
            f"源文件与目标文件记录数不一致：{len(source_rows)} != {len(target_rows)}"
        )
    for row_index, (source, target) in enumerate(zip(source_rows, target_rows)):
        if str(source.get("sentence", "")) != str(target.get("sentence", "")):
            raise ValueError(f"第 {row_index} 条 sentence 不一致，已停止以防错位。")
        if str(source.get("translation", "")) != str(target.get("翻译内容", "")):
            raise ValueError(f"第 {row_index} 条参考译文不一致，已停止以防错位。")
        model_translations = target.get("模型翻译")
        if not isinstance(model_translations, dict):
            raise ValueError(f"目标文件第 {row_index} 条“模型翻译”必须是对象。")


def build_work_groups(
    source_rows: list[dict[str, Any]],
) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    """按 poem_id 分组，并按 sentence_index 排列每篇作品。"""
    groups: OrderedDict[str, list[tuple[int, dict[str, Any]]]] = OrderedDict()
    for row_index, row in enumerate(source_rows):
        poem_id = str(row.get("poem_id", "")).strip()
        if not poem_id:
            raise ValueError(f"源文件第 {row_index} 条缺少 poem_id。")
        groups.setdefault(poem_id, []).append((row_index, row))

    works: list[tuple[str, list[tuple[int, dict[str, Any]]]]] = []
    for poem_id, items in groups.items():
        indices = [row.get("sentence_index") for _, row in items]
        if any(not isinstance(index, int) for index in indices):
            raise ValueError(f"作品 {poem_id} 存在非整数 sentence_index。")
        if len(indices) != len(set(indices)):
            raise ValueError(f"作品 {poem_id} 存在重复 sentence_index。")
        items.sort(key=lambda item: item[1]["sentence_index"])
        title = str(items[0][1].get("poem_title", ""))
        for field in ("poem_title", "author"):
            values = {str(row.get(field, "")) for _, row in items}
            if len(values) != 1:
                raise ValueError(f"作品《{title}》的 {field} 字段不一致。")
        works.append((poem_id, items))
    return works


def get_model_map(target_row: dict[str, Any]) -> dict[str, Any]:
    """取得模型译文字典，并校验字段类型。"""
    model_map = target_row.get("模型翻译")
    if not isinstance(model_map, dict):
        raise ValueError("目标记录中的“模型翻译”必须是对象。")
    return model_map


def has_result(target_rows: list[dict[str, Any]], row_index: int) -> bool:
    """判断指定记录是否已有当前模型的非空译文。"""
    value = get_model_map(target_rows[row_index]).get(OUTPUT_MODEL_KEY)
    return isinstance(value, str) and bool(value.strip())


def sentence_text(items: list[tuple[int, dict[str, Any]]], position: int) -> str:
    return str(items[position][1].get("sentence", "")).strip()


def split_batches(
    items: list[tuple[int, dict[str, Any]]],
    positions: list[int],
    max_segments: int = MAX_BATCH_SEGMENTS,
    max_source_characters: int = MAX_BATCH_SOURCE_CHARS,
) -> list[list[int]]:
    """按句段数与字符数双重限制进行稳定分批。"""
    batches: list[list[int]] = []
    current: list[int] = []
    current_characters = 0
    for position in positions:
        text_characters = len(sentence_text(items, position))
        exceeds_limit = current and (
            len(current) >= max_segments
            or current_characters + text_characters > max_source_characters
        )
        if exceeds_limit:
            batches.append(current)
            current = []
            current_characters = 0
        current.append(position)
        current_characters += text_characters
    if current:
        batches.append(current)
    return batches


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def context_positions(
    items: list[tuple[int, dict[str, Any]]],
    positions: list[int],
) -> list[int]:
    """只选取每个待译句段附近的少量语境，避免重复发送整篇全文。"""
    if CONTEXT_SEGMENTS == 0:
        return []
    targets = set(positions)
    contexts: set[int] = set()
    for position in positions:
        start = max(0, position - CONTEXT_SEGMENTS)
        end = min(len(items), position + CONTEXT_SEGMENTS + 1)
        for nearby in range(start, end):
            if nearby not in targets and sentence_text(items, nearby):
                contexts.add(nearby)
    return sorted(contexts)


def build_translation_prompt(
    items: list[tuple[int, dict[str, Any]]],
    positions: list[int],
) -> str:
    """构造短提示词；只发送待译内容及少量邻近语境。"""
    first_row = items[0][1]
    targets = {str(position): sentence_text(items, position) for position in positions}
    context_ids = context_positions(items, positions)
    contexts = {
        str(position): sentence_text(items, position) for position in context_ids
    }
    prompt = (
        "将古诗文译为现代汉语。忠实、完整、通顺，不增添原文没有的信息；"
        "逐ID翻译，禁止合并、拆分。待译对象的键就是最终ID，必须原样复制；"
        "即使首个ID不是0，也不得从0重新编号。仅返回JSON对象，结构必须为"
        '{"translations":{"ID":"译文"}}，字符串必须符合标准JSON转义规则。'
        "不得参考任何现成译文。\n"
        f"篇名:{str(first_row.get('poem_title', '')).strip()}\n"
        f"作者:{str(first_row.get('author', '')).strip()}\n"
    )
    if contexts:
        prompt += f"邻近语境（只参考，不翻译）:{compact_json(contexts)}\n"
    prompt += f"待译:{compact_json(targets)}"
    return prompt


def build_format_repair_prompt(raw_reply: str, required_ids: list[int]) -> str:
    """只整理已有回复的格式，不附带原文，也不重新翻译。"""
    return (
        "只修正格式，不翻译、不润色、不补写。提取原回复中已有译文，仅返回JSON对象，"
        '结构为{"translations":{"ID":"原译文"}}。应保留的ID为:'
        f"{compact_json(required_ids)}。原回复:\n{raw_reply}"
    )


def completion_limit(
    items: list[tuple[int, dict[str, Any]]],
    positions: list[int],
) -> int:
    """按本批原文长度设置输出上限，防止异常生成消耗过多 token。"""
    source_characters = sum(len(sentence_text(items, position)) for position in positions)
    estimated = source_characters * 2 + 4000
    return min(MAX_COMPLETION_TOKENS, max(MIN_COMPLETION_TOKENS, estimated))


def content_to_text(content: Any) -> str:
    """兼容常见 OpenAI 风格内容块。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and str(part.get("type", "")).lower() in {
                "thinking", "redacted_thinking", "reasoning", "reasoning_content"
            }:
                continue
            parts.append(content_to_text(part))
        return "".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text"):
            if key in content:
                text = content_to_text(content[key])
                if text:
                    return text
    return ""


def api_error_text(response_data: dict[str, Any]) -> str | None:
    """提取 HTTP 200 响应中可能夹带的业务错误。"""
    error = response_data.get("error")
    if error not in (None, "", False):
        if isinstance(error, dict):
            message = error.get("message", error.get("detail"))
            code = error.get("code", error.get("type"))
            if message:
                return f"{code}: {message}" if code else str(message)
        return compact_json(error)[:500]
    status = str(response_data.get("status", "")).lower()
    success = response_data.get("success")
    code = response_data.get("code")
    bad_code = isinstance(code, int) and code not in {0, 200}
    if success is False or status in {"error", "failed", "failure"} or bad_code:
        return str(response_data.get("message", response_data.get("msg", "未知错误")))
    return None


def is_transient_message(message: str) -> bool:
    normalized = message.casefold()
    markers = (
        "overload", "service load", "service unavailable", "server is busy",
        "rate limit", "too many requests", "timeout", "temporarily",
        "服务负载", "服务繁忙", "请求过多", "请求超时", "限流", "稍后重试",
    )
    return any(marker in normalized for marker in markers)


TRUNCATION_REASONS = {"length", "max_tokens", "model_context_window_exceeded"}
REFUSAL_REASONS = {"refusal", "content_filter", "safety"}


def handle_stop_reason(reason: Any, stop_details: Any = None) -> None:
    """处理千帆 OpenAI 兼容接口可能返回的终止原因。"""
    normalized = str(reason or "").casefold()
    if normalized in TRUNCATION_REASONS:
        print("  模型输出达到 token 上限；已取得的 ID 会保留，随后只补缺失 ID。")
    if normalized in REFUSAL_REASONS:
        category = ""
        if isinstance(stop_details, dict) and stop_details.get("category"):
            category = f"，类别：{stop_details['category']}"
        raise RuntimeError(f"ERNIE 5.1 拒绝了本次请求{category}。")


def extract_response_text(response_data: dict[str, Any]) -> str:
    """从常见兼容响应结构中提取 message.content。"""
    error = api_error_text(response_data)
    if error:
        if is_transient_message(error):
            raise TransientAPIError(error)
        raise RuntimeError(f"千帆返回业务错误：{error}")
    if response_data.get("stop_reason") is not None:
        handle_stop_reason(response_data.get("stop_reason"), response_data.get("stop_details"))
    choices = response_data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            if response_data.get("stop_reason") is None:
                handle_stop_reason(
                    choice.get("finish_reason"), choice.get("stop_details")
                )
            message = choice.get("message")
            if isinstance(message, dict):
                text = content_to_text(message.get("content")).strip()
                if text:
                    return text
            text = content_to_text(choice.get("text")).strip()
            if text:
                return text
    for key in ("output_text", "content", "output", "result"):
        text = content_to_text(response_data.get(key)).strip()
        if text:
            return text
    for key in ("data", "response"):
        nested = response_data.get(key)
        if isinstance(nested, dict):
            return extract_response_text(nested)
    raise RuntimeError("千帆响应中没有可识别的模型正文。")


def read_streaming_response(response: Any) -> str:
    """读取 SSE 流，只拼接正文 content，忽略 reasoning_content。"""
    content_parts: list[str] = []
    fallback_lines: list[str] = []
    saw_sse = False
    finish_reason = ""
    stop_details: Any = None
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            fallback_lines.append(line)
            continue
        saw_sse = True
        event_text = line[5:].strip()
        if event_text == "[DONE]":
            break
        try:
            event = json.loads(event_text)
        except json.JSONDecodeError as error:
            raise ValueError("千帆流式响应中出现无效 JSON 数据块。") from error
        if not isinstance(event, dict):
            continue
        event_error = api_error_text(event)
        if event_error:
            if is_transient_message(event_error):
                raise TransientAPIError(event_error)
            raise RuntimeError(f"千帆流式响应错误：{event_error}")
        if event.get("stop_reason") is not None:
            finish_reason = str(event["stop_reason"]).casefold()
        if event.get("stop_details") is not None:
            stop_details = event["stop_details"]
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"]).casefold()
        if choice.get("stop_details") is not None:
            stop_details = choice["stop_details"]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content_parts.append(content_to_text(delta.get("content", delta.get("text"))))
        elif isinstance(choice.get("message"), dict):
            content_parts.append(content_to_text(choice["message"].get("content")))
        else:
            content_parts.append(content_to_text(choice.get("text")))
    if saw_sse:
        text = "".join(content_parts).strip()
        handle_stop_reason(finish_reason, stop_details)
        if not text:
            raise ValueError("千帆流式响应结束，但没有收到 content 正文。")
        return text
    raw_response = "\n".join(fallback_lines).strip()
    if not raw_response:
        raise ValueError("千帆返回了空响应。")
    response_data = json.loads(raw_response)
    if not isinstance(response_data, dict):
        raise ValueError("千帆返回的 JSON 顶层不是对象。")
    return extract_response_text(response_data)


def build_payload(
    prompt: str,
    max_tokens: int,
    json_mode: bool,
    thinking_control: bool = True,
) -> dict[str, Any]:
    """构造 ERNIE 5.1 请求；不发送 temperature 或 reasoning_effort。"""
    payload: dict[str, Any] = {
        "model": QIANFAN_MODEL,
        "messages": [
            {"role": "system", "content": ERNIE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # thinking=disabled 时，max_tokens 只限制输出正文。
        "max_tokens": max_tokens,
        "stream": USE_STREAMING,
    }
    if thinking_control and USE_THINKING_CONTROL:
        payload["thinking"] = {"type": THINKING_TYPE}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def send_once(payload: dict[str, Any]) -> str:
    """使用 Python 标准库发送一次千帆请求，无第三方 SDK 依赖。"""
    request = Request(
        QIANFAN_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {QIANFAN_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        if payload.get("stream"):
            return read_streaming_response(response)
        raw_response = response.read().decode("utf-8")
    response_data = json.loads(raw_response)
    if not isinstance(response_data, dict):
        raise ValueError("千帆返回的 JSON 顶层不是对象。")
    return extract_response_text(response_data)


def call_qianfan(prompt: str, max_tokens: int, use_json_mode: bool = True) -> str:
    """调用千帆；仅对网络、限流和临时服务错误进行重试。"""
    json_mode = USE_JSON_MODE and use_json_mode
    json_mode_fallback_used = False
    thinking_control = USE_THINKING_CONTROL
    thinking_fallback_used = False
    last_error: Exception | None = None
    attempt = 1
    while attempt <= MAX_RETRIES:
        try:
            return send_once(
                build_payload(prompt, max_tokens, json_mode, thinking_control)
            )
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            message = f"HTTP {error.code}: {detail[:800]}"
            normalized = message.casefold()
            generic_parameter_error = (
                "unknown parameter" in normalized
                or "unrecognized request argument" in normalized
                or "extra inputs are not permitted" in normalized
                or "未知参数" in normalized
            )
            thinking_parameter_error = (
                '"thinking"' in normalized
                or "'thinking'" in normalized
                or " thinking" in normalized
            )
            if (
                error.code == 400 and thinking_control and not thinking_fallback_used
                and thinking_parameter_error
            ):
                thinking_control = False
                thinking_fallback_used = True
                print("  千帆接口未接受 thinking 参数，本次自动移除后继续。")
                continue
            if (
                error.code == 400 and json_mode and not json_mode_fallback_used
                and (
                    "response_format" in normalized
                    or "json_object" in normalized
                    or generic_parameter_error
                )
            ):
                json_mode = False
                json_mode_fallback_used = True
                print("  千帆接口未接受 JSON Mode，本次自动改用普通文本输出。")
                continue
            if (
                error.code == 400 and thinking_control and not thinking_fallback_used
                and generic_parameter_error
            ):
                thinking_control = False
                thinking_fallback_used = True
                print("  千帆接口未接受 thinking 参数，本次自动移除后继续。")
                continue
            if error.code in {408, 409, 425, 429} or error.code >= 500:
                last_error = TransientAPIError(message)
            elif is_transient_message(message):
                last_error = TransientAPIError(message)
            else:
                raise RuntimeError(f"千帆请求失败，且不会重试：{message}") from error
        except (URLError, TimeoutError, TransientAPIError, ValueError) as error:
            last_error = error
        if attempt >= MAX_RETRIES:
            break
        wait_seconds = min(2 ** (attempt - 1), 30) + random.uniform(0, 0.8)
        print(
            f"  临时请求错误（第 {attempt}/{MAX_RETRIES} 次）：{last_error}；"
            f"{wait_seconds:.1f} 秒后重试。"
        )
        time.sleep(wait_seconds)
        attempt += 1
    raise RuntimeError(f"连续 {MAX_RETRIES} 次请求失败：{last_error}")


def remove_markdown_fence(text: str) -> str:
    """移除模型偶尔附加的 Markdown 代码块。"""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_translation(value: Any) -> str:
    """把常见单项结构规范化为译文字符串。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("translation", value.get("text"))
        return nested.strip() if isinstance(nested, str) else ""
    return ""


def align_returned_ids(
    mapping: dict[int, str], required_ids: list[int]
) -> dict[int, str]:
    """优先使用原 ID；若模型把补译项重编号为 0..N-1，则安全映射回来。"""
    required_set = set(required_ids)
    direct = {
        item_id: value for item_id, value in mapping.items()
        if item_id in required_set and value
    }
    if direct:
        return direct
    # 仅在完全没有原 ID 命中，且所有键都属于局部编号范围时才转换，避免
    # 把真正的无关 ID 猜成待译 ID。允许局部回复不完整，例如只返回 {"0": ...}。
    if mapping and all(0 <= item_id < len(required_ids) for item_id in mapping):
        return {
            required_ids[local_id]: value
            for local_id, value in mapping.items()
            if value
        }
    return {}


def candidate_mapping(candidate: Any, required_ids: list[int]) -> dict[int, str]:
    """从 JSON 候选值中提取指定 ID，并兼容补译时从 0 重新编号。"""
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
    if isinstance(candidate, dict) and "translations" in candidate:
        candidate = candidate["translations"]
    mapping: dict[int, str] = {}
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            try:
                item_id = int(str(key).strip())
            except ValueError:
                continue
            translation = normalize_translation(value)
            if translation:
                mapping[item_id] = translation
        return align_returned_ids(mapping, required_ids)
    elif isinstance(candidate, list) and 0 < len(candidate) <= len(required_ids):
        for item_id, value in zip(required_ids, candidate):
            translation = normalize_translation(value)
            if translation:
                mapping[item_id] = translation
    return mapping


def decode_lenient_json_string(value: str) -> str:
    """解码常见 JSON 转义，同时保留模型未转义的内部双引号。"""
    escape_map = {
        '"': '"', "'": "'", "\\": "\\", "/": "/",
        "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t",
    }
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\" or index + 1 >= len(value):
            result.append(character)
            index += 1
            continue
        escaped = value[index + 1]
        if escaped == "u" and index + 5 < len(value):
            hexadecimal = value[index + 2 : index + 6]
            if re.fullmatch(r"[0-9a-fA-F]{4}", hexadecimal):
                result.append(chr(int(hexadecimal, 16)))
                index += 6
                continue
        if escaped in escape_map:
            result.append(escape_map[escaped])
        else:
            result.extend(("\\", escaped))
        index += 2
    return "".join(result)


def parse_lenient_translation_mapping(raw_text: str) -> dict[int, str]:
    """容错提取因内部英文双引号未转义而损坏的 translations 对象。"""
    translations_match = re.search(
        r"[\"']translations[\"']\s*:\s*\{", raw_text, flags=re.IGNORECASE
    )
    if translations_match is None:
        return {}
    index = translations_match.end()
    entries: dict[int, str] = {}
    key_pattern = re.compile(r"([\"'])(\d+)\1\s*:")
    value_end_pattern = re.compile(
        r"\s*(?:,\s*(?:[\"']\s*)?\d+\s*(?:[\"'])?\s*:|\})"
    )
    while index < len(raw_text):
        while index < len(raw_text) and raw_text[index] in " \t\r\n,":
            index += 1
        if index >= len(raw_text) or raw_text[index] == "}":
            break
        key_match = key_pattern.match(raw_text, index)
        if key_match is None:
            break
        item_id = int(key_match.group(2))
        index = key_match.end()
        while index < len(raw_text) and raw_text[index].isspace():
            index += 1
        if index >= len(raw_text) or raw_text[index] not in {"\"", "'"}:
            break
        quote = raw_text[index]
        value_start = index + 1
        scan_index = value_start
        value_end: int | None = None
        while scan_index < len(raw_text):
            character = raw_text[scan_index]
            if character == "\\":
                scan_index += 2
                continue
            if character == quote and value_end_pattern.match(raw_text[scan_index + 1 :]):
                value_end = scan_index
                break
            scan_index += 1
        if value_end is None:
            break
        translation = decode_lenient_json_string(raw_text[value_start:value_end]).strip()
        if translation:
            entries[item_id] = translation
        delimiter = value_end_pattern.match(raw_text[value_end + 1 :])
        if delimiter is None:
            break
        index = value_end + 1
        if delimiter.group(0).lstrip().startswith("}"):
            break
    return entries


NUMBERED_PATTERN = re.compile(
    r"(?m)^\s*(?:[-+*]\s*)?(?:(?:ID|译文)\s*)?(?:第\s*)?"
    r"[\[\(（【{\"'“]?\s*(\d+)\s*[\]\)）】}\"'”]?\s*"
    r"(?:项|段|句|条)?\s*(?:[:：=、.．]|->|→)\s*"
)


def parse_numbered_mapping(raw_text: str) -> dict[int, str]:
    """兼容模型偶尔返回的逐行“ID: 译文”格式。"""
    matches = list(NUMBERED_PATTERN.finditer(raw_text))
    entries: dict[int, str] = {}
    for match_index, match in enumerate(matches):
        value_end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(raw_text)
        value = raw_text[match.end() : value_end].strip().rstrip(",，").strip()
        if len(value) >= 2 and value[0] in {'"', "'", "“", "‘"}:
            matching_quote = {'"': '"', "'": "'", "“": "”", "‘": "’"}[value[0]]
            if value.endswith(matching_quote):
                value = value[1:-1].strip()
        if value:
            entries[int(match.group(1))] = value
    return entries


def parse_translations(raw_text: str, required_ids: list[int]) -> dict[int, str]:
    """优先严格解析，再使用本地容错解析；返回已成功提取的指定 ID。"""
    cleaned = remove_markdown_fence(raw_text)
    if not cleaned:
        raise ContentParseError("模型回复为空。")
    best: dict[int, str] = {}
    try:
        complete = json.loads(cleaned)
    except json.JSONDecodeError:
        complete = None
    if complete is not None:
        best = candidate_mapping(complete, required_ids)
    decoder = json.JSONDecoder()
    for position, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned, position)
        except json.JSONDecodeError:
            continue
        mapping = candidate_mapping(candidate, required_ids)
        if len(mapping) > len(best):
            best = mapping
    lenient = align_returned_ids(
        parse_lenient_translation_mapping(cleaned), required_ids
    )
    if len(lenient) > len(best):
        best = lenient
    numbered = align_returned_ids(parse_numbered_mapping(cleaned), required_ids)
    if len(numbered) > len(best):
        best = numbered
    if not best:
        preview = " ".join(cleaned.split())[:260]
        raise ContentParseError(f"无法安全提取任何译文。回复开头：{preview!r}")
    return best


def fetch_translations_once(
    items: list[tuple[int, dict[str, Any]]], positions: list[int]
) -> dict[int, str]:
    """翻译指定 ID；格式异常时最多额外整理一次，不重复翻译。"""
    prompt = build_translation_prompt(items, positions)
    raw_reply = call_qianfan(prompt, completion_limit(items, positions))
    try:
        return parse_translations(raw_reply, positions)
    except ContentParseError as original_error:
        if (
            not ALLOW_ONE_FORMAT_REPAIR or not raw_reply.strip()
            or len(raw_reply) > MAX_FORMAT_REPAIR_INPUT_CHARS
        ):
            raise original_error
        print("  本地未能拆分回复；只整理一次已有译文格式，不重新翻译。")
        repair_prompt = build_format_repair_prompt(raw_reply, positions)
        repaired_reply = call_qianfan(
            repair_prompt,
            min(MAX_COMPLETION_TOKENS, max(MIN_COMPLETION_TOKENS, len(raw_reply))),
        )
        return parse_translations(repaired_reply, positions)


def translate_batch(
    items: list[tuple[int, dict[str, Any]]], positions: list[int]
) -> dict[int, str]:
    """翻译一批；漏项时多轮只补缺失 ID，绝不重译已取得的内容。"""
    translations = fetch_translations_once(items, positions)
    previous_round_progress = True
    for round_number in range(1, MAX_MISSING_REPAIR_ROUNDS + 1):
        missing = [position for position in positions if position not in translations]
        if not missing:
            return translations
        missing_preview = "、".join(str(position) for position in missing[:20])
        if len(missing) > 20:
            missing_preview += "……"
        if round_number == 1:
            print(f"  首轮缺少 ID {missing_preview}，现在只补译缺失句段。")
        else:
            print(
                f"  第 {round_number} 轮继续只补缺失 ID {missing_preview}，"
                "已取得的译文不会重译。"
            )
        repair_batch_segments = (
            MISSING_REPAIR_BATCH_SEGMENTS if previous_round_progress else 1
        )
        before_count = len(translations)
        missing_batches = split_batches(
            items,
            missing,
            max_segments=repair_batch_segments,
            max_source_characters=MISSING_REPAIR_BATCH_SOURCE_CHARS,
        )
        for batch_index, missing_batch in enumerate(missing_batches):
            try:
                repaired = fetch_translations_once(items, missing_batch)
            except ContentParseError as error:
                preview = "、".join(str(position) for position in missing_batch[:10])
                print(f"  补译 ID {preview} 的回复仍无法解析：{error}")
                repaired = {}
            translations.update(repaired)
            if REQUEST_INTERVAL_SECONDS > 0 and batch_index + 1 < len(missing_batches):
                time.sleep(REQUEST_INTERVAL_SECONDS)
        previous_round_progress = len(translations) > before_count
    still_missing = [position for position in positions if position not in translations]
    if still_missing:
        missing_text = "、".join(str(position) for position in still_missing)
        raise RuntimeError(f"补译后仍缺少 ID：{missing_text}")
    return translations


def pending_positions(
    target_rows: list[dict[str, Any]], items: list[tuple[int, dict[str, Any]]]
) -> list[int]:
    """返回当前作品中尚需生成的非空句段位置。"""
    result: list[int] = []
    for position, (row_index, _) in enumerate(items):
        if not sentence_text(items, position):
            model_map = get_model_map(target_rows[row_index])
            if OVERWRITE_EXISTING or not has_result(target_rows, row_index):
                model_map[OUTPUT_MODEL_KEY] = EMPTY_SENTENCE_RESULT
            continue
        if OVERWRITE_EXISTING or not has_result(target_rows, row_index):
            result.append(position)
    return result


def main() -> None:
    check_config()
    source_rows = load_json_array(SOURCE_PATH)
    target_rows = load_or_create_target(source_rows)
    work_groups = build_work_groups(source_rows)
    stop_index = len(work_groups) if END_WORK_INDEX is None else min(END_WORK_INDEX, len(work_groups))
    if START_WORK_INDEX >= stop_index:
        print("没有需要处理的作品，请检查 START_WORK_INDEX 和 END_WORK_INDEX。")
        return
    selected = work_groups[START_WORK_INDEX:stop_index]
    planned_batches = 0
    pending_segment_total = 0
    for _, items in selected:
        positions = pending_positions(target_rows, items)
        pending_segment_total += len(positions)
        planned_batches += len(split_batches(items, positions))
    atomic_save(TARGET_PATH, target_rows)
    print(f"源文件：{SOURCE_PATH.name}")
    print(f"目标文件：{TARGET_PATH.name}")
    print(
        f"模型：{QIANFAN_MODEL}；thinking={THINKING_TYPE}；"
        f"JSON Mode={'开启' if USE_JSON_MODE else '关闭'}"
    )
    print(
        f"待处理 {len(selected)} 篇中的 {pending_segment_total} 个句段，"
        f"正常预计 {planned_batches} 次翻译请求。"
    )
    print(
        f"分批上限：每批 {MAX_BATCH_SEGMENTS} 段或 {MAX_BATCH_SOURCE_CHARS} 个原文字符；"
        f"只携带前后各 {CONTEXT_SEGMENTS} 段邻近语境。"
    )
    completed_works = 0
    completed_segments = 0
    skipped_works = 0
    try:
        for relative_index, (_, items) in enumerate(selected):
            work_index = START_WORK_INDEX + relative_index
            title = str(items[0][1].get("poem_title", ""))
            positions = pending_positions(target_rows, items)
            if not positions:
                skipped_works += 1
                print(f"[{work_index + 1}/{stop_index}] 《{title}》已有完整结果，跳过。")
                continue
            batches = split_batches(items, positions)
            source_characters = sum(len(sentence_text(items, p)) for p in positions)
            print(
                f"[{work_index + 1}/{stop_index}] 正在翻译《{title}》："
                f"{len(positions)} 个待译句段、{source_characters} 个原文字符，"
                f"分为 {len(batches)} 批。"
            )
            for batch_number, batch in enumerate(batches, start=1):
                first_id, last_id = batch[0], batch[-1]
                batch_characters = sum(len(sentence_text(items, p)) for p in batch)
                print(
                    f"  [批次 {batch_number}/{len(batches)}] ID {first_id} 至 {last_id}，"
                    f"{len(batch)} 段、{batch_characters} 字。"
                )
                translations = translate_batch(items, batch)
                for position, translation in translations.items():
                    row_index = items[position][0]
                    get_model_map(target_rows[row_index])[OUTPUT_MODEL_KEY] = translation
                # 每批立即保存；异常退出后可直接按缺失句段续跑。
                atomic_save(TARGET_PATH, target_rows)
                completed_segments += len(batch)
                print("    本批已保存。")
                if REQUEST_INTERVAL_SECONDS > 0 and batch_number < len(batches):
                    time.sleep(REQUEST_INTERVAL_SECONDS)
            completed_works += 1
            if REQUEST_INTERVAL_SECONDS > 0:
                time.sleep(REQUEST_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        atomic_save(TARGET_PATH, target_rows)
        print("\n检测到手动停止，已保存所有完成批次；下次可直接续跑。")
        return
    except Exception:
        atomic_save(TARGET_PATH, target_rows)
        print("发生错误，已保存此前完成的批次。")
        raise
    atomic_save(TARGET_PATH, target_rows)
    print(
        f"全部完成！本次完成 {completed_works} 篇、{completed_segments} 个句段；"
        f"跳过已有完整结果 {skipped_works} 篇。"
    )


if __name__ == "__main__":
    main()
