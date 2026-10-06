#!/usr/bin/env python3
"""在 VS Code 中直接运行：通过 Anthropic SDK 调用 MiniMax M3 翻译古诗文。"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    import anthropic
except ImportError:  # 允许先打开脚本查看配置，运行时再给出安装提示。
    anthropic = None  # type: ignore[assignment]


# ======================== 请在这里填写配置 ========================

# 推荐把 Key 放进 MINIMAX_API_KEY 环境变量；也可以直接粘贴到最后的引号内。
MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY', "")

# MiniMax Anthropic 兼容接口中的正式模型 ID（注意大小写）。
MINIMAX_MODEL = "MiniMax-M3"

# 写入“模型翻译”字段时使用的键名。
OUTPUT_MODEL_KEY = "minimax-m3"

# 把脚本与源文件放在同一文件夹内。目标文件不存在时会自动创建。
SOURCE_FILE_NAME = "Final_gsw_new.json"
TARGET_FILE_NAME = "Final_gsw_chatgpt-5.6-sol.json"

# 按作品控制运行范围。END_WORK_INDEX=None 表示处理到最后一篇。
START_WORK_INDEX = 0
END_WORK_INDEX = None

# MiniMax-M3 默认可关闭 thinking。翻译任务结构明确，关闭可显著节省输出 token。
# 如需让模型加强推理，可改为 "adaptive"。
THINKING_MODE = "disabled"  # 可选："disabled"、"adaptive"

# MiniMax 官方建议 temperature=1.0；standard 比 priority 更省费用。
TEMPERATURE = 1.0
SERVICE_TIER = "standard"  # 可选："standard"、"priority"

# 某些 Anthropic SDK 版本的 Messages.create 不接收 temperature、thinking、
# service_tier。False 时省略这三项：MiniMax-M3 默认关闭 thinking，默认使用
# standard 层级，因此不会增加 token 或费用。升级 SDK 后如需显式控制可改为 True。
SEND_OPTIONAL_PARAMETERS = False

# 避开部分 Anaconda 环境中 urllib3 与旧 Brotli 的 output_buffer_limit 冲突。
# gzip/deflate 仍可压缩传输，但不会调用有版本冲突的 Brotli 解压器。
RESPONSE_ACCEPT_ENCODING = "gzip, deflate"

# 每批同时受句段数和原文字符数限制；任一达到上限就自动切下一批。
# 根据当前数据分布，100 段的《鸿门宴》一次请求，189 段的《陈涉世家》自动切分。
MAX_BATCH_SEGMENTS = 150
MAX_BATCH_SOURCE_CHARS = 12000

# 长篇分批时，每个待译句段最多携带前后各几个邻近句段作为语境。
# 不再为每一批重复发送整篇全文，可显著减少输入 token。
CONTEXT_SEGMENTS = 2

# 单次允许生成的最大 token。它只是生成上限，不会预先消耗这些 token。
MAX_COMPLETION_TOKENS = 32000
MIN_COMPLETION_TOKENS = 5000

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

# False：已有 minimax-m3 结果时跳过；True：覆盖它，保留其他模型结果。
OVERWRITE_EXISTING = False

# MiniMax 官方 Anthropic API 兼容地址，一般无需修改。
MINIMAX_BASE_URL = "https://api.minimaxi.com/anthropic"

# ======================== 配置结束 ========================


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / SOURCE_FILE_NAME
TARGET_PATH = SCRIPT_DIR / TARGET_FILE_NAME
EMPTY_SENTENCE_RESULT = "（原句为空，无可译内容）"
SYSTEM_PROMPT = (
    "你是严谨的古诗文翻译专家。只完成用户指定的翻译或格式整理任务，"
    "严格保持输入 ID，不输出解释、Markdown 或任务外内容。"
)


class TransientAPIError(ValueError):
    """适合等待后重试的临时 API 错误。"""


class ContentParseError(ValueError):
    """模型正文中没有可安全提取的译文。"""


def check_config() -> None:
    """检查运行配置，避免请求发出后才发现参数错误。"""
    if anthropic is None:
        raise RuntimeError("未安装 Anthropic SDK，请先运行：pip install -U anthropic")
    if not MINIMAX_API_KEY.strip() or "请在这里填写" in MINIMAX_API_KEY:
        raise RuntimeError(
            "请设置 MINIMAX_API_KEY 环境变量，或在脚本顶部填写 MiniMax API Key。"
        )
    if not MINIMAX_MODEL.strip():
        raise RuntimeError("MINIMAX_MODEL 不能为空。")
    if not OUTPUT_MODEL_KEY.strip():
        raise RuntimeError("OUTPUT_MODEL_KEY 不能为空。")
    if THINKING_MODE not in {"disabled", "adaptive"}:
        raise RuntimeError('THINKING_MODE 只能是 "disabled" 或 "adaptive"。')
    if not 0 <= TEMPERATURE <= 2:
        raise RuntimeError("TEMPERATURE 必须在 0 到 2 之间。")
    if SERVICE_TIER not in {"standard", "priority"}:
        raise RuntimeError('SERVICE_TIER 只能是 "standard" 或 "priority"。')
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
    estimated = source_characters * 2 + 6000
    return min(MAX_COMPLETION_TOKENS, max(MIN_COMPLETION_TOKENS, estimated))


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
    """处理 Anthropic Messages 接口可能返回的终止原因。"""
    normalized = str(reason or "").casefold()
    if normalized in TRUNCATION_REASONS:
        print("  模型输出达到 token 上限；已取得的 ID 会保留，随后只补缺失 ID。")
    if normalized in REFUSAL_REASONS:
        category = ""
        if isinstance(stop_details, dict) and stop_details.get("category"):
            category = f"，类别：{stop_details['category']}"
        raise RuntimeError(f"MiniMax M3 拒绝了本次请求{category}。")


def object_field(value: Any, name: str, default: Any = None) -> Any:
    """同时读取 SDK 对象属性和测试中使用的字典字段。"""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def build_request(
    prompt: str,
    max_tokens: int,
    stream: bool,
    include_optional_parameters: bool | None = None,
) -> dict[str, Any]:
    """构造 MiniMax Anthropic Messages API 请求。"""
    request: dict[str, Any] = {
        "model": MINIMAX_MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        "extra_headers": {"Accept-Encoding": RESPONSE_ACCEPT_ENCODING},
        "stream": stream,
    }
    if include_optional_parameters is None:
        include_optional_parameters = SEND_OPTIONAL_PARAMETERS
    if include_optional_parameters:
        request.update(
            {
                "temperature": TEMPERATURE,
                "thinking": {"type": THINKING_MODE},
                "service_tier": SERVICE_TIER,
            }
        )
    return request


_MINIMAX_CLIENT: Any = None
_SDK_ACCEPTS_OPTIONAL_PARAMETERS = True


def get_minimax_client() -> Any:
    """延迟创建客户端，避免导入脚本时就校验网络和 Key。"""
    global _MINIMAX_CLIENT
    if _MINIMAX_CLIENT is None:
        if anthropic is None:
            raise RuntimeError("未安装 Anthropic SDK，请先运行：pip install -U anthropic")
        _MINIMAX_CLIENT = anthropic.Anthropic(
            api_key=MINIMAX_API_KEY,
            base_url=MINIMAX_BASE_URL,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    return _MINIMAX_CLIENT


def extract_message_text(message: Any) -> str:
    """从非流式 Anthropic Message 中只提取 text 块。"""
    handle_stop_reason(
        object_field(message, "stop_reason"), object_field(message, "stop_details")
    )
    text_parts: list[str] = []
    for block in object_field(message, "content", []) or []:
        if str(object_field(block, "type", "")).casefold() == "text":
            text = object_field(block, "text", "")
            if isinstance(text, str) and text:
                text_parts.append(text)
    text = "".join(text_parts).strip()
    if not text:
        raise TransientAPIError("MiniMax 响应中没有收到 text 正文。")
    return text


def extract_stream_text(stream: Any) -> str:
    """读取 Anthropic SDK 流，只保留 text_delta，忽略 thinking_delta。"""
    text_parts: list[str] = []
    stop_reason = ""
    stop_details: Any = None
    for chunk in stream:
        chunk_type = str(object_field(chunk, "type", "")).casefold()
        if chunk_type == "content_block_start":
            block = object_field(chunk, "content_block")
            if str(object_field(block, "type", "")).casefold() == "text":
                text = object_field(block, "text", "")
                if isinstance(text, str) and text:
                    text_parts.append(text)
        elif chunk_type == "content_block_delta":
            delta = object_field(chunk, "delta")
            if str(object_field(delta, "type", "")).casefold() == "text_delta":
                text = object_field(delta, "text", "")
                if isinstance(text, str) and text:
                    text_parts.append(text)
        elif chunk_type == "message_delta":
            delta = object_field(chunk, "delta")
            reason = object_field(delta, "stop_reason")
            if reason is not None:
                stop_reason = str(reason)
            details = object_field(delta, "stop_details")
            if details is not None:
                stop_details = details
        elif chunk_type == "message_stop":
            message = object_field(chunk, "message")
            reason = object_field(message, "stop_reason")
            if reason is not None:
                stop_reason = str(reason)
            details = object_field(message, "stop_details")
            if details is not None:
                stop_details = details
    handle_stop_reason(stop_reason, stop_details)
    text = "".join(text_parts).strip()
    if not text:
        raise TransientAPIError("MiniMax 流式响应结束，但没有收到 text 正文。")
    return text


def send_once(prompt: str, max_tokens: int) -> str:
    """通过 Anthropic SDK 发送一次 MiniMax M3 请求。"""
    global _SDK_ACCEPTS_OPTIONAL_PARAMETERS
    client = get_minimax_client()
    include_optional = SEND_OPTIONAL_PARAMETERS and _SDK_ACCEPTS_OPTIONAL_PARAMETERS
    request = build_request(
        prompt, max_tokens, USE_STREAMING, include_optional_parameters=include_optional
    )
    optional_fallback_used = False
    header_fallback_used = False
    while True:
        try:
            response = client.messages.create(**request)
            break
        except TypeError as error:
            message = str(error)
            if "unexpected keyword argument" not in message:
                raise
            optional_names = ("temperature", "thinking", "service_tier")
            optional_argument_error = any(
                f"'{name}'" in message for name in optional_names
            )
            if include_optional and not optional_fallback_used and optional_argument_error:
                _SDK_ACCEPTS_OPTIONAL_PARAMETERS = False
                optional_fallback_used = True
                include_optional = False
                print(
                    "  当前 Anthropic SDK 不接受可选生成参数；已自动省略，"
                    "MiniMax-M3 继续使用默认 thinking=disabled 和 standard 层级。"
                )
                request = build_request(
                    prompt,
                    max_tokens,
                    USE_STREAMING,
                    include_optional_parameters=False,
                )
                continue
            if "'extra_headers'" in message and not header_fallback_used:
                header_fallback_used = True
                request.pop("extra_headers", None)
                print("  当前 Anthropic SDK 不接受自定义压缩请求头，已自动移除。")
                continue
            raise
    if USE_STREAMING:
        return extract_stream_text(response)
    return extract_message_text(response)


def exception_status_code(error: Exception) -> int | None:
    """提取 Anthropic SDK APIStatusError 等异常携带的 HTTP 状态码。"""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def is_transient_exception(error: Exception) -> bool:
    """仅将网络、超时、限流及服务端错误视为可重试。"""
    status_code = exception_status_code(error)
    if status_code in {408, 409, 425, 429} or (
        status_code is not None and status_code >= 500
    ):
        return True
    transient_names = {
        "apiconnectionerror", "apitimeouterror", "ratelimiterror",
        "internalservererror", "transientapierror",
    }
    return error.__class__.__name__.casefold() in transient_names or is_transient_message(
        str(error)
    )


def dependency_error_message(error: Exception) -> str | None:
    """把常见 Anaconda Brotli/urllib3 版本冲突转换为可执行的修复提示。"""
    message = str(error)
    if "output_buffer_limit" not in message or "decompress" not in message.casefold():
        return None
    return (
        "检测到 Brotli 解压依赖版本冲突。请在 PowerShell 中运行：\n"
        'D:\\anaconda\\python.exe -m pip install -U "Brotli>=1.2.0" '
        '"urllib3>=2.6.1"\n'
        "安装完成后关闭当前终端，重新打开 PowerShell 再运行脚本。"
    )


def call_minimax(prompt: str, max_tokens: int) -> str:
    """调用 MiniMax；临时错误指数退避，参数和鉴权错误立即终止。"""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return send_once(prompt, max_tokens)
        except Exception as error:
            if not is_transient_exception(error):
                dependency_hint = dependency_error_message(error)
                if dependency_hint:
                    raise RuntimeError(dependency_hint) from error
                status_code = exception_status_code(error)
                status_text = f"HTTP {status_code}，" if status_code is not None else ""
                raise RuntimeError(
                    f"MiniMax 请求失败，且不会重试：{status_text}{error}"
                ) from error
            last_error = error
        if attempt >= MAX_RETRIES:
            break
        wait_seconds = min(2 ** (attempt - 1), 30) + attempt * 0.2
        print(
            f"  临时请求错误（第 {attempt}/{MAX_RETRIES} 次）：{last_error}；"
            f"{wait_seconds:.1f} 秒后重试。"
        )
        time.sleep(wait_seconds)
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
    raw_reply = call_minimax(prompt, completion_limit(items, positions))
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
        repaired_reply = call_minimax(
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
    if SEND_OPTIONAL_PARAMETERS:
        parameter_status = (
            f"thinking={THINKING_MODE}；service_tier={SERVICE_TIER}；"
            f"temperature={TEMPERATURE}"
        )
    else:
        parameter_status = "兼容模式；thinking=disabled；service_tier=standard"
    print(f"模型：{MINIMAX_MODEL}；{parameter_status}")
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
