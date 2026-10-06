"""按（原句、模型名）匹配传统评分和人工评分，计算样本级相关系数。

依赖：pip install scipy
运行：python calculate_correlations.py
也可指定：python calculate_correlations.py --human 人工.json --auto 传统.json
人工综合分 = (5 * 信 + 3 * 达 + 2 * 雅) / 10。
将所有匹配的“原句×模型”作为观测，非先按模型求均值。
"""

import argparse
import json
import math
from pathlib import Path

from scipy.stats import kendalltau, pearsonr, spearmanr


def read_index(path, score_field):
    with Path(path).open(encoding="utf-8-sig") as file:
        data = json.load(file)
    index = {}
    for item in data:
        sentence = item["sentence"]
        for model, scores in item[score_field].items():
            key = (sentence, model)
            if key in index:
                raise ValueError(f"重复样本，无法唯一匹配：{key}")
            index[key] = (scores, item.get("模型翻译", {}).get(model))
    return index


def number(value):
    """兼容字符串分数；缺失、非法值和无穷值均不参与计算。"""
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def calculate(human, automatic):
    common = sorted(human.keys() & automatic.keys())
    if not common:
        raise ValueError("两个文件没有能按原句和模型名匹配的样本。")
    metrics = sorted({metric for scores, _ in automatic.values() for metric in scores})
    pairs = {metric: ([], []) for metric in metrics}
    invalid_human = 0
    for key in common:
        human_scores, human_text = human[key]
        auto_scores, auto_text = automatic[key]
        if human_text is not None and auto_text is not None and human_text != auto_text:
            raise ValueError(f"同一样本的模型译文不同，停止计算：{key}")
        values = [number(human_scores.get(dim)) for dim in ("信", "达", "雅")]
        if any(value is None for value in values):
            invalid_human += 1
            continue
        weighted = sum(v * w for v, w in zip(values, (1, 1, 1))) / 3
        for metric in metrics:
            score = number(auto_scores.get(metric))
            if score is not None:
                pairs[metric][0].append(score)
                pairs[metric][1].append(weighted)

    results = []
    for metric, (x, y) in pairs.items():
        row = {"metric": metric, "n": len(x), "excluded_matched": len(common) - len(x)}
        reason = ""
        if len(x) < 2:
            reason = "有效配对样本少于2个"
        elif len(set(x)) < 2 or len(set(y)) < 2:
            reason = "至少一组分数为常数，相关系数未定义"
        for name, function in (("Pearson", pearsonr), ("Spearman", spearmanr), ("Kendall", kendalltau)):
            if reason:
                row[name] = {"coefficient": None, "p_value": None}
            else:
                # tau-b 对并列排名进行校正，适合存在大量同分的人工评分。
                result = function(x, y, variant="b") if name == "Kendall" else function(x, y)
                row[name] = {"coefficient": number(result.statistic), "p_value": number(result.pvalue)}
        row["note"] = reason
        results.append(row)
    return {
        "human_formula": "(5*信 + 3*达 + 2*雅)/10",
        "unit": "原句×模型，所有匹配观测合并计算",
        "kendall_variant": "tau-b",
        "matched": len(common),
        "human_only": len(human.keys() - automatic.keys()),
        "automatic_only": len(automatic.keys() - human.keys()),
        "invalid_human": invalid_human,
        "results": results,
    }


def significance_mark(p_value):
    """返回显著性星号标记。"""
    if p_value is None:
        return ""
    if p_value < 0.001:
        return " ***"
    if p_value < 0.01:
        return " **"
    if p_value < 0.05:
        return " *"
    return ""


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=base / "deepseek_matched_60_ordered.json")
    parser.add_argument("--auto", default=base / "sample60_scores.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("dw_person_correlation_results.json"))
    args = parser.parse_args()
    report = calculate(read_index(args.human, "评分"), read_index(args.auto, "传统评分"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    print(f"Matched: {report['matched']}; human-only: {report['human_only']}; "
          f"auto-only: {report['automatic_only']}; invalid_human: {report['invalid_human']}")

    # 逐个指标打印，系数与 p 值成对显示
    for row in report["results"]:
        note = f"  (note: {row['note']})" if row["note"] else ""
        print(f"\n[{row['metric']}]  N={row['n']}  excluded={row['excluded_matched']}{note}")
        for name in ("Pearson", "Spearman", "Kendall"):
            coef = row[name]["coefficient"]
            pval = row[name]["p_value"]
            coef_s = f"{coef:>10.6f}" if coef is not None else f"{'NA':>10}"
            pval_s = f"{pval:>10.4g}" if pval is not None else f"{'NA':>10}"
            print(f"  {name:<9} r = {coef_s}   p = {pval_s}{significance_mark(pval)}")

    print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()