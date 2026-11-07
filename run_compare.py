#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P1-4 对照实验脚本
==================
用途：在**同一批病例**上分别跑基线版与改进版，产出可写入项目说明的对比数据。
赛题原文要求：「基线版与改进版在同一批公开练习病例上的结果对比」。

用法：
    # 离线对比（不消耗额度，先验证流程）
    python run_compare.py --dry-run

    # 真实对比（会调用模型，消耗额度）
    python run_compare.py

    # 指定输入文件（比如用 15 例测试集）
    python run_compare.py --input test_cases.json

输出（全部在 test-output/ 下，按时间戳建子目录，不覆盖历史）：
    test-output/compare_2026-11-07-17-00-00/
        ├── baseline_output.json
        ├── baseline_results.json
        ├── baseline_run_log.json
        ├── improved_output.json
        ├── improved_results.json
        ├── improved_run_log.json
        ├── baseline_2026-11-07-17-00-05.md    ← 基线版可读报告
        ├── improved_2026-11-07-17-00-30.md    ← 改进版可读报告
        └── compare_report.json                ← 对比结论
"""

import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 统一用 UTF-8 输出，避免 Windows 控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
# 脚本所在目录
MAIN = BASE / "main_p1.py"
# 主程序路径
OUT_ROOT = BASE / "test-output"
# 所有产物的根目录


def make_run_dir() -> Path:
    # 按时间戳建本次实验目录
    """创建本次实验目录。"""
    # 函数说明
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 时间戳
    d = OUT_ROOT / f"compare_{stamp}"
    # 实验目录
    d.mkdir(parents=True, exist_ok=True)
    # 不存在就创建
    return d
    # 返回


def run_mode(mode: str, dry: bool, run_dir: Path, input_file: str) -> int:
    # 跑指定模式。产物通过 --out-dir 落到本次实验目录，不污染根目录
    """运行一种模式。"""
    # 函数说明
    cmd = [
        sys.executable, str(MAIN),
        "--mode", mode,
        "--out-prefix", f"{mode}_",
        # 前缀区分两个模式的文件名
        "--out-dir", str(run_dir.relative_to(BASE)).replace("\\", "/"),
        # 让 JSON 三件套写进本次实验目录
    ]
    # 基础命令
    if input_file:
        # 指定了输入文件
        cmd += ["--input", input_file]
        # 加上输入参数，保证两模式跑的是同一份数据
    if dry:
        # 离线模式
        cmd.append("--dry-run")
        # 加参数
    print(f"[运行] {mode} 模式", file=sys.stderr)
    # 打印进度
    r = subprocess.run(cmd, cwd=str(BASE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # 执行子进程
    tail = (r.stderr or "").strip().splitlines()
    # 取日志行
    for line in tail[-3:]:
        # 只显示最后三行
        print("   " + line, file=sys.stderr)
        # 缩进显示
    return r.returncode
    # 返回退出码


def load_json(p: Path) -> dict:
    # 读取 JSON，不存在返回空字典
    """读取 JSON。"""
    # 函数说明
    if not p.exists():
        # 不存在
        return {}
        # 空字典
    return json.loads(p.read_text(encoding="utf-8"))
    # 解析返回


def summarize(log: dict) -> dict:
    # 从 run_log 提取可对比的指标
    """提取指标。"""
    # 函数说明
    cases = log.get("cases", [])
    # 每例日志
    n = len(cases) or 1
    # 病例数，避免除零
    return {
        # 返回指标字典
        "cases": len(cases),
        # 病例数
        "success": log.get("summary", {}).get("success", 0),
        # 成功数
        "failed": log.get("summary", {}).get("failed", 0),
        # 失败数
        "total_calls": log.get("summary", {}).get("total_calls", 0),
        # 总调用次数
        "avg_calls": round(log.get("summary", {}).get("total_calls", 0) / n, 2),
        # 平均每例调用
        "avg_problems_before": round(sum(c.get("problems_before", 0) or 0 for c in cases) / n, 2),
        # 平均修复前问题数
        "avg_problems_after": round(sum(c.get("problems_after", 0) or 0 for c in cases) / n, 2),
        # 平均修复后问题数
        "avg_rule_fixes": round(sum(len(c.get("rule_fixes", []) or []) for c in cases) / n, 2),
        # 平均规则修复次数
        "avg_elapsed": round(sum(c.get("elapsed_sec", 0) or 0 for c in cases) / n, 3),
        # 平均耗时
        "offline_cases": log.get("summary", {}).get("offline_cases", 0),
        # 离线生成条数
    }
    # 结束


def write_report(run_dir: Path, base: dict, impr: dict, dry: bool, input_file: str) -> dict:
    # 组装并写入对比报告
    """写对比报告。"""
    # 函数说明
    delta_calls = round(impr["avg_calls"] - base["avg_calls"], 2)
    # 调用次数变化
    delta_problems = round(impr["avg_problems_after"] - base["avg_problems_after"], 2)
    # 残留问题变化
    reduction = None
    # 问题减少比例
    if base["avg_problems_after"]:
        # 基线有问题才能算比例
        reduction = round((1 - impr["avg_problems_after"] / base["avg_problems_after"]) * 100, 1)
        # 计算百分比
    report = {
        # 报告字典
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # 生成时间
        "mode": "dry_run" if dry else "live",
        # 实验类型
        "input_file": input_file or "input.json",
        # 输入文件
        "baseline": base,
        # 基线指标
        "improved": impr,
        # 改进版指标
        "delta": {
            # 差值
            "avg_calls": delta_calls,
            # 调用次数差
            "avg_problems_after": delta_problems,
            # 残留问题差（负数=改进版更好）
        },
        "conclusion": {
            # 结论
            "problem_reduction_pct": reduction,
            # 问题减少百分比
            "call_overhead": f"每例多消耗 {delta_calls} 次调用",
            # 成本代价
            "note": (
                "离线模式下的对比仅验证流程，真实效果需在有模型时重跑"
                if dry else "真实模型对比结果"
            ),
            # 备注
        },
    }
    # 组装完毕
    p = run_dir / "compare_report.json"
    # 报告路径
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # 写文件
    return report
    # 返回


def main() -> int:
    # 主流程
    """主函数。"""
    # 函数说明
    dry = "--dry-run" in sys.argv
    # 是否离线
    input_file = ""
    # 输入文件参数
    for i, a in enumerate(sys.argv):
        # 遍历参数
        if a == "--input" and i + 1 < len(sys.argv):
            # 找到 --input
            input_file = sys.argv[i + 1]
            # 取下一个值
    run_dir = make_run_dir()
    # 建本次实验目录
    print("=" * 64, file=sys.stderr)
    print(f"对照实验：{'离线模式（不消耗额度）' if dry else '真实模型模式'}", file=sys.stderr)
    print(f"输出目录：{run_dir}", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    # 打印标题
    for mode in ("baseline", "improved"):
        # 依次跑两种模式，必须用同一批输入
        code = run_mode(mode, dry, run_dir, input_file)
        # 执行
        if code != 0:
            # 失败就停
            print(f"[错误] {mode} 模式运行失败，退出码 {code}", file=sys.stderr)
            # 提示
            return code
            # 返回
    base = summarize(load_json(run_dir / "baseline_run_log.json"))
    # 基线指标
    impr = summarize(load_json(run_dir / "improved_run_log.json"))
    # 改进版指标
    if base["offline_cases"] or impr["offline_cases"]:
        # 有离线生成的
        print("[警告] 存在离线生成的病例，该对比不能作为真实效果证据", file=sys.stderr)
        # 明确警告
    report = write_report(run_dir, base, impr, dry, input_file)
    # 写报告
    print("", file=sys.stderr)
    # 空行
    print("=" * 64, file=sys.stderr)
    # 分隔
    print(f"基线版  ：每例调用 {base['avg_calls']} 次，残留问题 {base['avg_problems_after']}", file=sys.stderr)
    # 基线摘要
    print(f"改进版  ：每例调用 {impr['avg_calls']} 次，残留问题 {impr['avg_problems_after']}", file=sys.stderr)
    # 改进摘要
    print(f"规则修复：每例 {impr['avg_rule_fixes']} 处", file=sys.stderr)
    # 修复摘要
    if report["conclusion"]["problem_reduction_pct"] is not None:
        # 有减少比例
        print(f"问题减少：{report['conclusion']['problem_reduction_pct']}%", file=sys.stderr)
        # 打印
    print(f"报告已写入：{run_dir / 'compare_report.json'}", file=sys.stderr)
    # 报告路径
    print("=" * 64, file=sys.stderr)
    # 分隔
    return 0
    # 成功


if __name__ == "__main__":
    # 入口
    sys.exit(main())
    # 执行并返回退出码
