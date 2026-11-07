#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自检脚本 —— 用来验证校验器能不能抓到错误
==========================================
用途：读 output.json，对每一例重新跑一次校验，并做一次"故意篡改"测试，
      确认规则真的生效（而不是形同虚设）。

用法：
    python check_output.py

为什么需要它：
    程序跑通 ≠ 校验器有效。很多时候校验器"看起来在跑"，但其实规则写错了，
    或者被修复逻辑提前抹平了。这个脚本用"故意制造错误"的方式验证。
"""

import io
import json
import sys
from pathlib import Path

# Windows 控制台默认编码可能不是 UTF-8，强制一下避免中文乱码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
# 脚本所在目录


def load_json(name):
    # 读 JSON 文件，不存在就报错退出
    """读取 JSON。"""
    # 函数说明
    p = BASE / name
    # 拼路径
    if not p.exists():
        # 不存在
        print(f"[错误] 找不到 {name}，请先运行 python main_p1.py --dry-run", file=sys.stderr)
        # 提示
        return None
        # 返回空
    return json.loads(p.read_text(encoding="utf-8"))
    # 解析返回


def main():
    # 主流程
    """入口。"""
    # 函数说明
    import main_p1 as m
    # 导入主程序模块，复用它的校验器

    cases = load_json("input.json")
    # 输入病例
    output = load_json("output.json")
    # 生成结果
    if cases is None or output is None:
        # 缺文件
        return 1
        # 返回失败

    print("=" * 64)
    print("第一部分：检查现有输出的合规性")
    print("=" * 64)

    total_problems = 0
    # 累计问题数
    for case, rec in zip(cases, output):
        # 逐例配对
        cid = case.get("case_id")
        # 编号
        if rec.get("status") != "success":
            # 失败的例子跳过
            print(f"  {cid}: 状态 {rec.get('status')}，跳过")
            # 提示
            continue
            # 下一个
        facts = m.extract_facts(case)
        # 抽事实
        problems = m.validate(rec["medical_record"], case, facts)
        # 校验
        total_problems += len(problems)
        # 累加
        if problems:
            # 有问题
            print(f"  {cid}: 发现 {len(problems)} 个问题")
            # 打印
            for p in problems:
                # 逐条
                print(f"      [{p['rule']}] {p['msg']}")
                # 打印规则编号与说明
        else:
            # 没问题
            print(f"  {cid}: 通过")
            # 打印

    print()
    print("=" * 64)
    print("第二部分：故意制造错误，验证校验器能抓到")
    print("=" * 64)

    if not output or output[0].get("status") != "success":
        # 没有可用样本
        print("  没有可用于测试的成功样例，跳过")
        # 提示
        return 0
        # 结束

    case = cases[0]
    # 取第一例
    facts = m.extract_facts(case)
    # 抽事实
    base_record = json.loads(json.dumps(output[0]["medical_record"]))
    # 深拷贝一份，避免影响原数据

    tests = [
        # 构造几个"典型的模型错误"，逐个测试能否被抓
        ("把'未提供'改写成'正常'",
         lambda r: r.update({"case_features": r["case_features"] + "血压正常。"}),
         "R04_REWRITE"),
        ("写入已实施表述",
         lambda r: r.update({"plan": r["plan"] + "已完成检查。"}),
         "R03_DONE"),
        ("引用不存在的来源",
         lambda r: r.update({"evidence": [{"statement": "x", "source_ids": ["S999"], "kind": "fact"}]}),
         "R08_SRC_ID"),
        ("kind 写成非法值",
         lambda r: r.update({"evidence": [{"statement": "x", "source_ids": [case["sources"][0]["source_id"]], "kind": "猜测"}]}),
         "R09_KIND"),
        ("evidence 不足 3 条",
         lambda r: r.update({"evidence": []}),
         "R07_EVIDENCE_MIN"),
        ("review_items 不是数组",
         lambda r: r.update({"review_items": "字符串"}),
         "R10_REVIEW"),
    ]

    passed = 0
    # 通过的测试数
    for desc, mutate, expect_rule in tests:
        # 逐个测试
        rec = json.loads(json.dumps(base_record))
        # 深拷贝基准
        mutate(rec)
        # 施加篡改
        rules = [p["rule"] for p in m.validate(rec, case, facts)]
        # 校验
        hit = expect_rule in rules
        # 是否命中预期规则
        mark = "✓" if hit else "✗"
        # 标记
        print(f"  {mark} {desc}")
        # 打印结果
        print(f"      期望规则: {expect_rule} | 实际触发: {rules}")
        # 打印规则
        if hit:
            # 命中了
            passed += 1
            # 计数

    print()
    print("=" * 64)
    print(f"小结：{passed}/{len(tests)} 项篡改测试通过；现有输出共 {total_problems} 个残留问题")
    # 汇总
    if passed < len(tests):
        # 有未通过的
        print("有测试未通过，说明对应规则没生效，需要检查 validate() 的实现")
        # 提示
    else:
        # 全部通过
        print("校验器工作正常。注意：'现有输出残留问题' 应当为 0 或很少，否则说明生成质量需改进")
        # 提示
    print("=" * 64)
    return 0
    # 成功


if __name__ == "__main__":
    # 入口
    sys.exit(main())
    # 执行
