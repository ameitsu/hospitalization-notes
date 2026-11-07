#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病例生成器 —— 自动批量生成"病人资料卡"
=================================================
用途：造出大量符合赛题协议的合成病例，用于压测、对照实验、演示。

设计原则（很重要）：
  1. 全部合成，不对应任何真实患者。赛题要求"不得使用真实用户个人信息"，
     这里所有数据由模板随机组合而来，仅供测试。
  2. 可控随机：给定 --seed 就一定能复现出同一批病例。
     这对"改进方法验证"很关键——不能每次跑出来的数据都不一样。
  3. 刻意保留赛题的特点：资料有缺失、有否定表述、有截止点，
     这样才能测出程序在"信息不足"时的表现。
  4. 自动加陷阱：按概率注入"未提供""未见"等表述，以及少量超截止点资料。

用法：
    python generate_cases.py                      # 生成 20 例到 generated_cases.json
    python generate_cases.py -n 100               # 生成 100 例
    python generate_cases.py -n 50 --seed 7       # 指定种子，可复现
    python generate_cases.py --trap-ratio 0.5     # 50% 病例带陷阱

生成后：
    python main_p1.py --input generated_cases.json
"""

import argparse
import io
import json
import random
import sys
from pathlib import Path

# 强制 UTF-8 输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
# 脚本所在目录

# ============================================================
# 语料库。扩充这些列表就能立刻提高病例多样性。
# ============================================================
SURNAMES = ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴"]
# 姓氏（不生成完整姓名，避免像真实个人信息）
SEX = ["男", "女"]
# 性别
AGE_RANGES = [(18, 30), (31, 45), (46, 60), (61, 85)]
# 年龄段

SYMPTOM_POOLS = [
    # 每个元素是 (主诉症状池, 伴随症状池, 典型体格检查池, 对应化验池)
    (
        ["乏力", "头晕", "活动后心悸", "面色苍白"],
        ["食欲下降", "活动耐力下降", "双下肢水肿"],
        ["结膜苍白", "双下肢轻度水肿", "指甲苍白"],
        ["血红蛋白", "红细胞计数", "网织红细胞百分比"],
    ),
    (
        ["间断上腹痛", "餐后腹部不适", "反酸"],
        ["恶心", "体重下降", "黑便"],
        ["上腹压痛", "无反跳痛", "肠鸣音正常"],
        ["白细胞", "血红蛋白", "C反应蛋白"],
    ),
    (
        ["发热", "咳嗽", "咽痛"],
        ["咳痰", "畏寒", "乏力"],
        ["咽部充血", "肺部呼吸音粗", "未闻及明显干湿啰音"],
        ["白细胞", "中性粒细胞百分比", "C反应蛋白"],
    ),
    (
        ["活动后气促", "胸闷", "心悸"],
        ["夜间不能平卧", "双下肢水肿"],
        ["心率偏快", "双肺底少许湿啰音", "颈静脉无怒张"],
        ["B型利钠肽", "肌钙蛋白", "肌酐"],
    ),
    (
        ["尿量减少", "眼睑水肿", "乏力"],
        ["腰酸", "泡沫尿"],
        ["双下肢凹陷性水肿", "眼睑水肿", "血压偏高"],
        ["肌酐", "尿素氮", "尿蛋白"],
    ),
]
# 五类症状群，覆盖血液/消化/呼吸/心血管/肾脏

LAB_TABLE = {
    # 指标名 → (最小值, 最大值, 单位, 小数位, 实验室标低/标高概率)
    "血红蛋白": (70, 160, "g/L", 0, 0),
    "红细胞计数": (2.5, 5.5, "×10^12/L", 1, 0),
    "网织红细胞百分比": (0.5, 3.0, "%", 1, 0),
    "白细胞": (3.2, 18.6, "×10^9/L", 1, 0),
    "中性粒细胞百分比": (45, 92, "%", 0, 0),
    "C反应蛋白": (2, 180, "mg/L", 1, 0),
    "肌酐": (60, 320, "μmol/L", 0, 0),
    "尿素氮": (3, 22, "mmol/L", 1, 0),
    "尿蛋白": (0, 3, "+", 0, 0),
    "B型利钠肽": (20, 1800, "pg/mL", 0, 0),
    "肌钙蛋白": (0.01, 2.5, "ng/mL", 2, 0),
    "空腹血糖": (4.1, 11.2, "mmol/L", 1, 0),
    "糖化血红蛋白": (5.0, 9.5, "%", 1, 0),
    "血小板": (60, 380, "×10^9/L", 0, 0),
    "白蛋白": (25, 46, "g/L", 1, 0),
}
# 化验指标生成表。范围和单位都贴近临床常见值

MISSING_POOL = [
    # 常见"未提供"资料项
    "既往病史", "用药史", "过敏史", "家族史", "吸烟饮酒史",
    "心电图", "腹部超声", "胸部影像", "粪便隐血", "血培养",
    "红细胞指数", "铁代谢", "肾功能基线", "甲状腺功能", "糖化血红蛋白",
]
# 缺失资料池。
# ⚠️ 注意：不要把"体温""血压"这类**体征**混进来，
#    否则会生成"既往体温不详"这种不通的表述（实测发现过）

VITAL_POOL = ["体温", "脉率", "呼吸频率", "血压", "尿量", "体重变化"]
# 体征类缺失项，只能出现在体格检查段落里

NEGATIVE_POOL = [
    # 常见"有价值的阴性"表述
    "未见明确发热", "未见明显出血表现", "未见皮疹", "未见关节痛",
    "未闻及明显干湿啰音", "无反跳痛", "未见水肿", "未见意识改变",
]
# 阴性体征池


def fmt_lab(name, rng):
    # 按指标名生成一个合理数值
    """生成化验值。"""
    # 函数说明
    lo, hi, unit, dec, _ = LAB_TABLE[name]
    # 取范围与格式
    if unit == "+":
        # 尿蛋白这种定性指标
        return f"{name} {rng.choice(['阴性', '弱阳性', '阳性'])}"
        # 返回定性值
    if unit == "×10^12/L" or unit == "×10^9/L":
        # 计数类，用科学计数法
        val = round(rng.uniform(lo, hi), dec)
        # 生成数值
        return f"{name} {val}×10^{unit.split('^')[1].split('/')[0]}/L"
        # 组装科学计数法
    val = round(rng.uniform(lo, hi), dec)
    # 浮点型
    text = f"{int(val)}" if dec == 0 else f"{val}"
    # 整数或小数
    return f"{name} {text}{unit}"
    # 组装。⚠️ 数值与单位之间不加空格：数值带单位才是规范写法（如"82 g/L"是通例，
    #    但"47 %"这种百分比带空格的写法会让正则抽取变复杂，统一紧贴更稳）


def _set_blood(text, rng, low=True):
    # 把某个化验值改成"偏低"或"偏高"，让病例看起来真的有病
    """调整化验值。"""
    # 函数说明
    parts = text.split(" ")
    # 拆成 名称/数值/单位
    if len(parts) < 3:
        # 格式不符（比如定性指标）就直接返回
        return text
        # 不动
    name = parts[0]
    # 指标名
    rest = parts[1:]
    # 其余部分
    num_part = rest[0]
    # 数值串
    if "×10^" in num_part:
        # 科学计数法，保持结构只改数值
        head, tail = num_part.split("×10^", 1)
        # 拆开
        try:
            # 尝试解析
            v = float(head)
            # 数值
            v = round(v * rng.uniform(0.55, 0.8), 1) if low else round(v * rng.uniform(1.3, 1.9), 1)
            # 压低或抬高
            return f"{name} {v}×10^{tail} " + " ".join(rest[1:])
            # 重组
        except ValueError:
            # 解析失败就原样返回
            return text
            # 不动
    try:
        # 普通数值
        v = float(num_part)
        # 解析
        v = round(v * rng.uniform(0.55, 0.8), 1) if low else round(v * rng.uniform(1.3, 1.9), 1)
        # 调整
        shown = f"{int(v)}" if v == int(v) else f"{v}"
        # 整数去掉小数点
        return f"{name} {shown} " + " ".join(rest[1:])
        # 重组
    except ValueError:
        # 解析失败
        return text
        # 原样返回


def make_case(idx, rng, trap_ratio):
    # 生成一例病例
    """生成单个病例。"""
    # 函数说明
    surname = rng.choice(SURNAMES)
    # 取姓
    sex = rng.choice(SEX)
    # 性别
    lo, hi = rng.choice(AGE_RANGES)
    # 年龄段
    age = rng.randint(lo, hi)
    # 年龄
    main_sx, assoc_sx, exams, labs = rng.choice(SYMPTOM_POOLS)
    # 选一个症状群
    duration = rng.choice(["3 天", "1 周", "2 周", "1 个月", "3 个月"])
    # 病程

    sources = []
    # 资料片段
    sid = 1
    # 编号计数

    # S01 主诉 + 现病史
    chief = f"{sex}，{age} 岁，{rng.choice(main_sx)} {duration}"
    assoc = rng.sample(assoc_sx, k=min(2, len(assoc_sx)))
    # 伴随症状
    s01 = chief + "；伴" + "、".join(assoc)
    # 主诉串
    sources.append({"source_id": f"S{sid:02d}", "content": s01})
    # 加入
    sid += 1

    # S02 体格检查
    exam_text = "体格检查：" + "，".join(rng.sample(exams, k=min(2, len(exams))))
    # 查体描述
    if rng.random() < trap_ratio:
        # 按概率加阴性表述
        exam_text += "；" + rng.choice(NEGATIVE_POOL)
        # 追加阴性
    if rng.random() < trap_ratio * 0.7:
        # 按概率加体征缺失（只用体征词，不能用既往史词）
        exam_text += f"；未提供{rng.choice(VITAL_POOL)}"
        # 追加体征缺失
    sources.append({"source_id": f"S{sid:02d}", "content": exam_text})
    # 加入
    sid += 1

    # S03 辅助检查（1-3 项）
    pick_labs = rng.sample(labs, k=min(rng.randint(1, 3), len(labs)))
    # 选化验
    lab_items = []
    # 化验条目
    for name in pick_labs:
        # 逐项生成
        lab_items.append(fmt_lab(name, rng))
        # 追加
    # 让关键指标"看起来有病"：按**指标名**逐项判断，而不是只看第一项。
    # ⚠️ 原实现只检查 lab_items[0]，而 pick_labs 的顺序是随机的，
    #    导致大部分病例的血红蛋白仍是正常值（实测发现）。
    for i, name in enumerate(pick_labs):
        # 逐项遍历
        if rng.random() > 0.75:
            # 部分指标保持正常，让数据不显得过于整齐
            continue
            # 跳过
        if name in ("血红蛋白", "红细胞计数", "白蛋白", "血小板"):
            # 这几项偏低才有临床意义
            lab_items[i] = _set_blood(lab_items[i], rng, low=True)
            # 压低
        elif name in ("肌酐", "尿素氮", "C反应蛋白", "白细胞", "空腹血糖",
                      "B型利钠肽", "肌钙蛋白", "糖化血红蛋白"):
            # 这几项偏高才有临床意义
            lab_items[i] = _set_blood(lab_items[i], rng, low=False)
            # 抬高
    lab_text = "辅助检查：" + "；".join(lab_items)
    # 化验串
    sources.append({"source_id": f"S{sid:02d}", "content": lab_text})
    # 加入
    sid += 1

    # S04 既往史 / 缺失项（只用既往史类词）
    missing = rng.sample(MISSING_POOL, k=rng.randint(1, 3))
    # 选缺失项
    if rng.random() < trap_ratio:
        # 方案 A：全部未提供
        s04 = "、".join(missing) + "未提供"
        # 串起来
    else:
        # 方案 B：部分给出
        if len(missing) > 1:
            # 有多项
            s04 = f"{missing[0]}不详；" + "、".join(missing[1:]) + "未提供"
            # 前面不详、后面未提供
        else:
            # 只有一项
            s04 = f"{missing[0]}未提供，余无特殊"
            # 单项
    sources.append({"source_id": f"S{sid:02d}", "content": s04})
    # 加入
    sid += 1

    # 可选的截止点陷阱：注入一段"T0 之后"的资料
    if rng.random() < trap_ratio * 0.3:
        # 低概率注入
        after = rng.choice([
            f"T0+1 天：复查{pick_labs[0]}，结果待回报",
            f"入院后第 2 天：完善{rng.choice(['腹部超声', '胸部 CT', '心电图'])}，提示异常",
            f"截止点之后完成：{rng.choice(['血培养', '骨髓穿刺', '胃镜'])}，结果未回报",
        ])
        # 未来时点资料
        sources.append({"source_id": f"S{sid:02d}", "content": after})
        # 加入
        sid += 1

    # 最后一例：入院初步判断（有时给，有时不给）
    if rng.random() < 0.5:
        # 一半概率给
        sources.append({
            "source_id": f"S{sid:02d}",
            "content": f"入院初步判断：{rng.choice(['原因待查', '待进一步明确', '初步考虑上述症状相关疾病可能'])}",
        })
        # 加入
        sid += 1

    return {
        # 返回病例
        "case_id": f"GEN{idx:04d}",
        # 编号
        "record_cutoff": "T0",
        # 截止点
        "sources": sources,
        # 资料
    }
    # 结束


def main():
    # 主流程
    """入口。"""
    # 函数说明
    p = argparse.ArgumentParser(description="合成病例生成器")
    # 解析器
    p.add_argument("-n", "--num", type=int, default=20)
    # 生成数量
    p.add_argument("--seed", type=int, default=None)
    # 随机种子，指定后可复现
    p.add_argument("--trap-ratio", type=float, default=0.6)
    # 陷阱注入比例
    p.add_argument("--out", default="generated_cases.json")
    # 输出文件名
    args = p.parse_args()
    # 解析
    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    # 种子：没指定就随机（但要打印出来，方便复现）
    rng = random.Random(seed)
    # 独立随机源，不污染全局
    cases = [make_case(i + 1, rng, args.trap_ratio) for i in range(args.num)]
    # 批量生成
    out = BASE / args.out
    # 输出路径
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    # 写文件
    total_src = sum(len(c["sources"]) for c in cases)
    # 资料总数
    total_chars = sum(len(s["content"]) for c in cases for s in c["sources"])
    # 总字数
    print(f"已生成 {len(cases)} 例 → {out.name}")
    # 提示
    print(f"随机种子：{seed}（用 --seed {seed} 可复现同一批数据）")
    # 打印种子，保证可复现
    print(f"合计 {total_src} 段资料，{total_chars} 字，平均每例 {total_src // max(1, len(cases))} 段")
    # 统计
    print()
    print("下一步：")
    print(f"  python main_p1.py --input {args.out} --dry-run")
    print(f"  python main_p1.py --input {args.out}")
    # 使用提示
    return 0
    # 成功


if __name__ == "__main__":
    # 入口
    sys.exit(main())
    # 执行
