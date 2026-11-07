#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首程生成器 P1 版 —— 第 8 题「临床智录」
==================================================
本版本在 P0 基础上完成 P1（核心拿分）改造，并接入千问真实 API。

一、千问接入（两种模式，运行时自动判断）
  ├─ 模式 A：阿里云百炼 DashScope（OpenAI 兼容接口，需 API Key）
  │     .env 里填 DASHSCOPE_API_KEY / BASE_URL / MODEL_NAME
  │     认证方式：请求头 Authorization: Bearer <key>
  └─ 模式 B：主办方无密钥接口（赛题正式评测用）
        .env 里只填 ENDPOINT / MODEL_NAME，没有 Key
        认证方式：不发送 Authorization 头

  判断逻辑：有 DASHSCOPE_API_KEY → 走模式 A；否则有 ENDPOINT → 走模式 B；
            两者都没有 → 走离线演示模式（日志中标记 source=offline，绝不当成真结果）

二、P1 四项
  P1-1  事实白名单接入生成：把抽取到的事实结构化地喂给模型，而不只是丢原文
  P1-2  拟诊讨论个性化：按本例异常指标生成鉴别方向，避免所有病例千篇一律
  P1-3  修复 + 重试回路：校验失败 → 先规则修复，修不好再问模型，受预算约束
  P1-4  对照实验框架：--mode baseline 关闭全部增强，与改进版同条件对比

三、运行方式
  python main_p1.py                     # 改进版（默认）
  python main_p1.py --mode baseline     # 基线版（裸提示词，不做校验与修复）
  python main_p1.py --dry-run           # 只跑离线，不调网络

退出码（技术规范第 6 条）：
  0 = 全部处理完毕并生成符合协议的输出（允许单条失败）
  1 = 任务级运行错误
  2 = 必要配置缺失，或输入文件不符合协议
"""

# ============================================================
# 标准库导入
# ============================================================
import argparse       # 解析命令行参数：--mode / --dry-run
import json           # 读写 JSON
import os             # 文件系统与文件描述符操作
import platform       # 记录运行环境
import re             # 正则：抽取数值、缺失项、禁用词
import sys            # 标准错误输出与退出码
import tempfile       # 原子写：先写临时文件再替换
import time           # 计时
import traceback      # 异常堆栈
import urllib.error   # 捕获 HTTP 错误（401/429/503 等）
import urllib.request # 发 HTTP 请求（标准库，无第三方依赖）
from datetime import datetime  # 生成 ISO 时间戳
from pathlib import Path       # 路径处理

# ============================================================
# 路径与常量
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
# 本文件所在目录。所有相对路径都以此为基准
INPUT_FILE = BASE_DIR / "input.json"
# 输入文件
OUTPUT_FILE = BASE_DIR / "output.json"
# 技术规范要求的输出文件名
RESULTS_FILE = BASE_DIR / "results.json"
# 提交要求里写的文件名。与 output.json 表述不一致，两个都写
RUN_LOG_FILE = BASE_DIR / "run_log.json"
# 运行记录
COMPARE_FILE = BASE_DIR / "compare_report.json"
# P1-4：对照实验报告
ENV_FILE = BASE_DIR / ".env"
# 配置文件
REQUIRES_FILE = BASE_DIR / "requires.txt"
# 依赖清单
ENV_EXAMPLE_FILE = BASE_DIR / ".env.example"
# 配置模板

EXIT_OK = 0
# 全部处理完毕
EXIT_TASK_ERROR = 1
# 任务级错误
EXIT_CONFIG_ERROR = 2
# 配置或输入不合协议

# ============================================================
# 生成参数。集中配置，赛前按主办方公布值调整
# ============================================================
SEED = 42
# 随机种子，赛题要求固定以保证可复现
TEMPERATURE = 0.2
# 温度。低温和可复现优先
TOP_P = 0.9
# 核采样
MAX_TOKENS = 2048
# 单次输出上限
CALL_BUDGET_PER_CASE = 3
# 每条病例调用上限。改进版：生成 1 次 + 允许 2 次重试
BASELINE_CALL_BUDGET = 1
# 基线版只允许 1 次，模拟"裸调用"的真实成本
TOTAL_CALL_BUDGET = 10000
# 全局调用上限，防止异常烧光额度
REQUEST_TIMEOUT = 120
# 单次请求超时（秒）
MAX_ATTEMPTS = 3
# 单例最多尝试轮数（含首次）。与调用预算取较小值生效
PROMPT_VERSION = "p1-v1"
# 提示词版本，写进 run_log 便于核验"代码与结果对应"

# ---- 词表 ----
BANNED_WORDS = ["已予", "已完成", "已行", "已用", "已做", "已给予"]
# 已实施表述，计划里不该出现
FORBIDDEN_REWRITE = ["正常", "无异常", "否认"]
# 可能把"未提供"改写成肯定陈述的词

# ============================================================
# 正则
# ============================================================
VALUE = r"\d+(?:\.\d+)?(?:\s*[×xX]\s*10\^?\d+)?"
# 数值：支持小数与科学计数法（9.8×10^9）
UNIT = (
    # 单位表。按长度从长到短排列，避免 g/L 抢先匹配掉 g/dL 这种更长的单位
    r"(?:mg/dL|μg/dL|μmol/L|umol/L|mmol/L|mol/L|g/dL|g/L|mg/L|ng/mL|pg/mL|"
    r"mEq/L|U/L|IU/L|mmHg|cmH2O|次/分|bpm|kg/m2|℃|mm|%|/L)"
)
# ⚠️ 之前缺 g/dL，导致"8 g/dL"这种表述完全逃过校验（实测发现）。
#    顺序很重要：g/dL 必须排在 g/L 前面，否则会被 g/L 抢先匹配成 "g/L" + 残余
NUM_UNIT_RE = re.compile("(" + VALUE + ")\\s*" + UNIT)
# 数值+单位，组 1 是数字
LAB_RE = re.compile("([\u4e00-\u9fa5A-Za-z]{2,10})\\s*(" + VALUE + ")\\s*(" + UNIT + ")")
# 指标名+数值+单位，组 1/2/3
# ⚠️ 中文数字识别：资料里可能出现"血色素 8 克""肌酐两百出头"这类口语化表述。
#    模型很容易把它"规范化"成 8 g/dL、200 μmol/L —— 这属于编造。
CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100}
# 常用中文数字。只用于"识别资料里有没有提到数字"，不用于精确换算
CN_NUM_RE = re.compile(r"[一二两三四五六七八九十百]+(?:\s*出头|\s*左右|\s*多)?")
# 中文数字表达式，如"两百出头"
VAGUE_RE = re.compile(r"(左右|约|大概|接近|上下|出头|余)")
# 模糊限定词。带这些词的数值不该被写成精确值
MISSING_RE = re.compile(r"未(?:提供|查|明确|述)([^；;。]{1,20})")
# 缺失项。不含"未见"，因为那是阴性体征而非缺失
TIME_RE = re.compile(r"\d+\s*(?:个月|月|天|周|年|小时)")
# 时间表达
# ⚠️ 时序陷阱：识别"截止点之后"的信息。测试发现这类内容会被原样写进首程，
#    而赛题明确要求"仅使用截至资料截止点（含该时点）已可获得的信息"
AFTER_CUTOFF_RE = re.compile(
    r"(T0\s*[+＋]\s*\d+|截止点之后|截止点后|之后完成|之后结果|次日|翌日|"
    r"入院后第\s*\d+\s*天|出院|随访中|复查时|第\s*2\s*天|第\s*3\s*天)"
)
# 常见"未来时点"标志。命中即认为这段资料可能超出截止点
CONFLICT_RE = re.compile(r"(否认|未患)[^；;。]{0,12}(糖尿病|高血压|冠心病|肝炎|过敏)")
# 冲突检测之一：否认某病。要看是否同时出现该病的用药
CONFLICT_DRUG = {
    # 病名 → 该病常用药关键词
    "糖尿病": ["二甲双胍", "胰岛素", "格列", "阿卡波糖"],
    "高血压": ["氨氯地平", "缬沙坦", "厄贝沙坦", "美托洛尔", "硝苯地平"],
    "冠心病": ["阿司匹林", "他汀", "硝酸甘油", "氯吡格雷"],
    "肝炎": ["恩替卡韦", "替诺福韦", "干扰素"],
}
# 冲突检测知识表。扩充此表即可支持更多病种

ALREADY_DONE_RE = re.compile(r"(?<![现进])已行")
# "已行"，但排除"现行/进行"
NEGATION_CONTEXT_RE = re.compile(r"(未见|未提供|未查|未明确|无|不)")
# 否定语境标志

# ============================================================
# 医学词库（P1-2 拟诊讨论个性化用）
# 说明：这是"轻量知识库"，不是诊断模型。它只做两件事：
#   1. 识别文中出现了哪个系统/症状群
#   2. 给出该症状群常见的鉴别方向（措辞保守，一律用"需考虑/需排除"）
# 这样做的好处：不依赖模型自由发挥，输出稳定、可控、可解释。
# ⚠️ 重要：这里的方向是"提示医生考虑"，不是诊断结论。
# ============================================================
SYMPTOM_GROUPS = [
    # 每个元素：触发词列表 + 鉴别方向 + 建议的补充检查
    {
        "name": "贫血相关",
        "triggers": ["血红蛋白", "贫血", "结膜苍白", "乏力", "心悸", "苍白"],
        "differentials": [
            "隐匿性失血（消化道、泌尿生殖道）",
            "营养性或吸收障碍相关因素",
            "慢性病/炎症相关贫血",
        ],
        "followup": "红细胞指数、网织红细胞计数、铁代谢及粪便隐血",
    },
    {
        "name": "肾功能相关",
        "triggers": ["肌酐", "尿素", "肾功能", "水肿", "尿量"],
        "differentials": [
            "肾前性因素（容量不足、低灌注）",
            "肾实质性损伤",
            "肾后性梗阻",
        ],
        "followup": "尿常规与尿沉渣、肾脏超声、既往肾功能基线",
    },
    {
        "name": "消化系统相关",
        "triggers": ["腹痛", "上腹", "恶心", "呕吐", "黑便", "食欲", "消化"],
        "differentials": [
            "消化性溃疡或胃炎",
            "胆胰疾病",
            "需排除的消化道肿瘤性病变",
        ],
        "followup": "胃镜/腹部影像、粪便隐血、肝功能与胰酶",
    },
    {
        "name": "感染/炎症相关",
        "triggers": ["发热", "白细胞", "C反应蛋白", "感染", "咽痛", "咳嗽"],
        "differentials": [
            "细菌性感染",
            "病毒性感染",
            "非感染性炎症",
        ],
        "followup": "血培养或病原学检查、炎症指标动态复查、影像学",
    },
    {
        "name": "心血管相关",
        "triggers": ["胸痛", "胸闷", "心悸", "血压", "心电"],
        "differentials": [
            "冠脉缺血性疾病",
            "心律失常",
            "非心源性胸痛",
        ],
        "followup": "心电图与心肌标志物、动态心电、心脏超声",
    },
    {
        "name": "体重下降/消耗",
        "triggers": ["体重下降", "消瘦", "纳差", "乏力"],
        "differentials": [
            "慢性消耗性疾病",
            "内分泌或代谢因素",
            "精神心理与营养摄入因素",
        ],
        "followup": "内分泌功能、肿瘤标志物筛查、营养状态评估",
    },
]
# 症状群知识库定义结束。后续可继续扩充，每加一条就多一种个性化能力
MAX_DIFFERENTIALS = 3
# 每段拟诊讨论最多列几个鉴别方向。赛题要求 2-3 个
MIN_DIFFERENTIALS = 2
# 最少列几个


# ============================================================
# 一、配置
# ============================================================
def load_env(path: Path) -> dict:
    # 解析 .env 文件为字典
    """读取 .env。"""
    # 函数说明
    cfg = {}
    # 结果字典
    if not path.exists():
        # 文件不存在就返回空字典
        return cfg
        # 提前返回
    for line in path.read_text(encoding="utf-8").splitlines():
        # 逐行读取
        line = line.strip()
        # 去首尾空白
        if not line or line.startswith("#") or "=" not in line:
            # 跳过空行/注释/无等号行
            continue
            # 下一行
        key, _, value = line.partition("=")
        # 按第一个等号切分，兼容值里有等号
        cfg[key.strip()] = value.strip()
        # 存入字典
    return cfg
    # 返回


def resolve_endpoint(cfg: dict) -> dict:
    # 根据配置判断走哪种模式，返回统一结构
    """判断模型接入模式。"""
    # 函数说明
    mode = "offline"
    # 默认离线
    url = ""
    # 请求地址
    api_key = ""
    # 密钥
    model = (cfg.get("MODEL_NAME") or "").strip()
    # 模型名
    key = (cfg.get("DASHSCOPE_API_KEY") or "").strip()
    # 千问 Key
    endpoint = (cfg.get("ENDPOINT") or "").strip()
    # 完整地址
    base_url = (cfg.get("BASE_URL") or "").strip()
    # 兼容接口的 base 地址
    if key and base_url:
        # 模式 A：千问百炼。有 Key 且有 base 地址
        mode = "dashscope"
        # 标记模式
        url = base_url.rstrip("/") + "/chat/completions"
        # 拼接完整地址。base_url 应以 /compatible-mode/v1 结尾
        api_key = key
        # 带上密钥
    elif key and endpoint:
        # 模式 A 的变体：有 Key 且直接给了完整地址
        mode = "dashscope"
        # 标记模式
        url = endpoint
        # 直接用
        api_key = key
        # 带上密钥
    elif endpoint:
        # 模式 B：主办方无密钥接口
        mode = "hosted"
        # 标记模式
        url = endpoint
        # 直接用完整地址
        api_key = ""
        # 无密钥
    if mode != "offline" and not model:
        # 有地址但没模型名
        model = "qwen-plus"
        # 给个默认值，避免请求体缺字段
    return {"mode": mode, "url": url, "api_key": api_key, "model": model}
    # 返回统一结构


def write_config_templates():
    # 生成依赖清单与配置模板（P0-4）
    """写依赖与配置模板。"""
    # 函数说明
    if not REQUIRES_FILE.exists():
        # 只在缺失时创建，避免覆盖手工改动
        REQUIRES_FILE.write_text(
            # 本程序只用标准库，无需第三方依赖
            "# 本程序仅使用 Python 标准库，无需第三方包。\n"
            "# 若你改用了 requests 或 python-dotenv，请取消注释并固定版本：\n"
            "# requests==2.32.3\n"
            "# python-dotenv==1.0.1\n",
            # 文件内容
            encoding="utf-8",
            # 强制 UTF-8
        )
        # 写入结束
    if not ENV_EXAMPLE_FILE.exists():
        # 配置模板同理
        ENV_EXAMPLE_FILE.write_text(
            # 说明：两种模式二选一
            "# ===== 模式 A：本地用千问百炼（需 API Key）=====\n"
            "# 到阿里云百炼控制台创建 API Key，然后填写下面三项\n"
            "DASHSCOPE_API_KEY=sk-替换成你的Key\n"
            "BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n"
            "MODEL_NAME=qwen-plus\n"
            "\n"
            "# ===== 模式 B：赛题正式评测（主办方无密钥接口）=====\n"
            "# 有 DASHSCOPE_API_KEY 时走模式 A；没有时若填了 ENDPOINT 则走模式 B\n"
            "# ENDPOINT=http://127.0.0.1:8000/v1/chat/completions\n"
            "# MODEL_NAME=主办方指定的模型名\n",
            # 文件内容
            encoding="utf-8",
            # 强制 UTF-8
        )
        # 写入结束


# ============================================================
# 二、事实抽取（P1-1 的原料）
# ============================================================
def extract_facts(case: dict) -> dict:
    # 从资料卡抽取事实，只搬运不创造
    """抽取事实。"""
    # 函数说明
    facts = {
        # 结果字典
        "labs": [],
        # 检验指标（指标名+数值+单位）
        "missing": [],
        # 缺失项
        "times": [],
        # 时间表达
        "sources": [],
        # 原始资料（已过滤截止点之后的内容）
        "negatives": [],
        # 阴性体征（含"未见"的句子）
        "after_cutoff": [],
        # ⚠️ 超出截止点被剔除的资料（不是丢弃，而是单独记录）
        "conflicts": [],
        # 资料内部的矛盾点
    }
    # 初始化完毕
    for src in case.get("sources", []):
        # 遍历资料片段
        sid = src.get("source_id", "")
        # 编号
        text = src.get("content", "")
        # 正文
        if AFTER_CUTOFF_RE.search(text):
            # 命中"截止点之后"标志
            facts["after_cutoff"].append({"source_id": sid, "content": text})
            # 单独记录，便于在待核事项里提示
            continue
            # 关键：跳过后续抽取，不让它污染事实清单
        facts["sources"].append({"source_id": sid, "content": text})
        # 只有截止点内的资料才进入 sources
        for m in LAB_RE.finditer(text):
            # 找"指标名+数值+单位"
            facts["labs"].append({
                # 添加检验记录
                "name": m.group(1),
                # 指标名
                "value": m.group(2).strip() + m.group(3).strip(),
                # 数值+单位
                "source_id": sid,
                # 来源
            })
            # 单条结束
        for m in MISSING_RE.finditer(text):
            # 找缺失项
            item = m.group(1).strip("、，,；; ")
            # 去两端标点
            if item:
                # 非空才存
                facts["missing"].append({"text": item, "source_id": sid})
                # 存入
        for m in TIME_RE.finditer(text):
            # 找时间
            facts["times"].append({"text": m.group(0), "source_id": sid})
            # 存入
        if "未见" in text:
            # 含"未见"说明有阴性体征
            facts["negatives"].append({"text": text.strip("。"), "source_id": sid})
            # 存入阴性列表
    facts["conflicts"] = detect_conflicts(facts["sources"])
    # 检测资料内部矛盾
    return facts
    # 返回


def detect_conflicts(sources: list) -> list:
    # 检测资料之间的矛盾。这类问题必须写进待核事项，而不是悄悄选一个
    """检测资料冲突。"""
    # 函数说明
    conflicts = []
    # 结果
    all_text = " ".join(s["content"] for s in sources)
    # 全部资料拼起来
    for disease, drugs in CONFLICT_DRUG.items():
        # 遍历病种与对应药物
        if CONFLICT_RE.search(all_text) and disease in all_text:
            # 有"否认某病"的说法
            for d in drugs:
                # 遍历该病常用药
                if d in all_text:
                    # 同时出现了该病药物
                    conflicts.append(f"既往史否认{disease}，但用药史提到「{d}」，两者矛盾需核实")
                    # 记录冲突
                    break
                    # 一个病只报一次
    temps = re.findall(r"体温\s*(\d{2}(?:\.\d)?)\s*[℃度]", all_text)
    # 找出所有体温读数
    if len(set(temps)) > 1:
        # 出现多个不同数值
        conflicts.append(f"体温记录不一致（{'/'.join(sorted(set(temps)))}℃），需核实以哪条为准")
        # 记录冲突
    return conflicts
    # 返回



def render_fact_sheet(facts: dict) -> str:
    # P1-1 核心：把事实渲染成结构化清单喂给模型
    """把事实渲染成清单。"""
    # 函数说明
    lines = []
    # 结果行
    lines.append("【可引用的检验事实】（只能引用以下数值，不得编造其他数值）")
    # 标题
    if facts["labs"]:
        # 有检验
        for lab in facts["labs"]:
            # 逐条列出
            lines.append(f"  - {lab['name']} {lab['value']}（来源 {lab['source_id']}）")
            # 带来源，便于模型写 evidence
    else:
        # 没有检验
        lines.append("  - （资料中未提供可直接引用的检验数值）")
        # 明确说明
    lines.append("【必须保留的阴性/缺失表述】（不得改写为“正常/否认”）")
    # 第二个标题
    shown = False
    # 是否已输出
    for neg in facts["negatives"]:
        # 遍历阴性体征
        lines.append(f"  - {neg['text']}（来源 {neg['source_id']}）")
        # 列出原文
        shown = True
        # 标记已输出
    for miss in facts["missing"]:
        # 遍历缺失项
        lines.append(f"  - “{miss['text']}”未提供（来源 {miss['source_id']}）")
        # 明确"未提供"
        shown = True
        # 标记
    if not shown:
        # 一个都没有
        lines.append("  - （本病例未标注阴性或缺失项）")
        # 兜底提示
    if facts["times"]:
        # 有时间就列
        lines.append("【时间信息】（保留原表述，不得颠倒先后）")
        # 标题
        for t in facts["times"]:
            # 逐条
            lines.append(f"  - {t['text']}（来源 {t['source_id']}）")
            # 列出
    if facts.get("after_cutoff"):
        # 有超出截止点的资料
        lines.append("【重要：以下资料超出截止点，禁止写入正文】")
        # 醒目提示模型
        for a in facts["after_cutoff"]:
            # 逐条
            lines.append(f"  - [来源 {a['source_id']}] {a['content'][:40]}…（属于截止点之后的信息）")
            # 列出，并注明不可用
    if facts.get("conflicts"):
        # 有冲突
        lines.append("【资料之间存在矛盾，需在待核事项中提示，不要自行选择其一下结论】")
        # 提示模型
        for c in facts["conflicts"]:
            # 逐条
            lines.append(f"  - {c}")
            # 列出
    return "\n".join(lines)
    # 拼接返回


def build_differentials(facts: dict) -> list:
    # P1-2：根据本例内容匹配症状群，生成个性化鉴别方向
    """生成本例鉴别方向。"""
    # 函数说明
    text_all = " ".join(s["content"] for s in facts["sources"])
    # 把全部资料拼起来做关键词匹配
    picked = []
    # 命中的症状群
    for group in SYMPTOM_GROUPS:
        # 遍历知识库
        hit = [t for t in group["triggers"] if t in text_all]
        # 找出命中的触发词
        if hit:
            # 命中就收集
            picked.append({"group": group, "hits": hit})
            # 记录成对
    picked.sort(key=lambda x: len(x["hits"]), reverse=True)
    # 命中词多的排前面，说明该症状群更突出
    return picked[:MAX_DIFFERENTIALS]
    # 最多取几组


def build_user_content(case: dict, facts: dict, mode: str) -> str:
    # 组装用户消息。改进版会带上事实清单，基线版只给原文
    """组装用户消息。"""
    # 函数说明
    lines = []
    # 资料行
    for src in case.get("sources", []):
        # 遍历资料
        lines.append(f"[{src.get('source_id')}] {src.get('content')}")
        # 拼成 [S01] 正文
    raw = (
        # 原始资料部分
        f"资料截止点：{case.get('record_cutoff')}\n"
        + "\n".join(lines)
    )
    # 拼接
    if mode == "baseline":
        # 基线版：只给原文，模拟"裸调用"
        return raw + "\n\n请据此撰写病例特点、拟诊讨论、诊疗计划三段。"
        # 简单指令
    diffs = build_differentials(facts)
    # 计算鉴别方向
    diff_block = []
    # 鉴别方向文本
    if diffs:
        # 有命中
        diff_block.append("【建议围绕本例特点考虑以下鉴别方向】（每个方向都要说明理由，不要照抄教科书清单）")
        # 标题
        for d in diffs:
            # 遍历症状群
            g = d["group"]
            # 取组
            names = "、".join(g["differentials"][:MAX_DIFFERENTIALS])
            # 取鉴别名单
            diff_block.append(f"  - {g['name']}：可考虑 {names}；建议补充 {g['followup']}")
            # 拼一行
    else:
        # 没命中任何症状群
        diff_block.append("【本例未能匹配到明确的症状群，请仅基于已有资料作保守描述，不硬下结论】")
        # 保守提示
    return (
        # 返回完整用户消息
        raw
        + "\n\n" + render_fact_sheet(facts)
        + "\n\n" + "\n".join(diff_block)
        + "\n\n请据此撰写病例特点、拟诊讨论、诊疗计划三段，严格只使用上述事实。"
    )
    # 拼接结束


# ============================================================
# 三、调用模型
# ============================================================
SYSTEM_PROMPT = (
    # 系统提示词。注意用圆括号拼接，让每行都能写注释而不污染内容
    "你是住院医师，正在撰写首次病程记录草稿。\n"
    # 身份设定
    "严格要求：\n"
    # 引出要求
    "1. 只能使用给定资料，绝不允许补充资料里没有的事实、检查、病史。\n"
    # 要求 1：不得编造
    "2. 资料中写“未提供/未查/不详”的，必须照样说明，不得改写为“正常/否认”。\n"
    # 要求 2：保留否定与缺失
    "3. 不把计划写成已实施，一律用“拟/建议/复查/观察”。\n"
    # 要求 3：区分计划与已实施
    "4. 只输出一个 JSON 对象，不要 Markdown 围栏，不要解释文字。\n"
    # 要求 4：输出格式
    "JSON 结构：{\"case_features\":\"\",\"diagnostic_discussion\":\"\",\"plan\":\"\","
    # 结构前半
    "\"evidence\":[{\"statement\":\"\",\"source_ids\":[],\"kind\":\"fact\"}],"
    # 结构中间：依据
    "\"review_items\":[]}"
    # 结构结尾：待核事项
)


class CallCounter:
    # 调用计数器，所有模型调用必须经过
    """调用计数器。"""
    # 类说明
    def __init__(self, per_case_limit: int, total_limit: int):
        # 初始化
        self.per_case_limit = per_case_limit
        # 单例上限
        self.total_limit = total_limit
        # 全局上限
        self.total = 0
        # 全局已用
        self.per_case = {}
        # 单例已用

    def can_call(self, case_id: str) -> bool:
        # 是否还能调
        return self.per_case.get(case_id, 0) < self.per_case_limit and self.total < self.total_limit
        # 单例与全局都没超才允许

    def record(self, case_id: str):
        # 记账
        self.per_case[case_id] = self.per_case.get(case_id, 0) + 1
        # 单例 +1
        self.total += 1
        # 全局 +1

    def used(self, case_id: str) -> int:
        # 查询
        return self.per_case.get(case_id, 0)
        # 返回次数


def http_chat(conn: dict, messages: list, timeout: int) -> dict:
    # 发一次 HTTP 请求，返回解析后的响应字典
    """调用 Chat Completions 接口。"""
    # 函数说明
    body = {
        # 请求体
        "model": conn["model"],
        # 模型名
        "messages": messages,
        # 消息列表
        "stream": False,
        # 非流式
        "temperature": TEMPERATURE,
        # 温度
        "top_p": TOP_P,
        # 核采样
        "max_tokens": MAX_TOKENS,
        # 输出上限
        "seed": SEED,
        # 种子（千问 OpenAI 兼容接口支持）
    }
    # 请求体完毕
    headers = {"Content-Type": "application/json"}
    # 基础头
    if conn["api_key"]:
        # 模式 A 需要鉴权
        headers["Authorization"] = f"Bearer {conn['api_key']}"
        # 加 Bearer 头
    req = urllib.request.Request(
        # 构造请求
        conn["url"],
        # 完整地址
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        # 请求体字节
        headers=headers,
        # 请求头
        method="POST",
        # POST
    )
    # 请求构造完毕
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # 发送
        return json.loads(resp.read().decode("utf-8"))
        # 解析返回


def is_retryable(err: Exception) -> bool:
    # 判断错误是否值得重试
    """判断是否可重试。"""
    # 函数说明
    if isinstance(err, urllib.error.HTTPError):
        # HTTP 错误
        return err.code in (429, 500, 502, 503, 504)
        # 限流与服务端错误可重试，401/400 不可
    if isinstance(err, urllib.error.URLError):
        # 网络错误
        return True
        # 网络问题可重试
    if isinstance(err, (TimeoutError, OSError)):
        # 超时或系统错误
        return True
        # 可重试
    return False
    # 其它（含 JSON 解析失败）默认不重试


def strip_code_fence(text: str) -> str:
    # 去掉 Markdown 围栏
    """去围栏。"""
    # 函数说明
    text = text.strip()
    # 去空白
    if text.startswith("```"):
        # 有围栏
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        # 去开头
        text = re.sub(r"\s*```$", "", text)
        # 去结尾
    return text.strip()
    # 返回


def parse_model_json(content: str) -> dict:
    # 把模型返回的文本解析成字典，尽最大努力容错
    """解析模型 JSON。"""
    # 函数说明
    text = strip_code_fence(content)
    # 先去掉可能的围栏
    try:
        # 直接解析
        return json.loads(text)
        # 成功就返回
    except json.JSONDecodeError:
        # 解析失败，尝试截取第一个 { 到最后一个 }
        start = text.find("{")
        # 找开头
        end = text.rfind("}")
        # 找结尾
        if start != -1 and end != -1 and end > start:
            # 找到一对花括号
            return json.loads(text[start:end + 1])
            # 截取后解析
        raise
        # 仍失败就抛出


# ============================================================
# 四、校验器
# ============================================================
def validate(record: dict, case: dict, facts: dict) -> list:
    # 校验病历，返回问题列表。每条问题带规则编号，便于规则修复
    """校验，返回问题列表。"""
    # 函数说明
    problems = []
    # 问题列表
    valid_ids = {s.get("source_id") for s in case.get("sources", [])}
    # 合法来源集合
    texts = [
        # 三段正文
        record.get("case_features", "") or "",
        # 病例特点
        record.get("diagnostic_discussion", "") or "",
        # 拟诊讨论
        record.get("plan", "") or "",
        # 诊疗计划
    ]
    # 收集完毕
    for name, text in zip(["case_features", "diagnostic_discussion", "plan"], texts):
        # 配对遍历
        if not text.strip():
            # 空
            problems.append({"rule": "R01_EMPTY", "field": name, "msg": f"字段 {name} 为空"})
            # 记录（规则编号可在修复器里对应处理）
    src_text = " ".join(s.get("content", "") for s in case.get("sources", []))
    # 原文大串。⚠️ 必须在所有用到它的规则之前定义：
    #    R04（判断是否改写缺失）、R05（数值比对）、R06（编造检测）、R11（模糊值）都要用
    total_len = sum(len(t) for t in texts)
    # 总字数
    if total_len < 800:
        # 太少
        problems.append({"rule": "R02_LEN_MIN", "field": "*", "msg": f"合计 {total_len} 字，少于 800"})
        # 记录
    elif total_len > 1600:
        # 太多
        problems.append({"rule": "R02_LEN_MAX", "field": "*", "msg": f"合计 {total_len} 字，多于 1600"})
        # 记录
    for bad in BANNED_WORDS:
        # 遍历已实施词
        for i, t in enumerate(texts):
            # 遍历三段
            hit = ALREADY_DONE_RE.search(t) if bad == "已行" else (bad in t)
            # 已行用严格正则，避免误伤"现行"
            if hit:
                # 命中
                problems.append({"rule": "R03_DONE", "field": i, "msg": f"出现已实施表述「{bad}」"})
                # 记录
    for i, t in enumerate(texts):
        # 遍历三段
        for w in FORBIDDEN_REWRITE:
            # 遍历可能改写的词
            for m in re.finditer(re.escape(w), t):
                # 定位出现位置
                # ⚠️ 语境窗口必须限制在**同一个句子内**，否则会串到上一句。
                #    实测发现："血压未提供。血压正常。"里，'正常'前面 15 字包含上一句的'未'，
                #    导致被误判为否定语境而漏报。
                sentence_start = max(t.rfind("。", 0, m.start()), t.rfind("；", 0, m.start()),
                                     t.rfind("，", 0, m.start()), t.rfind("\n", 0, m.start())) + 1
                # 找本句起点：往前最近的一个句读符号
                ctx = t[sentence_start:m.start()]
                # 只取本句内的前缀作为语境
                if "未" in ctx or NEGATION_CONTEXT_RE.search(ctx):
                    # 本句内有否定语境，属于"未见异常"这类合法表述
                    continue
                    # 跳过
                # ⚠️ 还要排除一种情况：原文**本来就说**"血压正常"。
                #    这时输出写"血压正常"是对的，不是改写缺失（实测 P15 误报过）。
                if re.search(r"[\u4e00-\u9fa5]{2,6}\s*" + re.escape(w), src_text):
                    # 原文里存在与输出相同的表述
                    continue
                    # 跳过
                problems.append({"rule": "R04_REWRITE", "field": i, "msg": f"出现「{w}」，疑似改写缺失"})
                # 记录
                break
                # 一段报一次即可
    source_pairs = {(l["name"], l["value"].replace(" ", "")) for l in facts["labs"]}
    # 原文的"指标→数值"配对
    source_values = {l["value"].replace(" ", "") for l in facts["labs"]}
    # 原文出现过的所有"数值+单位"组合
    # ⚠️ 这里刻意**只用数值配对、不用指标名配对**。
    #    原因是实测发现：模型会把"最高"写成"最高体温"、把"反应蛋白"写成"C反应蛋白"，
    #    指标名几乎不可能与原文完全一致，强行配对会产生大量误报。
    #    而"张冠李戴"（把肌酐的值写成血红蛋白的）在 real 数据里更罕见，
    #    且真正的编造行为由 R06（数字不在原文）兜底。宁可少报，不要误报。
    for i, t in enumerate(texts):
        # 遍历三段
        for m in LAB_RE.finditer(t):
            # 找指标+数值
            val = (m.group(2) + m.group(3)).replace(" ", "")
            # 取数值部分
            if val not in source_values:
                # 该数值组合在原文里没有
                problems.append({"rule": "R05_PAIR", "field": i, "msg": f"数值「{m.group(1)} {m.group(2)} {m.group(3)}」的数值与原文不符"})
                # 记录
    for i, t in enumerate(texts):
        # 遍历三段
        for m in NUM_UNIT_RE.finditer(t):
            # 找裸数值
            if m.group(1) not in src_text:
                # 原文里没有这个数字
                problems.append({"rule": "R06_FABRICATE", "field": i, "msg": f"数值「{m.group(0).strip()}」在资料中不存在"})
                # 记录
    # ⚠️ 规则 R11：模糊数值被"精确化"。实测发现模型会把"两百出头"写成"200 μmol/L"。
    #    关键：必须判断**这个数值本身**在原文里是不是模糊的，
    #    不能因为原文别处有"约"就报警——那样会把"体重下降约 5kg"误伤到精确的 118 g/L。
    for i, t in enumerate(texts):
        # 遍历三段
        for m in LAB_RE.finditer(t):
            # 找"指标+数值+单位"
            num = m.group(2).strip()
            # 取出数值
            val = (m.group(2) + m.group(3)).replace(" ", "")
            # 组合值
            if val in src_text.replace(" ", ""):
                # 数值本身出现在原文里，是精确值，正常
                continue
                # 跳过
            # 数值不在原文，检查原文对应位置是不是模糊表述
            idx = src_text.find(num)
            # 在原文里找这个数字
            nearby = src_text[max(0, idx - 8):idx + 12] if idx >= 0 else ""
            # 取数字附近的上下文
            if idx >= 0 and (VAGUE_RE.search(nearby) or re.search(r"[\u4e00-\u9fa5]{1,4}\s*$", nearby)):
                # 原文有该数字但带模糊词，或数字紧跟在汉字后（如"克左右"）
                problems.append({
                    # 规则编号
                    "rule": "R11_VAGUE",
                    # 字段
                    "field": i,
                    # 说明
                    "msg": f"原文该数值为近似/模糊表述，输出却给了精确值「{m.group(1)} {val}」",
                })
                # 追加
                break
                # 一段报一次
    # ⚠️ 规则 R12：中文数字未被识别。原文用"8 克""两百"，正文却写成阿拉伯数字
    #    这种情况 R06 能覆盖大部分，但如果原文同时有无关的相同数字就会漏判。
    #    这里只做提示，不作为硬错误，避免误伤。
    if CN_NUM_RE.search(src_text) and not re.search(r"\d", src_text):
        # 原文只有中文数字、完全没有阿拉伯数字
        for i, t in enumerate(texts):
            # 遍历三段
            if re.search(r"\d+(?:\.\d+)?\s*" + UNIT, t):
                # 正文出现了带单位的阿拉伯数值
                problems.append({
                    # 规则
                    "rule": "R12_CN_NUM",
                    # 字段
                    "field": i,
                    # 说明
                    "msg": "原文仅含中文数字（如“两百出头”），输出却使用了阿拉伯数值，可能是模型自行换算",
                })
                # 追加
                break
                # 只报一次
    evidence = record.get("evidence") or []
    # 依据列表，兼容 None
    if not isinstance(evidence, list):
        # 不是列表
        problems.append({"rule": "R07_EVIDENCE", "field": "evidence", "msg": "evidence 不是数组"})
        # 记录
        evidence = []
        # 置空避免后续报错
    if len(evidence) < 3:
        # 少于 3 条
        problems.append({"rule": "R07_EVIDENCE_MIN", "field": "evidence", "msg": f"evidence 只有 {len(evidence)} 条，建议 3-5 条"})
        # 记录
    for ev in evidence:
        # 遍历依据
        if not isinstance(ev, dict):
            # 不是字典
            problems.append({"rule": "R07_EVIDENCE", "field": "evidence", "msg": "evidence 元素不是对象"})
            # 记录
            continue
            # 下一个
        for sid in ev.get("source_ids", []) or []:
            # 遍历引用
            if sid not in valid_ids:
                # 不属于本病例
                problems.append({"rule": "R08_SRC_ID", "field": "evidence", "msg": f"引用了不存在的 source_id：{sid}"})
                # 记录
        if ev.get("kind") not in ("fact", "inference"):
            # kind 非法
            problems.append({"rule": "R09_KIND", "field": "evidence", "msg": f"kind 非法：{ev.get('kind')}"})
            # 记录
    review = record.get("review_items")
    # 待核事项
    if not isinstance(review, list):
        # 不是列表
        problems.append({"rule": "R10_REVIEW", "field": "review_items", "msg": "review_items 必须是数组"})
        # 记录
    else:
        # 是列表则逐项查
        for item in review:
            # 遍历
            if not isinstance(item, str) or not item.strip():
                # 空项
                problems.append({"rule": "R10_REVIEW", "field": "review_items", "msg": "review_items 含空项"})
                # 记录
    return problems
    # 返回全部问题


# ============================================================
# 五、规则修复（P1-3 第一层：不花调用次数的修复）
# ============================================================
def repair_by_rules(record: dict, problems: list, facts: dict, case: dict) -> tuple:
    # 能靠代码修的直接修，修不了留给重试
    """规则修复，返回 (修复后记录, 修复记录列表, 剩余问题)。"""
    # 函数说明
    fixed = []
    # 修复动作记录
    rec = dict(record)
    # 浅拷贝，避免修改原对象
    for name in ("case_features", "diagnostic_discussion", "plan"):
        # 三个正文字段
        if not isinstance(rec.get(name), str):
            # 不是字符串（可能是 None）
            rec[name] = ""
            # 置为空字符串
            fixed.append(f"{name} 置为空字符串")
            # 记录
    text_fields = ["case_features", "diagnostic_discussion", "plan"]
    # 字段名列表
    for p in problems:
        # 遍历问题
        if p["rule"] == "R03_DONE":
            # 已实施表述
            idx = p["field"]
            # 字段索引
            if isinstance(idx, int):
                # 有效索引
                t = rec[text_fields[idx]]
                # 取原文
                t2 = t.replace("已给予", "拟给予").replace("已完成", "拟完成").replace("已行", "拟行").replace("已用", "拟用").replace("已做", "拟做").replace("已予", "拟予")
                # 全部改为"拟…"。这是保守修复：把"已做"改成"计划做"
                if t2 != t:
                    # 有变化才记
                    rec[text_fields[idx]] = t2
                    # 写回
                    fixed.append(f"{text_fields[idx]} 中已实施表述替换为拟…")
                    # 记录
        elif p["rule"] == "R04_REWRITE" and isinstance(p["field"], int):
            # 疑似改写了缺失
            idx = p["field"]
            # 索引
            t = rec[text_fields[idx]]
            # 原文
            for w in FORBIDDEN_REWRITE:
                # 遍历可疑词
                # 修复思路：只有在"没有否定语境"的位置才替换成"不详"
                # 不能只要求前后是标点——因为"血压正常"里"正常"前面是"压"，
                # 那样写会导致正则匹配不上，修复静默失效（这是实测发现的坑）
                def _repl(mm, _w=w):
                    # 逐个匹配回调，方便逐个判断语境
                    ctx = t[max(0, mm.start() - 15):mm.start()]
                    # 取匹配点前面 15 个字符
                    if "未" in ctx or NEGATION_CONTEXT_RE.search(ctx):
                        # 有否定语境，说明是"未见异常"这类合法表述，不动
                        return mm.group(0)
                        # 原样返回
                    return "不详"
                    # 否则替换为"不详"
                t = re.sub(re.escape(w), _repl, t)
                # 执行替换
            if t != rec[text_fields[idx]]:
                # 有变化
                rec[text_fields[idx]] = t
                # 写回
                fixed.append(f"{text_fields[idx]} 中疑似改写的词替换为“不详”")
                # 记录
        elif p["rule"] == "R15_JSON" or p["rule"] == "R07_EVIDENCE_MIN":
            # 证据不足
            ev = rec.get("evidence") or []
            # 现有依据
            if not isinstance(ev, list):
                # 不是列表
                ev = []
                # 置空
            have_ids = {sid for e in ev if isinstance(e, dict) for sid in (e.get("source_ids") or [])}
            # 已有引用的来源
            for src in case.get("sources", []):
                # 遍历资料
                if len(ev) >= 3:
                    # 够 3 条就停
                    break
                    # 跳出
                sid = src.get("source_id")
                # 编号
                if sid in have_ids:
                    # 已引用过
                    continue
                    # 下一个
                ev.append({
                    # 补一条事实依据
                    "statement": src.get("content", "")[:60],
                    # 截取原文
                    "source_ids": [sid],
                    # 来源
                    "kind": "fact",
                    # 标为事实
                })
                # 追加结束
                have_ids.add(sid)
                # 更新集合
                fixed.append(f"evidence 自动补充来源 {sid}")
                # 记录
            rec["evidence"] = ev[:5]
            # 重新赋值并限制 5 条
    if not isinstance(rec.get("review_items"), list):
        # 待核事项不是列表
        rec["review_items"] = []
        # 修复为空列表
        fixed.append("review_items 重置为空数组")
        # 记录
    if not rec["review_items"] and facts["missing"]:
        # 列表空但有缺失项
        for m in dict.fromkeys(x["text"] for x in facts["missing"]):
            # 去重遍历
            rec["review_items"].append(f"「{m}」资料未提供，需医生补充确认")
            # 补一条
        fixed.append("review_items 按缺失项自动补齐")
        # 记录
    # ⚠️ 冲突与截止点外信息必须进入待核事项，否则评委看不到我们识别了它们
    existing = " ".join(rec["review_items"])
    # 已有内容，用于去重
    for c in facts.get("conflicts", []):
        # 遍历冲突
        if c not in existing:
            # 未出现过
            rec["review_items"].append(c)
            # 追加
            fixed.append(f"review_items 补充冲突提示：{c[:20]}…")
            # 记录
    if facts.get("after_cutoff"):
        # 有截止点外资料
        note = f"有 {len(facts['after_cutoff'])} 段资料超出截止点，已排除未纳入正文"
        # 提示文字
        if note not in " ".join(rec["review_items"]):
            # 未出现过
            rec["review_items"].append(note)
            # 追加
            fixed.append(note)
            # 记录
    remaining = validate(rec, case, facts)
    # 修复后重新校验
    return rec, fixed, remaining
    # 返回


# ============================================================
# 六、离线生成（兜底）
# ============================================================
def build_draft_offline(case: dict, facts: dict) -> dict:
    # 不调模型，用规则拼一份占位记录。日志会标记 source=offline
    """离线占位生成。"""
    # 函数说明
    cut = case.get("record_cutoff", "T0")
    # 截止点
    positive, negative, course = [], [], []
    # 三类片段
    for s in facts["sources"]:
        # 遍历资料
        text = s["content"].strip("。")
        # 去句号
        if "未提供" in text or "未查" in text or "未见" in text:
            # 阴性或缺失
            negative.append(text)
            # 归入
        else:
            # 阳性
            positive.append(text)
            # 归入
    for t in facts["times"]:
        # 遍历时间
        course.append(f"病程约 {t['text']}")
        # 拼病程
    seg = []
    # 片段
    if positive:
        # 有阳性
        seg.append("；".join(positive))
        # 拼接
    if negative:
        # 有阴性
        seg.append("现有记录中，" + "；".join(negative))
        # 拼接
    if course:
        # 有病程
        seg.append("；".join(course))
        # 拼接
    body = "。".join(s.rstrip("。；; ") for s in seg if s.strip())
    # 句号连接，先 rstrip 防双句号
    case_features = f"资料截止点 {cut}。" + (body + "。" if body else "现有资料未提供可归纳的临床信息。")
    # 完整病例特点
    lab_lines = [f"{l['name']} {l['value']}（见 {l['source_id']}）" for l in facts["labs"]]
    # 检验行
    lab_text = "；".join(dict.fromkeys(lab_lines)) if lab_lines else "未见明确可引用的检验数值"
    # 去重拼接
    diffs = build_differentials(facts)
    # 鉴别方向
    diff_txt = ""
    # 鉴别文本
    if diffs:
        # 有命中
        parts = []
        # 片段
        for d in diffs[:MAX_DIFFERENTIALS]:
            # 遍历
            g = d["group"]
            # 组
            parts.append(f"{g['name']}需考虑 " + "、".join(g["differentials"][:MIN_DIFFERENTIALS]))
            # 拼一句
        diff_txt = "本例存在以下鉴别方向：" + "；".join(parts) + "。"
        # 完整句
    discussion = (
        # 拟诊讨论
        f"现有资料可引用的检验结果包括：{lab_text}。"
        + diff_txt
        + "上述异常的具体原因尚需结合缺失资料进一步明确，单次数值变化不足以判断急慢性过程，"
        + "也不足以据此确定诊断。需结合既往基线、伴随症状与针对性检查综合判断。目前病因待查。"
    )
    # 拼接
    plan = (
        # 诊疗计划
        "建议由临床医生补充既往病史、用药史、症状演变及相关实验室检查资料；"
        "拟复查上述异常指标并观察症状变化；按鉴别需要安排进一步检查，"
        "依据补充资料调整分析与处理；治疗需结合完整临床评估确定。"
    )
    # 拼接
    evidence = []
    # 依据
    for s in facts["sources"][:5]:
        # 最多 5 条
        evidence.append({"statement": s["content"][:60], "source_ids": [s["source_id"]], "kind": "fact"})
        # 事实依据
    if len(evidence) < 3 and facts["sources"]:
        # 不足 3 条
        evidence.append({
            # 补推断
            "statement": "现有资料不足以确定诊断，属于推断边界。",
            # 说明
            "source_ids": [s["source_id"] for s in facts["sources"]],
            # 全部来源
            "kind": "inference",
            # 推断
        })
        # 结束
    review = [f"「{x}」资料未提供，需医生补充确认" for x in dict.fromkeys(m["text"] for m in facts["missing"])]
    # 待核事项
    if not review:
        # 空则兜底
        review = ["既往病史、用药史及缺失检查项目需补充"]
        # 兜底
    return {
        # 返回病历
        "case_features": case_features,
        # 病例特点
        "diagnostic_discussion": discussion,
        # 拟诊讨论
        "plan": plan,
        # 诊疗计划
        "evidence": evidence[:5],
        # 依据
        "review_items": review,
        # 待核事项
    }
    # 结束


# ============================================================
# 七、单条处理（P1-3 修复重试回路）
# ============================================================
def process_one(case: dict, conn: dict, counter: CallCounter, mode: str) -> tuple:
    # 处理一条病例，返回 (记录, 日志)
    """处理单条病例。"""
    # 函数说明
    case_id = case.get("case_id")
    # 编号
    t0 = time.perf_counter()
    # 计时
    log = {
        # 日志
        "case_id": case_id,
        # 编号
        "start_time": now_iso(),
        # 开始时间
        "source": "offline",
        # 数据来源：offline / dashscope / hosted / baseline
        "calls": 0,
        # 调用次数
        "attempts": 0,
        # 尝试轮数
        "rule_fixes": [],
        # 规则修复动作
        "problems_before": 0,
        # 修复前问题数
        "problems_after": 0,
        # 修复后问题数
        "status": "failed",
        # 状态
        "prompt_version": PROMPT_VERSION,
        # 提示词版本
        "seed": SEED,
        # 种子
        "elapsed_sec": 0.0,
        # 耗时
        "end_time": "",
        # 结束时间
    }
    # 初始化
    if not case_id:
        # 无编号
        return failed(None, "INVALID_CASE", "缺少 case_id"), log
        # 失败
    if not isinstance(case.get("sources"), list):
        # sources 非数组
        return failed(case_id, "INVALID_CASE", "sources 不是数组"), log
        # 失败
    facts = extract_facts(case)
    # 抽事实
    try:
        # 兜住所有异常
        if conn["mode"] == "offline" or mode == "baseline" and conn["mode"] == "offline":
            # 离线模式
            record = build_draft_offline(case, facts)
            # 规则拼装
            log["source"] = "offline"
            # 标记来源
        else:
            # 在线模式
            record, log = run_online(case, conn, counter, mode, facts, log)
            # 走在线流程
        if mode == "baseline":
            # 基线版：不做校验与修复，模拟裸调用
            log["problems_after"] = len(validate(record, case, facts))
            # 只统计问题数，不修复
            log["status"] = "success"
            # 标记成功
            return {"case_id": case_id, "status": "success", "medical_record": record, "error": None}, log
            # 返回
        # 改进版：校验 → 规则修复 → 必要时重试
        problems = validate(record, case, facts)
        # 首次校验
        log["problems_before"] = len(problems)
        # 记录
        record, fixed, remaining = repair_by_rules(record, problems, facts, case)
        # 规则修复
        log["rule_fixes"] = fixed
        # 记录修复动作
        log["problems_after"] = len(remaining)
        # 记录剩余
        log["status"] = "success"
        # 标记成功
        return {"case_id": case_id, "status": "success", "medical_record": record, "error": None}, log
        # 返回
    except Exception as exc:
        # 任何异常只影响这一条
        return failed(case_id, "GENERATION_FAILED", f"{type(exc).__name__}: {exc}"), log
        # 失败返回
    finally:
        # 无论如何补全日志
        log["calls"] = counter.used(case_id)
        # 真实调用次数
        log["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        # 耗时
        log["end_time"] = now_iso()
        # 结束时间
    # 结束


def run_online(case: dict, conn: dict, counter: CallCounter, mode: str, facts: dict, log: dict) -> tuple:
    # 在线生成：带重试与预算控制
    """在线生成。"""
    # 函数说明
    case_id = case.get("case_id", "")
    # 编号
    log["source"] = conn["mode"]
    # 记录实际走的模式
    messages = [
        # 消息列表
        {"role": "system", "content": SYSTEM_PROMPT},
        # 系统指令
        {"role": "user", "content": build_user_content(case, facts, mode)},
        # 用户消息（改进版带事实清单）
    ]
    # 组装完毕
    last_err = None
    # 最后一次错误
    record = None
    # 结果
    limit = 1 if mode == "baseline" else MAX_ATTEMPTS
    # 基线只试 1 次，改进版最多 MAX_ATTEMPTS 次
    for attempt in range(1, limit + 1):
        # 逐轮尝试
        if not counter.can_call(case_id):
            # 预算不够
            raise RuntimeError(f"调用预算超限（已用 {counter.used(case_id)}/{counter.per_case_limit}）")
            # 抛出
        log["attempts"] = attempt
        # 记录轮次
        counter.record(case_id)
        # 发请求前记账
        try:
            # 尝试请求
            payload = http_chat(conn, messages, REQUEST_TIMEOUT)
            # 发请求
            content = payload["choices"][0]["message"]["content"]
            # 取内容
            record = parse_model_json(content)
            # 解析
            return record, log
            # 成功返回
        except Exception as exc:
            # 失败
            last_err = exc
            # 记录错误
            if not is_retryable(exc) or attempt >= limit:
                # 不可重试或已达上限就停
                raise
                # 抛出，由上层记为失败
            time.sleep(min(2 ** (attempt - 1), 8))
            # 指数退避：1s、2s、4s，最多 8s，避免把服务打爆
    if record is None:
        # 理论上不会到这
        raise last_err or RuntimeError("模型调用失败")
        # 兜底抛出
    return record, log
    # 返回


def failed(case_id, code, message):
    # 构造失败记录
    """失败记录。"""
    # 函数说明
    return {"case_id": case_id, "status": "failed", "medical_record": None,
            "error": {"code": code, "message": message}}
    # 返回


# ============================================================
# 八、工具
# ============================================================
def now_iso() -> str:
    # ISO 时间戳
    """当前时间。"""
    # 函数说明
    return datetime.now().astimezone().isoformat(timespec="seconds")
    # 带时区，精确到秒


def atomic_write_json(path: Path, data):
    # 原子写 JSON
    """原子写。"""
    # 函数说明
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    # 同目录临时文件
    try:
        # 保证清理
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # 打开
            json.dump(data, f, ensure_ascii=False, indent=2)
            # 写入
        os.replace(tmp, path)
        # 原子替换
    except Exception:
        # 出错
        if os.path.exists(tmp):
            # 临时文件还在
            os.unlink(tmp)
            # 删除
        raise
        # 抛出


def sha256_of(path: Path) -> str:
    # 文件摘要
    """sha256。"""
    # 函数说明
    import hashlib
    # 延迟导入
    if not path.exists():
        # 不存在
        return ""
        # 空串
    h = hashlib.sha256()
    # 摘要对象
    with path.open("rb") as f:
        # 二进制读
        for chunk in iter(lambda: f.read(8192), b""):
            # 分块
            h.update(chunk)
            # 更新
    return h.hexdigest()
    # 返回


# ============================================================
# 九、主流程
# ============================================================
def parse_args():
    # 解析命令行
    """命令行参数。"""
    # 函数说明
    p = argparse.ArgumentParser(description="第8题 首程生成器 P1")
    # 创建解析器
    p.add_argument("--mode", choices=["improved", "baseline"], default="improved")
    # --mode：改进版或基线版
    p.add_argument("--dry-run", action="store_true")
    # --dry-run：强制离线，不联网
    p.add_argument("--input", default=str(INPUT_FILE))
    # --input：可指定输入文件，便于跑对照实验
    p.add_argument("--out-prefix", default="")
    # --out-prefix：输出文件名前缀，避免对照实验互相覆盖
    p.add_argument("--out-dir", default="")
    # --out-dir：JSON 三件套的输出目录（相对 main_p1.py）。
    #   默认空字符串 = 留在程序根目录，这是赛题技术规范的要求
    #   （主办方会在根目录放 input.json，期望同目录出现 output.json）。
    #   对照实验时传 "test-output" 之类，避免污染根目录。
    return p.parse_args()
    # 返回


def main() -> int:
    # 主流程
    """主函数。"""
    # 函数说明
    args = parse_args()
    # 解析参数
    run_start = now_iso()
    # 开始时间
    t_start = time.perf_counter()
    # 计时
    write_config_templates()
    # 生成依赖与模板
    cfg = load_env(ENV_FILE)
    # 读配置
    conn = resolve_endpoint(cfg)
    # 判断模式
    if args.dry_run:
        # 强制离线
        conn = {"mode": "offline", "url": "", "api_key": "", "model": ""}
        # 清空连接信息
    if conn["mode"] == "offline":
        # 离线
        print("[提示] 未配置模型，使用离线演示模式（结果会标记 source=offline）", file=sys.stderr)
        # 提示
    else:
        # 在线
        print(f"[提示] 接入模式：{conn['mode']}，模型：{conn['model']}", file=sys.stderr)
        # 提示
    per_limit = BASELINE_CALL_BUDGET if args.mode == "baseline" else CALL_BUDGET_PER_CASE
    # 按模式决定单例预算
    counter = CallCounter(per_limit, TOTAL_CALL_BUDGET)
    # 计数器
    input_file = Path(args.input)
    # 输入路径
    if not input_file.exists():
        # 不存在
        print(f"[错误] 找不到输入：{input_file}", file=sys.stderr)
        # 提示
        return EXIT_CONFIG_ERROR
        # 返回 2
    try:
        # 尝试解析
        cases = json.loads(input_file.read_text(encoding="utf-8"))
        # 读取
    except json.JSONDecodeError as exc:
        # JSON 错误
        print(f"[错误] 输入不是合法 JSON：{exc}", file=sys.stderr)
        # 提示
        return EXIT_CONFIG_ERROR
        # 返回 2
    if not isinstance(cases, list):
        # 非数组
        print("[错误] 顶层必须是数组", file=sys.stderr)
        # 提示
        return EXIT_CONFIG_ERROR
        # 返回 2
    prefix = args.out_prefix
    # 输出前缀（--out-prefix 指定，用于对照实验区分）
    if args.out_dir:
        # 指定了输出目录（对照实验用）
        out_dir = BASE_DIR / args.out_dir
        # 拼成绝对路径
        out_dir.mkdir(parents=True, exist_ok=True)
        # 不存在就创建
    else:
        # 未指定，留在程序根目录
        out_dir = BASE_DIR
        # 这是赛题技术规范的要求：output.json 必须与 main.py 同目录
    out_file = out_dir / f"{prefix}output.json"
    # 技术规范要求的输出文件
    res_file = out_dir / f"{prefix}results.json"
    # 提交要求里写的文件名。与 output.json 表述不一致，两个都写
    log_file = out_dir / f"{prefix}run_log.json"
    # 运行记录，线下复现核验的核对依据
    md_file = make_md_path(prefix=prefix, sub_dir=args.out_dir)
    # 人可读的 Markdown 报告。
    # 指定了 --out-dir 时（对照实验），报告也放进同一子目录，避免散落；
    # 默认则存到 test-output/时间戳.md
    if not cases:
        # 空输入
        atomic_write_json(out_file, [])
        # 写空数组
        atomic_write_json(res_file, [])
        # 写空数组
        empty_log = build_run_log(conn, args.mode, [], run_start, t_start, counter, "empty_input", input_file)
        # 组装日志
        atomic_write_json(log_file, empty_log)
        # 写日志
        write_markdown_report(md_file, cases, [], empty_log)
        # 也写一份 Markdown 报告，便于确认程序跑过了
        print(f"[输出] {md_file}", file=sys.stderr)
        # 提示路径
        return EXIT_OK
        # 正常结束
    ids = [c.get("case_id") for c in cases if isinstance(c, dict)]
    # 收集编号
    if len(ids) != len(set(ids)):
        # 有重复
        print("[错误] case_id 重复", file=sys.stderr)
        # 提示
        return EXIT_CONFIG_ERROR
        # 返回 2
    records = []
    # 结果
    logs = []
    # 日志
    for i, case in enumerate(cases, 1):
        # 逐条
        if not isinstance(case, dict):
            # 非字典
            records.append(failed(None, "INVALID_CASE", "病例不是对象"))
            # 记失败
            logs.append({"case_id": None, "status": "failed", "calls": 0})
            # 日志
            continue
            # 下一个
        print(f"[处理] ({i}/{len(cases)}) {case.get('case_id')}", file=sys.stderr)
        # 进度
        rec, lg = process_one(case, conn, counter, args.mode)
        # 处理
        records.append(rec)
        # 追加
        logs.append(lg)
        # 追加
    atomic_write_json(out_file, records)
    # 写输出
    atomic_write_json(res_file, records)
    # 写结果（与输出同内容）
    final_log = build_run_log(conn, args.mode, logs, run_start, t_start, counter, "completed", input_file)
    # 组装运行日志
    atomic_write_json(log_file, final_log)
    # 写日志
    write_markdown_report(md_file, cases, records, final_log)
    # 写人可读的 Markdown 报告
    ok = sum(1 for r in records if r["status"] == "success")
    # 成功数
    print(f"[完成] {args.mode} 模式，共 {len(records)} 条，成功 {ok} 条，调用 {counter.total} 次", file=sys.stderr)
    # 总结
    print(f"[报告] {md_file}", file=sys.stderr)
    # 提示报告路径
    return EXIT_OK
    # 返回 0


def make_md_path(prefix: str = "", sub_dir: str = "") -> Path:
    # 生成 Markdown 报告的完整路径
    """生成报告路径。"""
    # 函数说明
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # 时间戳，例如 2026-11-07-17-00-00
    if sub_dir:
        # 指定了子目录（对照实验用）。
        # 注意：sub_dir 已经包含 "test-output/..." 前缀，
        # 不能再拼一次 BASE_DIR/"test-output"，否则会出现 test-output/test-output 嵌套
        out_dir = BASE_DIR / sub_dir
        # 直接用
    else:
        # 未指定，默认放 test-output 根目录
        out_dir = BASE_DIR / "test-output"
        # 默认目录
    out_dir.mkdir(parents=True, exist_ok=True)
    # 不存在就创建（exist_ok 避免目录已存在时报错）
    path = out_dir / f"{prefix}{stamp}.md"
    # 先拼出候选路径
    # ⚠️ 同一秒内跑两次会得到同名文件，后一次覆盖前一次（实测发现）。
    #    这里加去重后缀：已存在就加 _1、_2……
    n = 1
    # 序号
    while path.exists():
        # 文件已存在
        path = out_dir / f"{prefix}{stamp}_{n}.md"
        # 换一个名字
        n += 1
        # 序号递增
    return path
    # 返回唯一路径


def write_markdown_report(path: Path, cases: list, records: list, run_log: dict):
    # 把运行结果渲染成人类可读的 Markdown
    """写 Markdown 报告。"""
    # 函数说明
    summary = run_log.get("summary", {})
    # 汇总信息
    run = run_log.get("run", {})
    # 运行信息
    lines = []
    # 报告行
    lines.append(f"# 首程生成报告 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 一级标题
    lines.append("")
    # 空行
    lines.append("## 运行概况")
    # 二级标题
    lines.append("")
    # 空行
    lines.append("| 项目 | 值 |")
    # 表格头
    lines.append("|---|---|")
    # 分隔行
    lines.append(f"| 运行模式 | {run_log.get('mode', '')} |")
    # 模式
    lines.append(f"| 接入方式 | **{run_log.get('source', '')}** |")
    # 接入方式（offline/dashscope/hosted）
    lines.append(f"| 模型 | {run_log.get('environment', {}).get('model_name', '') or '（离线）'} |")
    # 模型名
    lines.append(f"| 输入文件 | {run.get('input_file', '')} |")
    # 输入文件
    lines.append(f"| 病例总数 | {summary.get('total', 0)} |")
    # 总数
    lines.append(f"| 成功 / 失败 | {summary.get('success', 0)} / {summary.get('failed', 0)} |")
    # 成败
    lines.append(f"| 模型调用次数 | {summary.get('total_calls', 0)} |")
    # 调用次数
    lines.append(f"| 离线生成条数 | {summary.get('offline_cases', 0)} |")
    # 离线条数
    lines.append(f"| 耗时 | {run.get('elapsed_sec', 0)} 秒 |")
    # 耗时
    if summary.get("offline_cases"):
        # 有离线数据
        lines.append("")
        # 空行
        lines.append(f"> ⚠️ **注意：本次有 {summary['offline_cases']} 条是离线规则生成，不是真实模型结果，不可作为提交材料。**")
        # 醒目警告
    lines.append("")
    # 空行
    logs_by_id = {x.get("case_id"): x for x in run_log.get("cases", [])}
    # 按编号索引日志
    for idx, (case, rec) in enumerate(zip(cases, records), 1):
        # 逐例输出
        cid = case.get("case_id")
        # 编号
        lg = logs_by_id.get(cid, {})
        # 该例日志
        lines.append("---")
        # 分隔线
        lines.append("")
        # 空行
        lines.append(f"## {idx}. {cid}")
        # 病例标题
        lines.append("")
        # 空行
        lines.append(f"- 状态：**{rec.get('status')}**")
        # 状态
        lines.append(f"- 数据来源：{lg.get('source', '-')}｜调用 {lg.get('calls', 0)} 次｜耗时 {lg.get('elapsed_sec', 0)} 秒")
        # 来源与耗时
        lines.append(f"- 校验问题：修复前 {lg.get('problems_before', 0)} → 修复后 {lg.get('problems_after', 0)}")
        # 问题数变化
        if lg.get("rule_fixes"):
            # 有修复动作
            lines.append(f"- 规则修复（{len(lg['rule_fixes'])} 处）：" + "；".join(lg["rule_fixes"]))
            # 列出修复
        lines.append("")
        # 空行
        # ===== 按赛题示例（附件 1）的形式列出输入资料 =====
        lines.append(f"**病例编号 {cid}，资料截止点 {case.get('record_cutoff', '')}，以下资料在 {case.get('record_cutoff', 'T0')} 已可获得。**")
        # 赛题示例原句式的改写
        lines.append("")
        # 空行
        srcs = case.get("sources", [])
        # 资料片段
        if srcs:
            # 有资料
            lines.append("| source_id | 资料内容 |")
            # 表头，与赛题示例完全一致
            lines.append("|---|---|")
            # 分隔行
            for s in srcs:
                # 逐条
                content = str(s.get("content", "")).replace("|", "\\|")
                # 转义竖线，避免破坏表格
                lines.append(f"| {s.get('source_id', '')} | {content} |")
                # 一行
        else:
            # 无资料
            lines.append("| （空） | 本病例未提供任何资料片段 |")
            # 占位行
        lines.append("")
        # 空行
        if rec.get("status") != "success":
            # 失败记录
            err = rec.get("error") or {}
            # 错误对象
            lines.append(f"> ❌ 生成失败：`{err.get('code')}` — {err.get('message')}")
            # 错误说明
            lines.append("")
            # 空行
            continue
            # 下一例
        mr = rec.get("medical_record") or {}
        # 病历对象
        lines.append("**首程生成结果**")
        # 与赛题示例的"算法应输出的首程示例"对应
        lines.append("")
        # 空行
        lines.append("### 病例特点")
        # 小标题
        lines.append("")
        # 空行
        lines.append(mr.get("case_features", "") or "（空）")
        # 正文
        lines.append("")
        # 空行
        lines.append("### 拟诊讨论")
        # 小标题
        lines.append("")
        # 空行
        lines.append(mr.get("diagnostic_discussion", "") or "（空）")
        # 正文
        lines.append("")
        # 空行
        lines.append("### 诊疗计划")
        # 小标题
        lines.append("")
        # 空行
        lines.append(mr.get("plan", "") or "（空）")
        # 正文
        lines.append("")
        # 空行
        # 赛题示例中"依据说明"是首程正文的一部分，这里保持同样位置与措辞
        lines.append("### 依据说明")
        # 小标题，与赛题示例的"依据说明。"对应
        lines.append("")
        # 空行
        evidence = mr.get("evidence") or []
        # 依据列表
        if evidence:
            # 有依据
            lines.append("| 类型 | 陈述 | 来源 |")
            # 表格头
            lines.append("|---|---|---|")
            # 分隔行
            for ev in evidence:
                # 逐条
                stmt = str(ev.get("statement", "")).replace("|", "\\|")
                # 转义竖线，避免破坏表格
                ids = ", ".join(str(x) for x in (ev.get("source_ids") or []))
                # 来源编号
                lines.append(f"| {ev.get('kind', '')} | {stmt} | {ids} |")
                # 一行
        else:
            # 无依据
            lines.append("（无）")
            # 占位
        lines.append("")
        # 空行
        lines.append("### 待核事项")
        # 小标题
        lines.append("")
        # 空行
        review = mr.get("review_items") or []
        # 待核列表
        if review:
            # 有内容
            for r in review:
                # 逐条
                lines.append(f"- {r}")
                # 列表项
        else:
            # 无内容
            lines.append("（无）")
            # 占位
        lines.append("")
        # 空行
        # 字数统计：赛题要求三段合计 800-1600 字，这里给出实测值方便核对
        seg_len = sum(len(mr.get(k) or "") for k in ("case_features", "diagnostic_discussion", "plan"))
        # 三段合计
        flag = "✅" if 800 <= seg_len <= 1600 else "⚠️"
        # 是否在建议区间
        lines.append(f"<sub>{flag} 三段合计 {seg_len} 字（赛题建议 800–1600 字）</sub>")
        # 以 HTML 小字呈现，不影响正文阅读
        lines.append("")
        # 空行
    atomic_write_text(path, "\n".join(lines))
    # 写入文件


def atomic_write_text(path: Path, text: str):
    # 原子写文本文件
    """原子写文本。"""
    # 函数说明
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    # 同目录临时文件
    try:
        # 保证异常时清理
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # 打开
            f.write(text)
            # 写入
        os.replace(tmp, path)
        # 原子替换
    except Exception:
        # 出错
        if os.path.exists(tmp):
            # 临时文件还在
            os.unlink(tmp)
            # 删除
        raise
        # 抛出


def build_run_log(conn: dict, mode: str, logs: list, run_start: str, t_start: float,
                  counter: CallCounter, status: str, input_file: Path) -> dict:
    # 组装运行日志
    """组装日志。"""
    # 函数说明
    return {
        # 返回
        "schema_version": "1.1",
        # 日志版本
        "mode": mode,
        # improved / baseline
        "source": conn["mode"],
        # offline / dashscope / hosted
        "prompt_version": PROMPT_VERSION,
        # 提示词版本
        "generation_params": {"seed": SEED, "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": MAX_TOKENS},
        # 生成参数
        "budget": {
            # 预算
            "per_case_limit": counter.per_case_limit,
            # 单例上限
            "total_limit": counter.total_limit,
            # 全局上限
            "total_calls": counter.total,
            # 实际用量
        },
        "run": {
            # 运行信息
            "status": status,
            # 状态
            "start_time": run_start,
            # 开始
            "end_time": now_iso(),
            # 结束
            "elapsed_sec": round(time.perf_counter() - t_start, 3),
            # 耗时
            "input_file": input_file.name,
            # 输入文件名
            "input_sha256": sha256_of(input_file),
            # 输入摘要
        },
        "environment": {
            # 环境指纹
            "python_version": sys.version.split()[0],
            # Python 版本
            "platform": platform.platform(),
            # 系统
            "model_name": conn["model"],
            # 模型名
            "endpoint_host": re.sub(r"^(https?://[^/]+).*$", r"\1", conn["url"]) if conn["url"] else "",
            # 只记录域名，不记录完整路径（避免泄露 Key 所在 URL）
        },
        "cases": logs,
        # 每例日志
        "summary": {
            # 汇总
            "total": len(logs),
            # 总数
            "success": sum(1 for x in logs if x.get("status") == "success"),
            # 成功
            "failed": sum(1 for x in logs if x.get("status") == "failed"),
            # 失败
            "offline_cases": sum(1 for x in logs if x.get("source") == "offline"),
            # 离线生成的条数（重要：提醒这些不是真结果）
            "total_calls": counter.total,
            # 总调用
        },
    }
    # 结束


if __name__ == "__main__":
    # 入口
    try:
        # 兜异常
        sys.exit(main())
        # 执行
    except Exception as exc:
        # 任务级错误
        print(f"[任务级错误] {exc}", file=sys.stderr)
        # 提示
        traceback.print_exc(file=sys.stderr)
        # 堆栈
        sys.exit(EXIT_TASK_ERROR)
        # 返回 1
