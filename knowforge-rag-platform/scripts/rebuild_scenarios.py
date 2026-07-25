# -*- coding: utf-8 -*-
# ============================================================================
# 批量重建多个业务场景的知识库版本
# ============================================================================
# 默认用于初始化或重建全部 8 个冻结业务场景：
#   python scripts/rebuild_scenarios.py --reset-collections
#
# 也可以指定场景：
#   python scripts/rebuild_scenarios.py --scenarios enterprise_knowledge,equipment_ops --reset-collections
#
# 脚本通过子进程逐个调用 `scripts/rebuild_kb_version.py`，保证单场景已有的
# FAQ/文档入库、质量门禁、版本激活和 Milvus schema reset 逻辑完全复用。
#
# 8 个冻结场景：
#   compliance_qa           — 合规问答
#   cross_border_risk        — 跨境风险
#   engineering_project_qa   — 工程项目问答
#   enterprise_knowledge     — 企业知识库（默认场景）
#   equipment_ops            — 设备运维
#   insurance_claims         — 保险理赔
#   saas_support             — SaaS 支持
#   tender_contract_risk     — 招投标合同风险
#
# 用法示例：
#   # 重建全部 8 个场景（含 Collection 重置）
#   python scripts\rebuild_scenarios.py --reset-collections
#
#   # 只重建两个 staged 版本，不激活也不跑质量门禁
#   python scripts\rebuild_scenarios.py --scenarios enterprise_knowledge,equipment_ops --no-activate --no-quality-gate
#
#   # 预演模式（只打印命令不执行）
#   python scripts\rebuild_scenarios.py --dry-run
#
#   # 遇到失败继续执行后续场景
#   python scripts\rebuild_scenarios.py --continue-on-failure
# ============================================================================

"""按冻结场景清单批量构建并激活知识库版本。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# argparse: 命令行参数解析
import argparse

# subprocess: 子进程管理（通过 subprocess.run 调用 rebuild_kb_version.py）
import subprocess

# sys: 系统功能（sys.executable 获取当前 Python 解释器路径，sys.exit 退出码）
import sys

# time: 时间功能（time.perf_counter 高精度计时）
import time

# dataclasses: 数据类定义（ScenarioRunResult）
from dataclasses import dataclass

# pathlib.Path: 文件路径操作
from pathlib import Path

# ── 常量定义 ──

# PROJECT_ROOT: 项目根目录绝对路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# DEFAULT_REBUILD_SCRIPT: 单场景重建脚本的默认路径
DEFAULT_REBUILD_SCRIPT = PROJECT_ROOT / "scripts" / "rebuild_kb_version.py"

# DEFAULT_ALL_SCENARIOS: 全部 8 个冻结业务场景 ID
# 排序按注册时的声明顺序，保证每次执行顺序一致
DEFAULT_ALL_SCENARIOS = (
    "compliance_qa",
    "cross_border_risk",
    "engineering_project_qa",
    "enterprise_knowledge",
    "equipment_ops",
    "insurance_claims",
    "saas_support",
    "tender_contract_risk"
)

@dataclass(frozen=True)
class ScenarioRunResult:
    """单个场景重建结果。

    Attributes:
        scenario_id: 场景标识
        ok: 是否成功（returncode == 0）
        elapsed_seconds: 执行耗时（秒）
        returncode: 子进程退出码

    调用顺序：命令行入口 -> ScenarioRunResult。
    """
    scenario_id: str
    ok: bool
    elapsed_seconds: float
    returncode: int


def parse_scenarios(value: str) -> list[str]:
    """解析逗号分隔场景列表。

    示例：
      "enterprise_knowledge,equipment_ops" → ["enterprise_knowledge", "equipment_ops"]
      "  a , b , c  " → ["a", "b", "c"]

    调用顺序：命令行入口 -> parse_scenarios()。
    """
    return [item.strip() for item in value.split(",") if item.strip()]


def build_command(args: argparse.Namespace, scenario_id: str) -> list[str]:
    """构造单场景 rebuild 命令。

    将命令行参数映射为 rebuild_kb_version.py 的标准参数列表。
    每个场景使用 --new-version --force 确保全新构建。

    Args:
        args: 命令行参数命名空间
        scenario_id: 目标场景 ID

    Returns:
        命令参数列表，第一项是 Python 解释器路径

    调用顺序：命令行入口 -> build_command()。
    """
    command = [
        sys.executable,                          # 当前 Python 解释器
        str(Path(args.rebuild_script)),          # rebuild_kb_version.py 的绝对路径
        "--scenario", scenario_id,               # 目标场景
        "--new-version",                         # 创建新版本
        "--force",                               # 强制执行（即使 fingerprint 未变）
    ]
    # 可选参数：只有显式传入时才追加
    if args.reset_collections:
        command.append("--reset-collections")
    if args.quality_gate:
        command.append("--quality-gate")
    if args.activate:
        command.append("--activate")
    if args.description:
        command.extend(["--description", args.description])
    if args.tenant_id:
        command.extend(["--tenant-id", args.tenant_id])
    if args.dataset_id:
        command.extend(["--dataset-id", args.dataset_id])
    if args.visibility:
        command.extend(["--visibility", args.visibility])
    for role in args.allowed_role or []:
        command.extend(["--allowed-role", role])
    return command


def run_one(args: argparse.Namespace, scenario_id: str) -> ScenarioRunResult:
    """执行单个场景重建。

    通过 subprocess.run 调用 rebuild_kb_version.py，阻塞等待完成。
    打印场景分隔线和命令用于日志追踪。

    Args:
        args: 命令行参数命名空间
        scenario_id: 目标场景 ID

    Returns:
        ScenarioRunResult 实例

    调用顺序：命令行入口 -> run_one()。
    """
    command = build_command(args, scenario_id)
    print("\n" + "=" * 88)
    print(f"Rebuilding scenario: {scenario_id}")
    print("Command:", " ".join(command))
    print("=" * 88)

    started = time.perf_counter()

    # ── dry-run 模式：只打印命令不执行 ──
    if args.dry_run:
        return ScenarioRunResult(
            scenario_id=scenario_id,
            ok=True,
            elapsed_seconds=0.0,
            returncode=0,
        )

    # ── 真实执行 ──
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - started
    return ScenarioRunResult(
        scenario_id=scenario_id,
        ok=completed.returncode == 0,
        elapsed_seconds=elapsed,
        returncode=completed.returncode,
    )


def main() -> None:
    """解析命令行参数 → 逐场景执行重建 → 汇总输出。

    执行流程：
      1. 解析命令行参数
      2. 解析目标场景列表
      3. 逐个场景调用 run_one()：
         - 每个场景独立子进程执行
         - 某场景失败后：
           * --continue-on-failure → 继续下一个场景
           * 否则 → 中断执行
      4. 打印汇总表格
      5. 有失败场景时返回非零退出码
    """
    parser = argparse.ArgumentParser(description="Batch rebuild scenario knowledge base versions.")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_ALL_SCENARIOS),
        help="Comma-separated scenario ids. Defaults to all 8 frozen business scenarios.",
    )
    parser.add_argument(
        "--reset-collections", action="store_true",
        help="Drop each scenario FAQ/Doc Milvus collection before rebuild.",
    )
    parser.add_argument(
        "--no-quality-gate", dest="quality_gate", action="store_false",
        help="Disable ingestion quality gate. Not recommended for production.",
    )
    parser.add_argument(
        "--no-activate", dest="activate", action="store_false",
        help="Only create staged versions; do not activate them.",
    )
    parser.add_argument("--description", default="batch rebuild scenarios", help="Version description.")
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--visibility", default=None)
    parser.add_argument("--allowed-role", action="append", default=None)
    parser.add_argument(
        "--rebuild-script", default=str(DEFAULT_REBUILD_SCRIPT),
        help="Path to rebuild_kb_version.py. When running through a host volume mounted "
             "at /work inside the api container, use --rebuild-script /app/scripts/rebuild_kb_version.py.",
    )
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="Continue after a scenario fails.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")

    # set_defaults 确保 --no-quality-gate / --no-activate 未传时默认启用
    parser.set_defaults(quality_gate=True, activate=True)
    args = parser.parse_args()
    if args.activate and not args.quality_gate:
        parser.error("--no-quality-gate can only be used together with --no-activate.")

    # ── 解析场景列表 ──
    scenarios = parse_scenarios(args.scenarios)
    if not scenarios:
        parser.error("--scenarios is empty.")

    # ── 逐场景执行 ──
    results: list[ScenarioRunResult] = []
    for scenario_id in scenarios:
        result = run_one(args, scenario_id)
        results.append(result)
        if not result.ok and not args.continue_on_failure:
            break  # 中断执行，不再继续后续场景

    # ── 打印汇总表格 ──
    print("\nBatch rebuild summary")
    print("-" * 88)
    for result in results:
        status = "OK" if result.ok else f"FAILED({result.returncode})"
        print(f"{result.scenario_id:28s} {status:12s} {result.elapsed_seconds:8.2f}s")

    failed = [item for item in results if not item.ok]
    if failed:
        print("\nFailed scenarios:", ", ".join(item.scenario_id for item in failed))
        sys.exit(1)


if __name__ == "__main__":
    # 当脚本直接运行时，__name__ == "__main__"
    main()
