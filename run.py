"""
FedProRef 实验入口
修改此文件中的参数即可控制整个实验。
所有参数与 config.py 完全对应。
默认在 seed=42,43,44 上各运行一次，并汇总 mean/std。
"""
import os
import sys

# 防止 MKL/Intel 库冲突导致的中断
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import shutil
import random
import subprocess
import glob
import re
import numpy as np

# 清理 __pycache__ 缓存
def clean_pycache():
    for root, dirs, files in os.walk('.'):
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            shutil.rmtree(pycache_path, ignore_errors=True)

clean_pycache()

# 清理 checkpoints 文件夹
if os.path.exists("./checkpoints"):
    shutil.rmtree("./checkpoints")
    os.makedirs("./checkpoints")

# ══════════════════════════════════════════════════════════════════════
#  命令行参数解析
# ══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="FedProRef 实验入口")
parser.add_argument(
    "--refiner_type",
    type=str,
    default="mlp",
    choices=["none", "mlp", "rf"],
    help="Refiner 类型: none | mlp | rf（Rectified Flow）"
)
parser.add_argument(
    "--repeats",
    type=int,
    default=3,
    help="重复训练次数，用于计算 mean/std（默认 3，对应 seed=42,43,44）"
)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=["cifar10", "cifar100", "tinyimagenet", "pacs", "officehome"],
    help="临时覆盖数据集: cifar10 | cifar100 | tinyimagenet | pacs | officehome"
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.3,
    help="临时覆盖 Dirichlet 浓度参数"
)
parser.add_argument(
    "--method",
    type=str,
    default="fedproref",
    choices=["fedavg", "proto_aug", "proto_cal", "proto_sample", "fedproref"],
    help="临时覆盖训练方法: fedavg | proto_aug | proto_cal | proto_sample | fedproref"
)
parser.add_argument(
    "--pretrained",
    type=str,
    default=None,
    help="临时覆盖 OpenCLIP 预训练权重路径或标签，例如 openai"
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="起始随机种子（默认 42）；repeats>1 时依次使用 seed, seed+1, ..."
)
parser.add_argument(
    "--partition_seed",
    type=int,
    default=42,
    help="数据划分随机种子（默认 42）；不会随 repeats 改变"
)

parser.add_argument(
    "--tag",
    type=str,
    default=None,
    help="实验名附加标签，用于区分消融实验"
)
parser.add_argument("--proto_merge_threshold", type=float, default=0.90,
                    help="原型强相似合并初始阈值")
parser.add_argument("--no_proto_similarity_merge", dest="proto_similarity_merge",
                    action="store_false",
                    help="关闭基于 cosine 相似度的原型合并；上传原型各自保留为全局簇")
parser.add_argument("--no_proto_merge_adaptive", dest="proto_merge_adaptive",
                    action="store_false",
                    help="关闭基于 refiner proto loss 的原型合并阈值自适应")
parser.set_defaults(proto_merge_adaptive=True, proto_similarity_merge=True)
parser.add_argument("--proto_merge_threshold_min", type=float, default=0.70)
parser.add_argument("--proto_merge_threshold_max", type=float, default=0.98)
parser.add_argument("--proto_merge_threshold_step", type=float, default=0.03)
parser.add_argument("--proto_merge_target_low", type=float, default=0.04)
parser.add_argument("--proto_merge_target_high", type=float, default=0.10)
parser.add_argument("--no_proto_merge_learnable", dest="proto_merge_learnable",
                    action="store_false",
                    help="Disable gradient-learned prototype merge threshold")
parser.set_defaults(proto_merge_learnable=True)
parser.add_argument("--proto_merge_threshold_lr", type=float, default=0.05,
                    help="Learning rate for the prototype merge threshold parameter")
parser.add_argument("--proto_merge_temperature", type=float, default=0.05,
                    help="Temperature for soft prototype merge gates used to learn the threshold")
parser.add_argument("--proto_merge_learn_steps", type=int, default=10,
                    help="Gradient steps for learning the prototype merge threshold after each refiner update")
parser.add_argument("--proto_merge_tau_reg", type=float, default=0.1,
                    help="Regularization weight pulling learned threshold toward the refiner-quality target")
parser.add_argument("--proto_merge_acc_weight", type=float, default=0.5,
                    help="Weight of global accuracy feedback when learning the prototype merge threshold")
parser.add_argument("--proto_merge_acc_drop_tolerance", type=float, default=0.3,
                    help="Accuracy drop from historical best, in percentage points, that blocks looser merging")
parser.add_argument("--proto_merge_acc_gain_tolerance", type=float, default=0.2,
                    help="Accuracy change from previous threshold update, in percentage points, treated as strong feedback")
cli_args = parser.parse_args()

REPEATS = cli_args.repeats

# ══════════════════════════════════════════════════════════════════════
#  数据集 & 分区---------------------------------------------------------------------------------------------------------
# ══════════════════════════════════════════════════════════════════════
DATASET        = cli_args.dataset if cli_args.dataset is not None else "cifar100"  # cifar10 | cifar100 | tinyimagenet | pacs | officehome
DATA_DIR       = "./data"
NUM_CLIENTS    = 4 if DATASET in {"pacs", "officehome"} else 10
ALPHA          = cli_args.alpha if cli_args.alpha is not None else 0.01  # Dirichlet 浓度参数（越小越异构）
MIN_REQUIRE_SIZE = 1                 # 每个客户端每个类至少需要的样本数
MIN_SAMPLES_PER_CLASS = 10          # 每个客户端某类样本数低于此值则不上传该类统计信息

# ══════════════════════════════════════════════════════════════════════
#  骨干网络
# ══════════════════════════════════════════════════════════════════════
BACKBONE   = "ViT-B-16"             # OpenCLIP 模型名
PRETRAINED = cli_args.pretrained if cli_args.pretrained is not None else "./pretrain_path/old_open_clip_model.safetensors"  # 预训练权重路径或标签
FEAT_DIM   = 512                    # 特征维度

# ══════════════════════════════════════════════════════════════════════
#  分类头
# ══════════════════════════════════════════════════════════════════════
HEAD_TYPE   = "linear"              # linear | mlp
HEAD_HIDDEN = 512                   # MLP 头隐层维度（仅 head_type=mlp 时生效）

# ══════════════════════════════════════════════════════════════════════
#  联邦学习
# ══════════════════════════════════════════════════════════════════════
COMM_ROUNDS   = 100                 # 通信轮次
LOCAL_EPOCHS  = 10                  # 每轮本地训练轮数
BATCH_SIZE    = 64
LR            = 1e-3                # 本地训练学习率
BETA_HEAD     = 0.2                 # 头部融合系数: W = (1-β)*W_local + β*W_cal

# ══════════════════════════════════════════════════════════════════════
#  多模态分解
# ══════════════════════════════════════════════════════════════════════
NUM_MODES  = 1                      # 每个客户端每个类的 k-means 聚类数
PROTO_MERGE_THRESHOLD = cli_args.proto_merge_threshold  # 原型强相似合并初始阈值
PROTO_SIMILARITY_MERGE = cli_args.proto_similarity_merge  # 是否启用 cosine 相似度原型合并
PROTO_MERGE_ADAPTIVE = cli_args.proto_merge_adaptive  # 根据 refiner proto loss 自适应调整合并阈值
PROTO_MERGE_THRESHOLD_MIN = cli_args.proto_merge_threshold_min
PROTO_MERGE_THRESHOLD_MAX = cli_args.proto_merge_threshold_max
PROTO_MERGE_THRESHOLD_STEP = cli_args.proto_merge_threshold_step
PROTO_MERGE_TARGET_LOW = cli_args.proto_merge_target_low
PROTO_MERGE_TARGET_HIGH = cli_args.proto_merge_target_high
PROTO_MERGE_LEARNABLE = cli_args.proto_merge_learnable
PROTO_MERGE_THRESHOLD_LR = cli_args.proto_merge_threshold_lr
PROTO_MERGE_TEMPERATURE = cli_args.proto_merge_temperature
PROTO_MERGE_LEARN_STEPS = cli_args.proto_merge_learn_steps
PROTO_MERGE_TAU_REG = cli_args.proto_merge_tau_reg
PROTO_MERGE_ACC_WEIGHT = cli_args.proto_merge_acc_weight
PROTO_MERGE_ACC_DROP_TOLERANCE = cli_args.proto_merge_acc_drop_tolerance
PROTO_MERGE_ACC_GAIN_TOLERANCE = cli_args.proto_merge_acc_gain_tolerance

# ══════════════════════════════════════════════════════════════════════
#  提案分布
# ══════════════════════════════════════════════════════════════════════
PROPOSAL_SIGMA = 0.05               # 原型混合提案的噪声尺度

# ══════════════════════════════════════════════════════════════════════
#  Refiner
# ══════════════════════════════════════════════════════════════════════
REFINER_TYPE   = cli_args.refiner_type  # none | mlp | rf（Rectified Flow）
REFINER_HIDDEN = 512                # Refiner 隐层维度
REFINER_LAYERS = 3                  # Refiner 层数
RF_STEPS       = 4                  # RF 的 Euler 积分步数
REFINER_PRETRAIN_EPOCHS = 300       # Fed 开始前预训练 Refiner 的轮数
REFINER_FINETUNE_EPOCHS = 50        # Fed 过程中每次微调 Refiner 的轮数
REFINER_LR     = 1e-3               # Refiner 学习率
CAL_EVERY      = 10                # 每隔多少 round 微调一次 Refiner（0=不微调）

# ══════════════════════════════════════════════════════════════════════
#  损失权重
# ══════════════════════════════════════════════════════════════════════
W_REG    = 0.01                     # 正则化损失权重
W_PROTO  = 0.1                      # 原型对齐损失权重

# ══════════════════════════════════════════════════════════════════════
#  预算校准
# ══════════════════════════════════════════════════════════════════════
GEN_PER_CLASS     = 500             # 服务端每类生成的候选特征数（用于校准）
AUG_GEN_PER_CLASS = 100             # 客户端每个弱类生成的合成样本数（用于本地训练增强）
WEAK_CLASS_PERCENTILE = 10.0        # 样本数排在倒数 X% 的类也视为弱类（0=禁用）
                                    # 例: 20 → 样本量最少的 20% 的类触发合成增强

CAL_BUDGET    = 500                 # 总校准预算 B
CAL_EPOCHS    = 20                  # 校准训练轮数
CAL_LR        = 1e-3                # 校准学习率

# ══════════════════════════════════════════════════════════════════════
#  方法选择（消融实验）
# ══════════════════════════════════════════════════════════════════════
METHOD = cli_args.method            # fedavg | proto_aug | proto_cal | proto_sample | fedproref

# ══════════════════════════════════════════════════════════════════════
#  杂项
# ══════════════════════════════════════════════════════════════════════
DEVICE   = "auto"                   # auto | cuda | cpu
LOG_DIR  = "./logs"
SAVE_DIR = "./checkpoints"
EXP_NAME = "fedproref"

# ══════════════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════════════
def find_latest_log(exp_name):
    pattern = os.path.join(LOG_DIR, f"{exp_name}_*_*.log")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def parse_best_acc(log_file):
    if log_file is None or not os.path.exists(log_file):
        return None
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    matches = re.findall(r"Best:\s*Round\s+\d+\s*\|\s*Acc=([0-9.]+)%", content)
    return float(matches[-1]) if matches else None

# ══════════════════════════════════════════════════════════════════════
#  多次运行
# ══════════════════════════════════════════════════════════════════════
rng = random.SystemRandom()
best_accs = []

print("=" * 60)
print(f"FedProRef  dataset={DATASET}  alpha={ALPHA}  method={METHOD}  repeats={REPEATS}")
print("=" * 60)

for run_idx in range(1, REPEATS + 1):
    seed = cli_args.seed + run_idx - 1 #rng.randint(1, 2**31 - 1)
    tag_part = f"_{cli_args.tag}" if cli_args.tag else ""
    run_exp_name = f"{EXP_NAME}_{METHOD}_{DATASET}_a{ALPHA}{tag_part}_run{run_idx}"
    print(f"\n[{run_idx}/{REPEATS}] 开始  seed={seed}  exp={run_exp_name}")

    cmd = [
        sys.executable, "federated_loop.py",
        "--dataset",                  DATASET,
        "--data_dir",                 DATA_DIR,
        "--num_clients",              str(NUM_CLIENTS),
        "--alpha",                    str(ALPHA),
        "--min_require_size",         str(MIN_REQUIRE_SIZE),
        "--min_samples_per_class",    str(MIN_SAMPLES_PER_CLASS),
        "--backbone",                 BACKBONE,
        "--pretrained",               PRETRAINED,
        "--feat_dim",                 str(FEAT_DIM),
        "--head_type",                HEAD_TYPE,
        "--head_hidden",              str(HEAD_HIDDEN),
        "--comm_rounds",              str(COMM_ROUNDS),
        "--local_epochs",             str(LOCAL_EPOCHS),
        "--batch_size",               str(BATCH_SIZE),
        "--lr",                       str(LR),
        "--beta_head",                str(BETA_HEAD),
        "--num_modes",                str(NUM_MODES),
        "--proto_merge_threshold",    str(PROTO_MERGE_THRESHOLD),
        "--proto_merge_threshold_min", str(PROTO_MERGE_THRESHOLD_MIN),
        "--proto_merge_threshold_max", str(PROTO_MERGE_THRESHOLD_MAX),
        "--proto_merge_threshold_step", str(PROTO_MERGE_THRESHOLD_STEP),
        "--proto_merge_target_low",   str(PROTO_MERGE_TARGET_LOW),
        "--proto_merge_target_high",  str(PROTO_MERGE_TARGET_HIGH),
        "--proto_merge_threshold_lr", str(PROTO_MERGE_THRESHOLD_LR),
        "--proto_merge_temperature",  str(PROTO_MERGE_TEMPERATURE),
        "--proto_merge_learn_steps",  str(PROTO_MERGE_LEARN_STEPS),
        "--proto_merge_tau_reg",      str(PROTO_MERGE_TAU_REG),
        "--proto_merge_acc_weight",   str(PROTO_MERGE_ACC_WEIGHT),
        "--proto_merge_acc_drop_tolerance", str(PROTO_MERGE_ACC_DROP_TOLERANCE),
        "--proto_merge_acc_gain_tolerance", str(PROTO_MERGE_ACC_GAIN_TOLERANCE),
        *([] if PROTO_SIMILARITY_MERGE else ["--no_proto_similarity_merge"]),
        *([] if PROTO_MERGE_ADAPTIVE else ["--no_proto_merge_adaptive"]),
        *([] if PROTO_MERGE_LEARNABLE else ["--no_proto_merge_learnable"]),
        "--proposal_sigma",           str(PROPOSAL_SIGMA),
        "--refiner_type",             REFINER_TYPE,
        "--refiner_hidden",           str(REFINER_HIDDEN),
        "--refiner_layers",           str(REFINER_LAYERS),
        "--rf_steps",                 str(RF_STEPS),
        "--refiner_pretrain_epochs",  str(REFINER_PRETRAIN_EPOCHS),
        "--refiner_finetune_epochs",  str(REFINER_FINETUNE_EPOCHS),
        "--refiner_lr",               str(REFINER_LR),
        "--cal_every",                str(CAL_EVERY),
        "--w_reg",                    str(W_REG),
        "--w_proto",                  str(W_PROTO),
        "--gen_per_class",            str(GEN_PER_CLASS),
        "--aug_gen_per_class",        str(AUG_GEN_PER_CLASS),
        "--weak_class_percentile",    str(WEAK_CLASS_PERCENTILE),
        "--cal_budget",               str(CAL_BUDGET),
        "--cal_epochs",               str(CAL_EPOCHS),
        "--cal_lr",                   str(CAL_LR),
        "--method",                   METHOD,
        "--seed",                     str(seed),
        "--partition_seed",           str(cli_args.partition_seed),
        "--device",                   DEVICE,
        "--log_dir",                  LOG_DIR,
        "--save_dir",                 SAVE_DIR,
        "--exp_name",                 run_exp_name,
    ]

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"[{run_idx}/{REPEATS}] 运行失败 (exit_code={result.returncode})")
        continue

    log_file = find_latest_log(run_exp_name)
    acc = parse_best_acc(log_file)
    if acc is not None:
        best_accs.append(acc)
        print(f"[{run_idx}/{REPEATS}] 完成  Best Acc={acc:.4f}%")
    else:
        print(f"[{run_idx}/{REPEATS}] 完成，但未能解析 Best Acc（log={log_file}）")

# ══════════════════════════════════════════════════════════════════════
#  汇总统计
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"汇总结果  dataset={DATASET}  alpha={ALPHA}")
print("=" * 60)
if best_accs:
    accs_arr = np.array(best_accs)
    for i, v in enumerate(best_accs, 1):
        print(f"  Run {i}: {v:.4f}%")
    print(f"  Mean : {accs_arr.mean():.4f}%")
    print(f"  Std  : {accs_arr.std():.4f}%")
    print(f"  Max  : {accs_arr.max():.4f}%")
else:
    print("  无有效结果。")
print("=" * 60)
