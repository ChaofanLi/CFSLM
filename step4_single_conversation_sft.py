import os
import platform
import argparse
import time
import math
import warnings
import datetime
import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from Config import LMConfig
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')  # 忽略警告信息

def Logger(content, log_file="single_sft_log.txt"):
    # 检查是否为分布式数据并行（DDP）模式，或者当前进程是否为主进程（rank 0）
    if not ddp or dist.get_rank() == 0:
        # 获取当前时间，作为日志时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 拼接日志时间戳和内容
        log_content = f"[{timestamp}] {content}\n"
        
        # 打印日志内容
        print(log_content)
        
        # 将日志内容写入文件
        with open(log_file, "a") as log:
            log.write(log_content)

def get_lr(it, all):
    # 学习率调度函数
    warmup_iters = args.warmup_iters  # 预热迭代次数
    lr_decay_iters = all  # 总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率

    if it < warmup_iters:
        # 预热阶段，线性增加学习率
        return args.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        # 衰减完成，保持最小学习率
        return min_lr
    # 余弦退火学习率衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch, wandb):
    # 单个训练周期
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 数据加载到设备上
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        # 获取当前学习率
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 模型前向传播
            logits = model(X, Y).logits
            # 计算损失函数，忽略索引为0的部分
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()  # 反向传播并缩放梯度

        if (step + 1) % args.accumulation_steps == 0:
            # 梯度累积后进行优化
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer)  # 更新优化器
            scaler.update()

            optimizer.zero_grad(set_to_none=True)  # 清空梯度

        if step % args.log_interval == 0:
            # 打印日志信息
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item(),
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                # 使用wandb记录日志
                wandb.log({"loss": loss,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            # 周期性保存模型
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()

def init_model():
    # 初始化模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('./model/cfslm_tokenizer')
    model_from = 1  # 1表示从权重加载，2表示使用transformers

    def count_parameters(model):
        # 计算模型参数量
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    if model_from == 1:
        # 从预训练权重加载模型
        model = Transformer(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    else:
        # 从transformers加载模型
        model = AutoModelForCausalLM.from_pretrained('cfslm', trust_remote_code=True)

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    model = model.to(args.device)

    return model, tokenizer

def init_distributed_mode():
    # 初始化分布式训练模式
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 创建一个参数解析器，用于从命令行接收参数
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="out", help="输出目录")
    parser.add_argument("--epochs", type=int, default=10, help="训练的总轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每次迭代的批量大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练使用的设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用Weights & Biases进行日志记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="Weights & Biases项目名称")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载的工作线程数")
    parser.add_argument("--ddp", action="store_true", help="是否使用分布式数据并行(DistributedDataParallel)")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热的迭代次数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--local_rank', type=int, default=-1, help='分布式训练的本地rank')

    # 解析命令行参数
    args = parser.parse_args()

    # 加载语言模型配置
    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len

    # 创建保存目录
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 每次迭代的总tokens数目
    tokens_per_iter = args.batch_size * max_seq_len

    # 设置随机种子
    torch.manual_seed(1337)

    # 确定设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 设置Weights & Biases运行名称
    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 根据设备类型设置上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 检查是否是分布式运行
    # ddp = int(os.environ.get("RANK", -1)) != -1
    
    ddp=False
    
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()  # 初始化分布式模式
        args.device = torch.device(DEVICE)

    # 初始化Weights & Biases
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # 初始化模型和分词器
    model, tokenizer = init_model()

    # 加载训练数据集并随机打乱
    df = pd.read_csv('./dataset/sft_data_single.csv')
    df = df.sample(frac=1.0)
    train_ds = SFTDataset(df, tokenizer, max_length=max_seq_len)

    # 设置分布式采样器（如果是分布式模式）
    train_sampler = DistributedSampler(train_ds) if ddp else None

    # 初始化数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 创建梯度缩放器，用于混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 如果满足条件，可以尝试编译模型（仅支持PyTorch 2.0及以上）
    if False and not lm_config.use_moe and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("正在编译模型... (约需一分钟)")
        unoptimized_model = model
        model = torch.compile(model)

    # 如果是分布式运行，使用DistributedDataParallel包装模型
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)

    # 开始训练循环
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)  # 训练单个epoch
