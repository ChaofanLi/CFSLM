import os
import time
import math
import torch
import platform
import warnings
import datetime
import numpy as np
import pandas as pd
from torch import optim
import torch.distributed as dist
from contextlib import nullcontext
from model.model import Transformer
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config.slm_config import SLMConfig
from config.pretrain_config import train_config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

warnings.filterwarnings('ignore')

def Logger(content, log_file="./log/pretrain_log.txt"):
    if not ddp or dist.get_rank() == 0: 
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_content = f"[{timestamp}] {content}\n"
        print(log_content)
        with open(log_file, "a") as log:
            log.write(log_content)

class PretrainDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = self.df.iloc[index]
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        loss_mask = [1] * text_len + [0] * padding_len
        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

def get_lr(it, all):
    # warmup+余弦退火学习率调度
    # it：当前迭代次数
    # all：总迭代次数

    warmup_iters = train_config.warmup_iters
    lr_decay_iters = all
    min_lr = train_config.learning_rate / 10
    if it < warmup_iters:
        return train_config.learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (train_config.learning_rate - min_lr)


def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        
        # 将数据移动到指定的设备（GPU/CPU）
        X = X.to(train_config.device)
        Y = Y.to(train_config.device)
        loss_mask = loss_mask.to(train_config.device)
        
        # 计算当前学习率
        lr = get_lr(epoch * iter_per_epoch + step, train_config.epochs * iter_per_epoch)
        
        # 更新优化器中每个参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
        # 混合精度计算损失
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / train_config.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # 反向传播计算梯度
        scaler.scale(loss).backward()

        # 每经过一定的累积步数，进行参数更新
        if (step + 1) % train_config.accumulation_steps == 0:
            # 解缩放梯度，以准备进行优化器步骤
            scaler.unscale_(optimizer)
            # 对梯度进行裁剪，以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

            # 使用优化器进行参数更新
            scaler.step(optimizer)
            
            # 更新缩放器状态
            scaler.update()

            # 清零梯度，准备下一次反向传播
            optimizer.zero_grad(set_to_none=True)

        # 每隔一定步数记录训练状态
        if step % train_config.log_interval == 0:
            # 计算训练周期的花费时间
            spend_time = time.time() - start_time
            # 记录当前训练状态
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,                              # 当前训练周期
                    train_config.epochs,                        # 总训练周期
                    step,                               # 当前批次
                    iter_per_epoch,                    # 每个周期的总批次数
                    loss.item() * train_config.accumulation_steps,  # 计算的损失值
                    optimizer.param_groups[-1]['lr'],  # 当前学习率
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 已用时间（分钟）
                )
            )

            # 如果 wandb（Weights and Biases）对象不为 None 且处于主进程，则记录训练信息
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss.item() * train_config.accumulation_steps,  # 记录损失
                    "lr": optimizer.param_groups[-1]['lr'],         # 记录学习率
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 记录已用时间
                })

        if (step + 1) % train_config.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{train_config.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained('./cfslm_tokenizer')

    model = Transformer(lm_config).to(train_config.device)
    # moe_path = '_moe' if lm_config.use_moe else ''

    Logger(f'CFSLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    lm_config = SLMConfig()
    train_config = train_config()
    max_seq_len = lm_config.max_seq_len
    train_config.save_dir = train_config.out_dir
    os.makedirs(train_config.save_dir, exist_ok=True)
    tokens_per_iter = train_config.batch_size * max_seq_len
    torch.manual_seed(101)
    device_type = "cuda" if "cuda" in train_config.device else "cpu"

    train_config.wandb_run_name = f"cfslm-pretrain-epoch-{train_config.epochs}-batch_size-{train_config.batch_size}-learning_rate-{train_config.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = False # 单卡
    ddp_local_rank, DEVICE = 0, "cuda:0"
    
    if ddp:
        init_distributed_mode()
        train_config.device = torch.device(DEVICE)

    if train_config.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=train_config.wandb_project, name=train_config.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model()
    df = pd.read_csv(train_config.data_path)
    df = df.sample(frac=1.0)
    train_ds = PretrainDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=train_config.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)

    if False and platform.system() != 'Windows' and float(torch.__version__.split('.')[0]) >= 2:
        Logger("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(train_config.epochs):
        train_epoch(epoch, wandb)