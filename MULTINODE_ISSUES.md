# MeanFlow 多节点训练问题排查

针对 `train.sh` + `train.py` + `dataset.py` 在多节点（Arnold 多机多卡）场景下的潜在问题，按严重程度排序。

---

## 🔴 严重问题

### 2. Resume 完全没接通，抢占后从 step 0 重训

**位置**：`train.sh`（没有传 `--resume-step`）、`train.py:183`

```python
if args.resume_step > 0:
    ckpt = torch.load(...)
```

**问题**：
- `train.sh` 没传 `--resume-step`，默认 0。
- Arnold 上长任务经常被抢占重启，重启后**所有训练进度丢失**，只保留了已下载的数据。
- 即使手动传了 `--resume-step`，dataloader 也是从 epoch 0 重新迭代，**没有 skip 已训过的 batch**，会重复消耗初始 epoch 的样本顺序。

**修复建议**：
- 在 `train.sh` 里自动探测最新 ckpt：
  ```bash
  CKPT_DIR="exp/meanflow_l_2/checkpoints"
  LATEST=$(ls "$CKPT_DIR" 2>/dev/null | sort -n | tail -1 | sed 's/.pt$//' | sed 's/^0*//')
  RESUME_ARG=""
  [[ -n "$LATEST" ]] && RESUME_ARG="--resume-step $LATEST"
  ```
  然后把 `$RESUME_ARG` 传给 `train.py`。
- 在 `train.py` 里加上 `accelerator.skip_first_batches(train_dataloader, n)` 跳过已训过的 batch。

---

### 3. Checkpoint 保存方式不规范

**位置**：`train.py:245-256`

```python
if global_step % args.checkpointing_steps == 0 and global_step > 0 ...:
    if accelerator.is_main_process:
        checkpoint = {
            "model": model.module.state_dict(),   # ⚠️ 直接用 .module
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            ...
        }
        torch.save(checkpoint, checkpoint_path)
```

**问题**：
- `model.module.state_dict()` 在单 GPU/无 DDP 情况下会 AttributeError。应该用 `accelerator.unwrap_model(model).state_dict()`。
- 保存前后**没有 `accelerator.wait_for_everyone()`**。rank 0 写盘时其它 rank 进入下一个 batch，下次 `backward` 时会被 DDP all-reduce 阻塞 → 不会出错但浪费算力。如果 ckpt 较大，rank 间通信可能超时。
- 推荐用 `accelerator.save()` 或 `accelerator.save_state()`，框架会处理同步和分布式细节。

**修复建议**：

```python
if global_step % args.checkpointing_steps == 0 and global_step > 0 ...:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint = {
            "model": accelerator.unwrap_model(model).state_dict(),
            "ema": ema.state_dict(),
            "opt": optimizer.state_dict(),
            "args": vars(args),  # 用 dict 而不是 Namespace
            "steps": global_step,
        }
        accelerator.save(checkpoint, checkpoint_path)
    accelerator.wait_for_everyone()
```

---

## 🟡 中等问题

### 4. EMA 初始化时机错位，rank 间会漂移

**位置**：`train.py:177`

```python
update_ema(ema, model, decay=0)  # ⚠️ 在 accelerator.prepare 之前
model, optimizer, train_dataloader = accelerator.prepare(...)
```

**问题**：
- 每个 rank 因为 `set_seed(seed + process_index)`（train.py:106）用**不同种子**初始化了 model。
- `ema = deepcopy(model)` 拷贝的是各 rank 不同的 model 权重。
- `prepare` 后 DDP 把 rank 0 的 model 广播给所有 rank，**但 ema 不在 DDP 里**，仍然是各 rank 不同。
- 之后每步 `update_ema` 用相同的 model param 叠加在不同的 ema 上，ema 会**持续在 rank 间漂移**。
- 只有 rank 0 的 ema 被保存到 ckpt，所以保存本身没问题；但如果用 ema 做分布式推理/评估，各 rank 结果会不一致。

**修复建议**：把 ema 初始化放到 `accelerator.prepare` 之后：

```python
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
ema = deepcopy(accelerator.unwrap_model(model)).to(device)
requires_grad(ema, False)
```

---

### 5. 每个 process 用不同 seed 初始化模型

**位置**：`train.py:106`

```python
set_seed(args.seed + accelerator.process_index)
```

**问题**：
- 在 model 创建之前调用，导致各 rank 的模型权重不同。
- 功能上依赖 DDP `prepare` 时的初始 param 广播来纠正，**能跑但很脆弱**。
- 如果未来改成 FSDP 或换框架，可能不会自动广播，bug 会暴露。

**修复建议**：模型创建用统一 seed，DataLoader/augmentation 那层再用不同 seed：

```python
set_seed(args.seed)            # 统一种子建 model
model = SiT_models[...](...)
set_seed(args.seed + accelerator.process_index)  # 数据增强用不同种子
```

---

### 6. `max_train_steps` 计算逻辑混乱

**位置**：`train.py:174-175`

```python
steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
args.max_train_steps = args.epochs * steps_per_epoch // accelerator.num_processes
```

**问题**：
- `len(train_dataloader)` 在 `accelerator.prepare` **之前**算的，值为 `len(dataset) // local_batch_size = num_processes * len(dataset) / batch_size`。
- 最后再除一次 `num_processes`，数值上**恰好等于** `epochs * len(dataset) / batch_size`，但属于"凑巧正确"。
- 一旦改 batch 配置或加 `gradient_accumulation_steps`，公式就错。

**修复建议**：在 `accelerator.prepare` 之后再算：

```python
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
steps_per_epoch = len(train_dataloader) // accelerator.gradient_accumulation_steps
args.max_train_steps = args.epochs * steps_per_epoch
```

---

### 7. 没设置 NCCL / 网络环境变量

**位置**：`train.sh` 整体

**问题**：多机训练在 Arnold 多网卡环境下，NCCL 可能选错网卡，导致带宽低或卡死。

**修复建议**：

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0   # 根据 Arnold 实际网卡名调整
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
```

---

## 🟢 较小问题

### 8. `--data-dir` 指向 `data.mdb` 文件而不是 lmdb 目录

**位置**：`train.sh:30`

```bash
--data-dir "/opt/tiger/MeanFlow/data_and_model/imagenet_train_latents.lmdb/data.mdb"
```

**问题**：`dataset.py` 中 `lmdb.open(...)` 默认 `subdir=True`，期望传目录路径。当前能跑说明 lmdb 自动识别成 single-file 模式了，但写法不规范。

**修复建议**：传到 `.lmdb` 目录：

```bash
--data-dir "/opt/tiger/MeanFlow/data_and_model/imagenet_train_latents.lmdb"
```

---

### 9. 日志只在 main process 写文件

**位置**：`train.py:94`（`create_logger` 只在 `is_main_process` 分支调用）

**问题**：非 main rank 的报错只到 stdout，多节点时容易丢失非 rank 0 节点的关键错误信息（比如 OOM、NCCL 错误）。

**修复建议**：每个节点至少让本地 rank 0 写一份日志：

```python
if accelerator.is_local_main_process:
    log_file = f"{save_dir}/log_node{os.environ.get('ARNOLD_ID', 0)}.txt"
    # 配置 FileHandler
```

---

### 10. `accelerator.gather(grad_norm)` 类型不一致

**位置**：`train.py:232-261`

```python
grad_norm = 0.0
if accelerator.sync_gradients:
    grad_norm = accelerator.clip_grad_norm_(...)   # tensor
...
"grad_norm": accelerator.gather(grad_norm).mean()  # 当 0.0 时是 Python float
```

**问题**：当 `sync_gradients=False`（梯度累积期间）时，`grad_norm` 是 Python float，`accelerator.gather` 会报类型错误。当前 `gradient_accumulation_steps=1`（默认）不会触发，但只要打开梯度累积就坏。

**修复建议**：

```python
grad_norm = torch.tensor(0.0, device=device)
```

---

## 修复优先级

按"立即修 / 计划修 / 锦上添花"分组：

| 优先级 | 问题 | 影响 |
|--------|------|------|
| 🔥 立即修 | #1 HDFS 下载同步 | 雪崩 HDFS / 数据损坏 |
| 🔥 立即修 | #2 Resume 自动接通 | 抢占重启全部丢进度 |
| 🔥 立即修 | #3 Ckpt 保存加 wait_for_everyone | 算力浪费 + 潜在超时 |
| ⚙️ 计划修 | #4 EMA 初始化时机 | 多 rank EMA 漂移 |
| ⚙️ 计划修 | #5 模型初始化 seed | 依赖 DDP 隐式广播 |
| ⚙️ 计划修 | #6 `max_train_steps` 公式 | 改配置即出错 |
| ⚙️ 计划修 | #7 NCCL 环境变量 | 多机带宽 / 稳定性 |
| ✨ 锦上添花 | #8 lmdb 路径 | 不规范但能跑 |
| ✨ 锦上添花 | #9 多节点日志 | 排查困难 |
| ✨ 锦上添花 | #10 gather 类型 | 只在开启梯度累积时触发 |
