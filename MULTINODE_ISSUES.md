# MeanFlow 多节点训练问题排查

## 🟡 中等问题

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
