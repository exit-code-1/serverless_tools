import os
import sys
import time
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# 将项目根目录添加到 sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils

# 定义网络模型
class Exec_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_exp=0.0, max_exp=2.0):
        super(Exec_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 5),  # 输出 a, b, c, d, e
        )
        self.min_exp = min_exp
        self.max_exp = max_exp

    def forward(self, x):
        x = self.bn_input(x)
        pred_params = self.fc(x)

        span = self.max_exp - self.min_exp
        a = self.min_exp + span * torch.sigmoid(pred_params[:, 0])
        b = pred_params[:, 1]
        c = pred_params[:, 2]
        d = self.min_exp + span * torch.sigmoid(pred_params[:, 3])
        e = pred_params[:, 4]

        return torch.stack([a, b, c, d, e], dim=1)
    
# 定义网络模型
class Mem_CurveFitModel(nn.Module):
    def __init__(self, input_dim, min_a=-1, max_a=1):
        super(Mem_CurveFitModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 4),  # 输出 a, b, c, d
        )
        self.min_a = min_a
        self.max_a = max_a

    def forward(self, x): 
        x = self.bn_input(x)
        pred_params = self.fc(x)
        
        # 获取 a, b, c
        a, b, c, d = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
        
        # 对 a 应用 Sigmoid 激活函数并映射到 [min_a, max_a]
        # a = torch.sigmoid(a) * (self.max_a - self.min_a) + self.min_a

        # 返回映射后的参数
        return torch.stack([a, b, c, d], dim=1)
    
def reset_model(model):
    """重新初始化模型参数"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def _current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _build_lr_scheduler(
    optimizer,
    scheduler,
    epochs,
    lr,
    step_size=50,
    gamma=0.9,
    eta_min=None,
    plateau_patience=15,
    plateau_factor=0.5,
    exp_gamma=0.995,
):
    """Build LR scheduler. Default recommendation: ``cosine`` or ``plateau``.

    - none: constant LR
    - step: StepLR (gentler defaults than old step_size=10, gamma=0.8)
    - cosine: CosineAnnealingLR over full ``epochs``
    - plateau: ReduceLROnPlateau when loss stops improving
    - exponential: ExponentialLR per epoch (exp_gamma~0.995 decays slowly)
    """
    name = (scheduler or "cosine").lower()
    if name == "none":
        return None
    if name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "cosine":
        if eta_min is None:
            eta_min = lr * 0.01
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=eta_min
        )
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=lr * 1e-4,
        )
    if name == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_gamma)
    raise ValueError(
        f"Unknown scheduler '{scheduler}'; use none|step|cosine|plateau|exponential"
    )


def _step_lr_scheduler(scheduler, avg_loss):
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(avg_loss)
    else:
        scheduler.step()


def _derive_segment_ids(X, decimals=6):
    """Group samples by feature row.

    For dop-aware training the feature vector does NOT include ``dop`` itself
    (see ``dop_operator_features``), so rows with identical feature values
    correspond to the same operator configuration measured at different dops,
    i.e. one curve. We collapse those rows into a single segment id.
    Floating-point noise from feature engineering is suppressed by rounding
    each entry to ``decimals`` digits before deduplication.
    """
    arr = X.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    keys = np.round(arr, decimals=decimals)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    # numpy 2.x may return shape (n, 1); flatten so downstream comparisons work.
    inverse = np.asarray(inverse).reshape(-1)
    return torch.as_tensor(inverse, dtype=torch.long)


def _iter_segment_batches(segment_ids, batch_segments, shuffle=True):
    """Yield row indices grouped by ``batch_segments`` curves at a time.

    Each yielded tensor contains the full set of (feature, dop) rows belonging
    to the selected curves, so a single batch carries every dop measurement
    for ``batch_segments`` distinct operator configurations.
    """
    seg_np = segment_ids.cpu().numpy()
    unique_segs = np.unique(seg_np)
    if shuffle:
        unique_segs = np.random.permutation(unique_segs)

    seg_to_rows = {int(s): np.where(seg_np == s)[0] for s in unique_segs}

    for i in range(0, len(unique_segs), batch_segments):
        batch_segs = unique_segs[i : i + batch_segments]
        rows = np.concatenate([seg_to_rows[int(s)] for s in batch_segs])
        yield torch.as_tensor(rows, dtype=torch.long)


def _pairwise_grad_loss(pred_time, true_time, dop, epsilon):
    """Pairwise dT/d(dop) mismatch averaged along a single curve."""
    if dop.numel() < 2:
        return torch.tensor(0.0, device=pred_time.device, dtype=pred_time.dtype)

    sort_idx = torch.argsort(dop)
    dop_sorted = dop[sort_idx]
    delta_dop = dop_sorted[1:] - dop_sorted[:-1]
    valid = torch.abs(delta_dop) > epsilon
    if not torch.any(valid):
        return torch.tensor(0.0, device=pred_time.device, dtype=pred_time.dtype)

    delta_dop = delta_dop[valid]
    true_sorted = true_time[sort_idx]
    pred_sorted = pred_time[sort_idx]
    true_grad = (true_sorted[1:] - true_sorted[:-1])[valid] / delta_dop
    pred_grad = (pred_sorted[1:] - pred_sorted[:-1])[valid] / delta_dop
    return torch.mean(torch.abs(pred_grad - true_grad) / (torch.abs(true_grad) + epsilon))


def _segment_grad_loss(pred_time, true_time, dop, segment_ids, epsilon):
    """Average ``_pairwise_grad_loss`` across all curves in a batch."""
    if segment_ids is None:
        return _pairwise_grad_loss(pred_time, true_time, dop, epsilon)

    unique_segs = torch.unique(segment_ids)
    losses = []
    for seg in unique_segs:
        mask = segment_ids == seg
        if mask.sum() < 2:
            continue
        losses.append(
            _pairwise_grad_loss(pred_time[mask], true_time[mask], dop[mask], epsilon)
        )

    if not losses:
        return torch.tensor(0.0, device=pred_time.device, dtype=pred_time.dtype)
    return torch.stack(losses).mean()


def curve_exec_loss(
    pred_params,
    dop,
    true_time,
    segment_ids=None,
    epsilon=1e-2,
    grad_weight=1.0,
    rel_denom_floor=1.0,
    log_weight=1.0,
    use_dynamic_grad_weight=True,
    log_file="loss_debug.log",
):
    """Point + gradient loss for the execution-time curve.

    Point term = relative error + log_weight * log1p error.
    Do NOT multiply ``|pred-true|/true`` by ``true/mean`` — they cancel and
    collapse huge ms-level mistakes to O(1) (~1.0 even when pred is wrong by
    four orders of magnitude). The log term penalises order-of-magnitude gaps.
    """
    a, b, c, d, e = (
        pred_params[:, 0],
        pred_params[:, 1],
        pred_params[:, 2],
        pred_params[:, 3],
        pred_params[:, 4],
    )

    dop_safe = torch.clamp(dop, min=epsilon)
    pred_time = b / (dop_safe ** a) + c * (dop_safe ** d) + e

    true_pos = torch.clamp(torch.abs(true_time), min=rel_denom_floor)
    pred_pos = torch.clamp(pred_time, min=0.0)

    # ~1.0 means ~100% relative error (not clamped by batch mean).
    rel_err = torch.abs(pred_time - true_time) / true_pos
    # Large when pred and true differ by orders of magnitude (e.g. 7 vs 13000).
    log_err = torch.abs(torch.log1p(pred_pos) - torch.log1p(true_pos))
    point_loss = torch.mean(rel_err + log_weight * log_err)

    grad_loss = _segment_grad_loss(
        pred_time=pred_time,
        true_time=true_time,
        dop=dop_safe,
        segment_ids=segment_ids,
        epsilon=epsilon,
    )

    if grad_weight == 0:
        return point_loss

    if use_dynamic_grad_weight:
        scale = point_loss.detach() / (grad_loss.detach() + 1e-8)
        effective_grad_weight = grad_weight * scale
    else:
        effective_grad_weight = grad_weight

    return point_loss + effective_grad_weight * grad_loss

def curve_mem_loss(pred_params, dop, true_mem, epsilon=1e-2, alpha=0.5, log_file="loss_debug.log"):
    # 打开日志文件（以追加模式），并写入错误信息
    # def log_to_file(message):
    #     with open(log_file, "a") as f:
    #         f.write(message + "\n")
    # 修改 log_to_file 函数内部打开文件的路径
    def log_to_file(message):
        # 确保目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f:
            f.write(message + "\n")

    a, b, c, d = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
    
    # 计算预测时间
    pred_mem = torch.max(b * (dop ** a) + c, d)

    # 如果 pred_time 为 NaN 或者小于等于零，打印 a, b, c 和 pred_time
    if torch.any(torch.isnan(pred_mem)):
        log_to_file(f"NaN or invalid pred_time detected!")
        log_to_file(f"a: {a}")
        log_to_file(f"b: {b}")
        log_to_file(f"c: {c}")
        log_to_file(f"pred_time: {pred_mem}")


    # 如果 pred_time 小于 true_time，将 abs_error 乘以 2
    abs_error = torch.abs(pred_mem - true_mem)
    log_error = torch.log(abs_error + 1)
    log_error = torch.where(pred_mem < true_mem, log_error, log_error)
    pred_mem = torch.clamp(pred_mem, min=1e-2)
    # 计算相对误差
    relative_error = torch.log(torch.max(pred_mem/true_mem, true_mem/pred_mem))
    
    # 返回最终损失
    loss = torch.mean(log_error + relative_error)  # 加上负值惩罚项
    
    return loss

def train_exec_curve_model(
    X_train,
    y_train,
    dop_train,
    segment_ids=None,
    batch_size=8,
    batch_segments=8,
    epochs=100,
    lr=5e-3,
    grad_weight=1.0,
    rel_denom_floor=1.0,
    use_dynamic_grad_weight=True,
    scheduler="cosine",
    scheduler_step_size=50,
    scheduler_gamma=0.9,
    scheduler_eta_min=None,
    plateau_patience=15,
    plateau_factor=0.5,
    exp_gamma=0.995,
):
    """Train the dop-aware execution-time curve model.

    By default each batch is built from ``batch_segments`` distinct curves
    (each curve = one operator configuration measured at multiple dops), so
    that the gradient-matching term inside ``curve_exec_loss`` is well
    defined. ``segment_ids`` can be supplied by the caller; otherwise it is
    derived from feature-row uniqueness (the dop column is not part of the
    feature vector in this module). When ``batch_segments`` is 0 or no
    multi-dop curve can be found, training falls back to the legacy row-batch
    DataLoader path.

    Parameters
    ----------
    X_train, y_train, dop_train : torch.Tensor
        Feature matrix, target execution time and dop column.
    segment_ids : torch.LongTensor, optional
        Per-sample curve id. Auto-derived when omitted.
    batch_size : int
        Row-batch size for the fallback path.
    batch_segments : int
        Number of curves per segment-batch (set to 0 to disable).
    epochs : int
    lr : float
    grad_weight : float
        Scale for the gradient term after dynamic rebalancing; 0 disables it.
    rel_denom_floor : float
        Lower bound on ``true_time`` in the point-loss denominator.
    use_dynamic_grad_weight : bool
        When True, rescale grad_loss so it matches point_loss in magnitude.
    scheduler : str
        ``none`` | ``cosine`` | ``plateau`` | ``step`` | ``exponential``.
        Default ``cosine`` (smooth decay). Use ``plateau`` to drop LR only when
        loss stalls; ``none`` for constant LR.
    """
    input_dim = X_train.shape[1]
    model = Exec_CurveFitModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)
    lr_scheduler = _build_lr_scheduler(
        optimizer,
        scheduler=scheduler,
        epochs=epochs,
        lr=lr,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
        eta_min=scheduler_eta_min,
        plateau_patience=plateau_patience,
        plateau_factor=plateau_factor,
        exp_gamma=exp_gamma,
    )

    use_segment_batching = bool(batch_segments and batch_segments > 0)
    if use_segment_batching:
        if segment_ids is None:
            segment_ids = _derive_segment_ids(X_train)
        else:
            segment_ids = torch.as_tensor(segment_ids, dtype=torch.long)
        n_segments = int(torch.unique(segment_ids).numel())
        # Disable segment batching when there is essentially nothing to group
        # (e.g. every row is its own curve) so the gradient term cannot fire.
        if n_segments < 2 or n_segments >= X_train.shape[0]:
            use_segment_batching = False

    if not use_segment_batching:
        train_dataset = TensorDataset(X_train, y_train, dop_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_segment_batching:
            for batch_rows in _iter_segment_batches(
                segment_ids, batch_segments=batch_segments, shuffle=True
            ):
                if batch_rows.numel() < 2:
                    continue

                optimizer.zero_grad()
                X_batch = X_train[batch_rows]
                y_batch = y_train[batch_rows]
                dop_batch = dop_train[batch_rows]
                seg_batch = segment_ids[batch_rows]

                pred_params = model(X_batch)
                loss = curve_exec_loss(
                    pred_params,
                    dop_batch,
                    y_batch,
                    segment_ids=seg_batch,
                    grad_weight=grad_weight,
                    rel_denom_floor=rel_denom_floor,
                    use_dynamic_grad_weight=use_dynamic_grad_weight,
                )
                if torch.any(torch.isnan(loss)):
                    print(
                        f"NaN detected at epoch {epoch} (segment batch). "
                        f"Resetting model parameters."
                    )
                    model = reset_model(model)
                    cur_lr = _current_lr(optimizer)
                    optimizer = optim.Adam(model.parameters(), lr=cur_lr, eps=1e-4)
                    lr_scheduler = _build_lr_scheduler(
                        optimizer,
                        scheduler=scheduler,
                        epochs=epochs,
                        lr=cur_lr,
                        step_size=scheduler_step_size,
                        gamma=scheduler_gamma,
                        eta_min=scheduler_eta_min,
                        plateau_patience=plateau_patience,
                        plateau_factor=plateau_factor,
                        exp_gamma=exp_gamma,
                    )
                    break

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
        else:
            for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                pred_params = model(X_batch)
                loss = curve_exec_loss(
                    pred_params,
                    dop_batch,
                    y_batch,
                    grad_weight=grad_weight,
                    rel_denom_floor=rel_denom_floor,
                    use_dynamic_grad_weight=use_dynamic_grad_weight,
                )
                if torch.any(torch.isnan(loss)):
                    print(
                        f"NaN detected at epoch {epoch}, batch {batch_idx}. "
                        f"Resetting model parameters."
                    )
                    model = reset_model(model)
                    cur_lr = _current_lr(optimizer)
                    optimizer = optim.Adam(model.parameters(), lr=cur_lr, eps=1e-4)
                    lr_scheduler = _build_lr_scheduler(
                        optimizer,
                        scheduler=scheduler,
                        epochs=epochs,
                        lr=cur_lr,
                        step_size=scheduler_step_size,
                        gamma=scheduler_gamma,
                        eta_min=scheduler_eta_min,
                        plateau_patience=plateau_patience,
                        plateau_factor=plateau_factor,
                        exp_gamma=exp_gamma,
                    )
                    break

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        _step_lr_scheduler(lr_scheduler, avg_epoch_loss)

        if (epoch + 1) % 10 == 0:
            mode = "segment" if use_segment_batching else "row"
            print(
                f"Epoch [{epoch + 1}/{epochs}] mode={mode} "
                f"avg_loss={avg_epoch_loss:.4f} "
                f"lr={_current_lr(optimizer):.6f} scheduler={scheduler}"
            )

    training_time = time.time() - start_time
    return model, training_time


def train_mem_curve_model(
    X_train,
    y_train,
    dop_train,
    batch_size=16,
    epochs=100,
    lr=1e-2,
    scheduler="cosine",
    scheduler_step_size=50,
    scheduler_gamma=0.9,
    scheduler_eta_min=None,
    plateau_patience=15,
    plateau_factor=0.5,
    exp_gamma=0.995,
):
    """
    训练用于预测曲线参数的模型，使用批量训练，并加入学习率调度器。

    Parameters:
    - X_train: Tensor, 特征
    - y_train: Tensor, 实际执行时间
    - dop_train: Tensor, 并行度
    - batch_size: int, 批次大小
    - epochs: int, 训练轮数
    - lr: float, 初始学习率
    - scheduler: see ``train_exec_curve_model``

    Returns:
    - model: CurveFitModel, 训练后的模型
    - training_time: float, 训练时间
    """
    input_dim = X_train.shape[1]
    model = Mem_CurveFitModel(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-4)

    train_dataset = TensorDataset(X_train, y_train, dop_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    lr_scheduler = _build_lr_scheduler(
        optimizer,
        scheduler=scheduler,
        epochs=epochs,
        lr=lr,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
        eta_min=scheduler_eta_min,
        plateau_patience=plateau_patience,
        plateau_factor=plateau_factor,
        exp_gamma=exp_gamma,
    )

    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0  # 初始化该 epoch 的总损失
        
        for batch_idx, (X_batch, y_batch, dop_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            pred_params = model(X_batch)
            loss = curve_mem_loss(pred_params, dop_batch, y_batch)
            if torch.any(torch.isnan(loss)):
                print(f"NaN detected at epoch {epoch}, batch {batch_idx}. Resetting model parameters.")
                model = reset_model(model)
                cur_lr = _current_lr(optimizer)
                optimizer = optim.Adam(model.parameters(), lr=cur_lr, eps=1e-4)
                lr_scheduler = _build_lr_scheduler(
                    optimizer,
                    scheduler=scheduler,
                    epochs=epochs,
                    lr=cur_lr,
                    step_size=scheduler_step_size,
                    gamma=scheduler_gamma,
                    eta_min=scheduler_eta_min,
                    plateau_patience=plateau_patience,
                    plateau_factor=plateau_factor,
                    exp_gamma=exp_gamma,
                )
                break

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        n_batches = max(len(train_loader), 1)
        avg_epoch_loss = epoch_loss / n_batches
        _step_lr_scheduler(lr_scheduler, avg_epoch_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_epoch_loss:.4f}, "
                f"lr={_current_lr(optimizer):.6f} scheduler={scheduler}"
            )

    training_time = time.time() - start_time  # 计算训练时间
    return model, training_time  # 返回模型和训练时间

def predict_and_evaluate_exec_curve(
    model,
    X_test,
    y_test,
    dop_test,
    epsilon=1e-2,
    operator=None,
    suffix="",
    onnx_model_dir: str = None,
):
    """
    使用模型进行预测并评估性能，同时保存 ONNX 模型并比较预测时间。

    Parameters:
    - model: CurveFitModel, 训练好的 PyTorch 模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度
    - operator: str, 操作符类型，用于命名保存的 ONNX 文件
    - output_prefix: str, 保存 ONNX 模型的前缀路径
    - suffix: str, ONNX 模型文件名后缀

    Returns:
    - results: dict, 包含预测性能和时间对比
    """
    model.eval()

    # 原生 PyTorch 模型预测
    start_time = time.time()
    with torch.no_grad():
        pred_params = model(X_test)
        a, b, c, d, e = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3], pred_params[:, 4]
        predictions_native = torch.relu(b / (dop_test**a) + c * dop_test**d + e)
        predictions_native = torch.clamp(predictions_native, 1e-2)
        # predictions_native = torch.maximum(b * (dop_test ** a), c)
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        if onnx_model_dir is None:
            # Backward compatibility (previous hard-coded output).
            onnx_model_dir = "../output/models/operator_dop_aware"
        operator_name = operator.replace(' ', '_')
        onnx_path = os.path.join(onnx_model_dir, operator, f"{suffix}_{operator_name}.onnx")
        onnx_dir = os.path.dirname(onnx_path)
        os.makedirs(onnx_dir, exist_ok=True)

        # 导出 ONNX 模型
        dummy_input = torch.randn(X_test.size(0), X_test.size(1))
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"ONNX model saved to: {onnx_path}")

    # 使用 ONNX Runtime 进行预测
    predictions_onnx = None
    onnx_time = None
    if onnx_path:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        start_time = time.time()
        predictions_onnx = session.run(None, {input_name: X_test.numpy().astype(np.float32)})[0]
        onnx_time = time.time() - start_time

     # Calculate standard mean absolute error (MAE) for native predictions
    mae_native = torch.mean(torch.abs(y_test - predictions_native))

    # Calculate Prediction Accuracy for native model
    Q_error = torch.mean(
         torch.maximum(y_test / predictions_native, predictions_native / y_test) - 1
    )

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Native': predictions_native,
        'Difference_Native': y_test - predictions_native,
    })

    # Calculate Prediction Accuracy for ONNX model
    time_accuracy_onnx = None
    if onnx_time is not None:
        time_accuracy_onnx = torch.mean(
            (1 - torch.abs(y_test - predictions_native) / (y_test + epsilon)) * 100
        )

    # Calculate the average of the actual target values
    avg_actual_value = torch.mean(y_test)

    # Print model prediction times
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # Organize the performance metrics into one dictionary
    performance = {
        "metrics": {
            "MAE_error": mae_native,
            "Q_error": Q_error,
            "average_actual_value": avg_actual_value
        },
        "comparisons": comparisons,
        "native_time": native_time,
        "onnx_time": onnx_time,
        "onnx_accuracy": time_accuracy_onnx
    }

    return performance


def predict_and_evaluate_mem_curve(
    model,
    X_test,
    y_test,
    dop_test,
    epsilon=1e-2,
    operator=None,
    suffix="",
    onnx_model_dir: str = None,
):
    """
    使用模型进行预测并评估性能，同时保存 ONNX 模型并比较预测时间。

    Parameters:
    - model: CurveFitModel, 训练好的 PyTorch 模型
    - X_test: Tensor, 测试特征
    - y_test: Tensor, 实际执行时间
    - dop_test: Tensor, 测试并行度
    - operator: str, 操作符类型，用于命名保存的 ONNX 文件
    - output_prefix: str, 保存 ONNX 模型的前缀路径
    - suffix: str, ONNX 模型文件名后缀

    Returns:
    - results: dict, 包含预测性能和时间对比
    """
    model.eval()

    # 原生 PyTorch 模型预测
    start_time = time.time()
    with torch.no_grad():
        pred_params = model(X_test)
        a, b, c, d = pred_params[:, 0], pred_params[:, 1], pred_params[:, 2], pred_params[:, 3]
        predictions_native = torch.relu(torch.max(b * (dop_test ** a) + c, d))
        predictions_native = torch.clamp(predictions_native, 1e-2)
        # predictions_native = torch.maximum(b * (dop_test ** a), c)
    native_time = time.time() - start_time

    # 保存为 ONNX 模型
    onnx_path = None
    if operator:
        if onnx_model_dir is None:
            # Backward compatibility (previous hard-coded output).
            onnx_model_dir = "../output/models/operator_dop_aware"
        operator_name = operator.replace(' ', '_')
        onnx_path = os.path.join(onnx_model_dir, operator, f"{suffix}_{operator_name}.onnx")
        onnx_dir = os.path.dirname(onnx_path)
        os.makedirs(onnx_dir, exist_ok=True)

        # 导出 ONNX 模型
        dummy_input = torch.randn(X_test.size(0), X_test.size(1))
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"ONNX model saved to: {onnx_path}")

    # 使用 ONNX Runtime 进行预测
    predictions_onnx = None
    onnx_time = None
    if onnx_path:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        start_time = time.time()
        predictions_onnx = session.run(None, {input_name: X_test.numpy().astype(np.float32)})[0]
        onnx_time = time.time() - start_time

     # Calculate standard mean absolute error (MAE) for native predictions
    mae_native = torch.mean(torch.abs(y_test - predictions_native))

    # Calculate Prediction Accuracy for native model
    Q_error = torch.mean(
         torch.maximum((y_test / predictions_native) , (predictions_native / y_test)) - 1
    )

    # Create comparison DataFrame
    comparisons = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Native': predictions_native,
        'Difference_Native': y_test - predictions_native,
    })

    # Calculate Prediction Accuracy for ONNX model
    time_accuracy_onnx = None
    if onnx_time is not None:
        time_accuracy_onnx = torch.mean(
            (1 - torch.abs(y_test - predictions_native) / (y_test + epsilon)) * 100
        )

    # Calculate the average of the actual target values
    avg_actual_value = torch.mean(y_test)

    # Print model prediction times
    print(f"Native model prediction time: {native_time:.6f} seconds")
    if onnx_time is not None:
        print(f"ONNX model prediction time: {onnx_time:.6f} seconds")

    # Organize the performance metrics into one dictionary
    performance = {
        "metrics": {
            "MAE_error": mae_native,
            "Q_error": Q_error,
            "average_actual_value": avg_actual_value
        },
        "comparisons": comparisons,
        "native_time": native_time,
        "onnx_time": onnx_time,
        "onnx_accuracy": time_accuracy_onnx
    }

    return performance