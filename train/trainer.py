"""训练器：完整的训练循环，包含 AMP 混合精度、梯度裁剪、checkpoint 和日志。"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import logging

from .losses import CombinedLoss
from .optimizer import build_optimizer, build_scheduler
from .early_stopping import EarlyStopping
from utils.checkpoint import save_checkpoint, load_checkpoint
from data.augmentation import TrajectoryAugmentation


class Trainer:
    """轻量化轨迹预测模型训练器。

    特性:
    - 自动混合精度 (AMP) 训练，节省显存
    - 梯度裁剪防止爆炸
    - 自动 checkpoint 保存（保存 top-k 最佳模型）
    - TensorBoard 日志记录
    - 早停机制
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        logger: Optional[logging.Logger] = None,
        tb_writer=None,
        monitor_metric: str = "val_min_ade",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.tb_writer = tb_writer

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 损失函数
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedLoss(
            lambda_traj=loss_cfg.get("lambda_traj", 0.7),
            lambda_mode=loss_cfg.get("lambda_mode", 0.3),
            lambda_unc=loss_cfg.get("lambda_unc", 0.0),
            lambda_neighbor=loss_cfg.get("lambda_neighbor", 0.4),
            lambda_collision=loss_cfg.get("lambda_collision", 0.1),
            collision_margin=loss_cfg.get("collision_margin", 2.0),
            traj_loss_type=loss_cfg.get("traj_loss_type", "smooth_l1"),
        )

        # 优化器
        train_cfg = config.get("training", {})
        self.optimizer = build_optimizer(
            self.model,
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

        # 学习率调度器
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_epochs=train_cfg.get("warmup_epochs", 2),
            total_epochs=train_cfg.get("epochs", 20),
            steps_per_epoch=len(train_loader),
            lr_min=train_cfg.get("lr_min", 1e-6),
        )

        # 混合精度
        self.use_amp = train_cfg.get("use_amp", True)
        if self.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # 训练配置
        self.epochs = train_cfg.get("epochs", 20)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)

        # 早停
        es_cfg = config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 5),
            mode=es_cfg.get("mode", "min"),
        )

        # 日志和保存
        log_cfg = config.get("logging", {})
        self.checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "./outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = log_cfg.get("save_top_k", 3)
        self.log_every_n_steps = log_cfg.get("log_every_n_steps", 50)

        # 数据增强
        aug_cfg = config.get("data", {}).get("augmentation", {})
        self.augmentation = TrajectoryAugmentation(
            enable=aug_cfg.get("enable", True),
            rotation_range=aug_cfg.get("rotation_range", 3.14159),
            scale_range=tuple(aug_cfg.get("scale_range", [0.9, 1.1])),
            noise_std=aug_cfg.get("noise_std", 0.02),
        )

        # 追踪
        self.current_epoch = 0
        self.monitor_metric = monitor_metric  # 保存/早停的监控指标（Stage B 用 val_loss）
        self.best_metric = float("inf")
        self.train_losses = []
        self.val_ades = []
        self.saved_checkpoints = []

    def _extract_optional_inputs(self, batch: dict) -> dict:
        """从 batch 提取第二阶段可选输入（车道线 + 城市），传给 model.forward。

        参数:
            batch: collate_fn 输出的 batch dict

        返回:
            kwargs: 可选关键字参数（lane_nodes/lane_adj/lane_mask/city_id）
        """
        kwargs = {}
        if "lane_nodes" in batch:
            kwargs["lane_nodes"] = batch["lane_nodes"].to(self.device)
            kwargs["lane_adj"] = batch["lane_adj"].to(self.device)
            kwargs["lane_mask"] = batch["lane_mask"].to(self.device)
        if "city_id" in batch:
            kwargs["city_id"] = batch["city_id"].to(self.device)
        return kwargs

    def train_epoch(self) -> float:
        """训练一个 epoch。

        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        total_traj_loss = 0.0
        total_mode_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Train]")

        for batch_idx, batch in enumerate(pbar):
            history = batch["history"].to(self.device)   # [B, T_obs, input_dim]
            future = batch["future"].to(self.device)       # [B, T_pred, 2]

            # 第三阶段邻居字段（需增强，单独处理；缺键时为 None）
            neighbor_history = batch["neighbor_history"].to(self.device) if "neighbor_history" in batch else None
            neighbor_future = batch["neighbor_future"].to(self.device) if "neighbor_future" in batch else None
            neighbor_mask = batch["neighbor_mask"].to(self.device) if "neighbor_mask" in batch else None

            # 第二阶段可选输入（车道线为场景级环境信息，不做数据增强）
            model_kwargs = self._extract_optional_inputs(batch)

            # 数据增强（ego 与邻居共享旋转/缩放，保证相对几何一致）
            history, future, neighbor_history, neighbor_future = self.augmentation(
                history, future, neighbor_history, neighbor_future
            )

            # 邻居有效性掩码（top-1 邻居存在性，不随增强改变）
            neighbor_valid = neighbor_mask[:, 0] if neighbor_mask is not None else None

            # 前向传播（使用 AMP）
            if self.use_amp and self.scaler is not None:
                with autocast("cuda"):
                    output = self.model(
                        history,
                        neighbor_history=neighbor_history,
                        neighbor_mask=neighbor_mask,
                        **model_kwargs,
                    )
                    loss_dict = self.criterion(output, future, neighbor_future, neighbor_valid)
                    loss = loss_dict["loss"]

                # 反向传播（AMP）
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(
                    history,
                    neighbor_history=neighbor_history,
                    neighbor_mask=neighbor_mask,
                    **model_kwargs,
                )
                loss_dict = self.criterion(output, future, neighbor_future, neighbor_valid)
                loss = loss_dict["loss"]

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.optimizer.step()

            # 更新学习率（按步调度）
            self.scheduler.step()

            # 累积统计
            total_loss += loss.item()
            total_traj_loss += loss_dict["traj_loss"].item()
            total_mode_loss += loss_dict["mode_loss"].item()

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "traj": f"{loss_dict['traj_loss'].item():.4f}",
                "mode": f"{loss_dict['mode_loss'].item():.4f}",
                "col": f"{loss_dict['collision_loss'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

            # TensorBoard 日志
            if self.tb_writer and (batch_idx + 1) % self.log_every_n_steps == 0:
                global_step = self.current_epoch * num_batches + batch_idx
                self.tb_writer.add_scalar("train/loss", loss.item(), global_step)
                self.tb_writer.add_scalar("train/traj_loss", loss_dict["traj_loss"].item(), global_step)
                self.tb_writer.add_scalar("train/mode_loss", loss_dict["mode_loss"].item(), global_step)
                self.tb_writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        if self.tb_writer:
            self.tb_writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """在验证集上评估模型。

        返回:
            包含 val_loss, val_min_ade, val_min_fde 等指标的字典
        """
        self.model.eval()
        from eval.metrics import compute_min_ade, compute_min_fde, compute_ade, compute_fde

        total_loss = 0.0
        total_min_ade = 0.0
        total_min_fde = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs} [Val]  ")

        for batch in pbar:
            history = batch["history"].to(self.device)
            future = batch["future"].to(self.device)

            # 第三阶段邻居字段（验证不增强）
            neighbor_history = batch["neighbor_history"].to(self.device) if "neighbor_history" in batch else None
            neighbor_future = batch["neighbor_future"].to(self.device) if "neighbor_future" in batch else None
            neighbor_mask = batch["neighbor_mask"].to(self.device) if "neighbor_mask" in batch else None

            model_kwargs = self._extract_optional_inputs(batch)
            neighbor_valid = neighbor_mask[:, 0] if neighbor_mask is not None else None
            output = self.model(
                history,
                neighbor_history=neighbor_history,
                neighbor_mask=neighbor_mask,
                **model_kwargs,
            )
            loss_dict = self.criterion(output, future, neighbor_future, neighbor_valid)

            total_loss += loss_dict["loss"].item()

            # 计算各项指标（逐样本 [B] → 取均值）
            trajectories = output["trajectories"]  # [B, K, T_pred, 2]
            min_ade = compute_min_ade(trajectories, future).mean()   # scalar
            min_fde = compute_min_fde(trajectories, future).mean()   # scalar

            # 最佳模态的 ADE/FDE（用于参考）
            best_idx = loss_dict["best_mode_idx"]
            batch_ade = compute_ade(trajectories, future, best_idx).mean()
            batch_fde = compute_fde(trajectories, future, best_idx).mean()

            total_min_ade += min_ade.item()
            total_min_fde += min_fde.item()
            total_ade += batch_ade.item()
            total_fde += batch_fde.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss_dict['loss'].item():.4f}",
                "minADE": f"{min_ade.item():.4f}",
                "minFDE": f"{min_fde.item():.4f}",
            })

        num_batches = max(num_batches, 1)
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_min_ade": total_min_ade / num_batches,
            "val_min_fde": total_min_fde / num_batches,
            "val_ade": total_ade / num_batches,
            "val_fde": total_fde / num_batches,
        }

        self.val_ades.append(metrics["val_min_ade"])

        # TensorBoard
        if self.tb_writer:
            for key, val in metrics.items():
                self.tb_writer.add_scalar(f"val/{key}", val, self.current_epoch)

        return metrics

    def train(self, resume_from: Optional[str] = None) -> Dict:
        """完整训练流程。

        参数:
            resume_from: checkpoint 路径（用于恢复训练）

        返回:
            包含训练历史的最佳指标字典
        """
        start_epoch = 0

        # 恢复训练
        if resume_from:
            info = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            start_epoch = info["epoch"] + 1
            if self.logger:
                self.logger.info(f"从 checkpoint 恢复: epoch {start_epoch}")

        if self.logger:
            self.logger.info(f"开始训练: {self.epochs} epochs, 设备: {self.device}")
            self.logger.info(f"AMP: {self.use_amp and self.device.type == 'cuda'}")

        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 日志
            if self.logger:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val minADE: {val_metrics['val_min_ade']:.4f} | "
                    f"Val minFDE: {val_metrics['val_min_fde']:.4f}"
                )

            # 保存最佳 checkpoint（监控指标可配置：Stage A 用 minADE，Stage B 用 val_loss）
            current_metric = val_metrics[self.monitor_metric]
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                best_path = self.checkpoint_dir / "best_model.pt"
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, str(best_path),
                )
                if self.logger:
                    self.logger.info(
                        f"  → 保存最佳模型: {self.monitor_metric}={self.best_metric:.4f}"
                    )

            # 定期保存（记录监控指标用于 top-k 筛选）
            ckpt_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, val_metrics, str(ckpt_path),
            )
            self.saved_checkpoints.append((val_metrics[self.monitor_metric], str(ckpt_path)))

            # 保留 minADE 最小的 save_top_k 个 checkpoint
            if len(self.saved_checkpoints) > self.save_top_k:
                # 按 minADE 降序排序，删除最差的
                self.saved_checkpoints.sort(key=lambda x: x[0], reverse=True)
                _, old_ckpt = self.saved_checkpoints.pop(0)
                if os.path.exists(old_ckpt) and "best_model" not in old_ckpt:
                    os.remove(old_ckpt)

            # 早停检查（监控指标与保存一致）
            if self.early_stopping(val_metrics[self.monitor_metric], epoch):
                if self.logger:
                    self.logger.info(
                        f"早停触发！最佳 epoch: {self.early_stopping.best_epoch + 1}, "
                        f"最佳 {self.monitor_metric}: {self.early_stopping.best_score:.4f}"
                    )
                break

        # 训练结束总结
        result = {
            "best_val_min_ade": self.best_metric,  # 向后兼容：默认监控 val_min_ade 时即 minADE
            "best_metric": self.best_metric,
            "monitor_metric": self.monitor_metric,
            "best_epoch": self.early_stopping.best_epoch,
            "train_losses": self.train_losses,
            "val_ades": self.val_ades,
            "early_stopped": self.early_stopping.should_stop,
        }

        if self.logger:
            self.logger.info(
                f"训练完成！最佳 {self.monitor_metric}: {self.best_metric:.4f}"
            )

        return result
