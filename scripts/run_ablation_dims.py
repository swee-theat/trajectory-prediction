"""补跑 input_dim=6 和 input_dim=4 消融实验（需切片数据特征）。"""
import sys, os, json, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from pathlib import Path
from torch.utils.data import Dataset

class SlicedDataset(Dataset):
    """只保留前 input_dim 维特征的包装器。"""
    def __init__(self, ds, input_dim):
        self.ds = ds
        self.input_dim = input_dim
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        item = self.ds[idx]
        item["history"] = item["history"][:, :self.input_dim].clone()
        return item

EXPERIMENTS = [
    ("无上下文6维", {"input_dim": 6}),
    ("最小特征4维", {"input_dim": 4}),
]

RESULTS_PATH = "outputs/results/ablation_results.json"

def run_one(name, overrides):
    print(f"\n{'='*50}")
    print(f"  消融实验: {name}")
    print(f"{'='*50}")

    cfg = config_to_dict(load_config("configs/default.yaml"))
    cfg["training"]["epochs"] = 5
    cfg["training"]["warmup_epochs"] = 1
    for k, v in overrides.items():
        cfg["model"][k] = v

    exp_id = name.replace("(", "").replace(")", "").replace("=", "_")
    cfg["logging"]["log_dir"] = f"outputs/logs/abl_{exp_id}"
    cfg["logging"]["checkpoint_dir"] = f"outputs/checkpoints/abl_{exp_id}"

    set_seed(42)
    logger = setup_logger(cfg["logging"]["log_dir"], exp_id)

    mc = cfg["model"]
    input_dim = mc.get("input_dim", 9)

    # 用 SlicedDataset 裁剪特征维度
    train_ds = SlicedDataset(
        ArgoverseTrajectoryDataset("data/processed/train.npz"), input_dim)
    val_ds = SlicedDataset(
        ArgoverseTrajectoryDataset("data/processed/val.npz"), input_dim)
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = TrajectoryModel(
        input_dim=input_dim,
        hidden_dim=mc.get("hidden_dim", 128),
        num_mlp_layers=mc.get("num_mlp_layers", 3),
        dropout=mc.get("dropout", 0.1),
        num_modes=mc.get("num_modes", 3),
        pred_len=cfg["data"].get("pred_len", 30),
        pooling_type=mc.get("pooling", "attention"),
        use_uncertainty=mc.get("use_uncertainty", True),
    )
    total_params, _ = count_parameters(model)
    logger.info(f"参数量: {total_params:,}")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        config=cfg, logger=logger, tb_writer=None,
    )
    result = trainer.train()

    return {
        "name": name,
        "params": total_params,
        "best_val_min_ade": result["best_val_min_ade"],
        "best_epoch": result["best_epoch"] + 1,
        "train_losses": result["train_losses"],
        "val_ades": result["val_ades"],
    }

def main():
    t0 = time.time()
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    done_names = {r["name"] for r in results}
    print(f"已有 {len(results)} 个结果: {done_names}")

    for name, overrides in EXPERIMENTS:
        if name in done_names:
            print(f"跳过已完成: {name}")
            continue
        try:
            r = run_one(name, overrides)
            results.append(r)
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            elapsed = (time.time() - t0) / 60
            print(f"  => {name}: Val minADE={r['best_val_min_ade']:.4f} | {elapsed:.0f}min")
        except Exception as e:
            print(f"  => {name}: 失败 - {e}")
            traceback.print_exc()

    print(f"\n补跑完成! 共 {len(results)} 个变体, {(time.time()-t0)/60:.0f}分钟")

if __name__ == "__main__":
    main()
