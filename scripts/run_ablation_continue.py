"""消融实验续跑：跳过已完成的"完整模型(基线)"，从"去不确定性"开始。"""
import sys, os, json, traceback, time
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

ABLATIONS = [
    ("去不确定性", {"use_uncertainty": False}),
    ("均值池化", {"pooling_type": "mean"}),
    ("LSTM池化", {"pooling_type": "lstm"}),
    ("2层MLP", {"num_mlp_layers": 2}),
    ("4层MLP", {"num_mlp_layers": 4}),
    ("K_1单模态", {"num_modes": 1}),
    ("K_5多模态", {"num_modes": 5}),
    ("无上下文6维", {"input_dim": 6}),
    ("最小特征4维", {"input_dim": 4}),
]


RESULTS_PATH = "outputs/results/ablation_results.json"

def load_existing_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # 没有文件则从基线结果开始
    return [{"name": "完整模型(基线)", "params": 145331,
             "best_val_min_ade": 0.7629, "best_epoch": 5}]

def save_results(results):
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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

    train_ds = ArgoverseTrajectoryDataset("data/processed/train.npz")
    val_ds = ArgoverseTrajectoryDataset("data/processed/val.npz")
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    mc = cfg["model"]
    input_dim = mc.get("input_dim", 9)

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
    results = load_existing_results()
    done_names = {r["name"] for r in results}
    print(f"已有 {len(results)} 个结果: {done_names}")

    for name, overrides in ABLATIONS:
        if name in done_names:
            print(f"跳过已完成: {name}")
            continue
        try:
            r = run_one(name, overrides)
            results.append(r)
            save_results(results)  # 每完成一个立即保存
            elapsed = (time.time() - t0) / 60
            print(f"  => {name}: Val minADE={r['best_val_min_ade']:.4f} | 累计耗时 {elapsed:.0f}分钟")
        except Exception as e:
            print(f"  => {name}: 失败 - {e}")
            traceback.print_exc()
            # 继续下一个，不中断整体流程

    elapsed = (time.time() - t0) / 60
    print(f"\n消融实验完成! 共 {len(results)}/{len(ABLATIONS)+1} 个，耗时 {elapsed:.0f}分钟")
    for r in results:
        print(f"  {r['name']}: minADE={r['best_val_min_ade']:.4f}, params={r['params']:,}")

if __name__ == "__main__":
    main()
