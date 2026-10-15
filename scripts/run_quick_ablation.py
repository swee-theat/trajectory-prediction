"""快速消融实验：每变体 5 epoch，用于快速收集对比数据。"""

import sys, os, torch, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, config_to_dict
from utils.seed import set_seed
from utils.logger import setup_logger
from utils.param_counter import count_parameters
from models.trajectory_model import TrajectoryModel
from data.dataset import ArgoverseTrajectoryDataset
from data.dataloader import create_dataloader
from train.trainer import Trainer
from eval.evaluator import Evaluator
from eval.metrics import compute_min_ade, compute_min_fde
from pathlib import Path

ABLATIONS = [
    ("完整模型(基线)", {}),
    ("去不确定性", {"use_uncertainty": False}),
    ("均值池化", {"pooling_type": "mean"}),
    ("LSTM池化", {"pooling_type": "lstm"}),
    ("2层MLP", {"num_mlp_layers": 2}),
    ("4层MLP", {"num_mlp_layers": 4}),
    ("K=1单模态", {"num_modes": 1}),
    ("K=5多模态", {"num_modes": 5}),
    ("无上下文(6维)", {"input_dim": 6}),
    ("最小特征(4维)", {"input_dim": 4}),
]

def run_one(name, overrides):
    print(f"\n{'='*50}")
    print(f"  消融实验: {name}")
    print(f"{'='*50}")

    cfg = config_to_dict(load_config('configs/default.yaml'))
    cfg['training']['epochs'] = 5
    cfg['training']['warmup_epochs'] = 1
    # 更新模型参数
    for k, v in overrides.items():
        cfg['model'][k] = v

    exp_id = name.replace('(', '').replace(')', '').replace('=', '_').replace(' ', '_')
    cfg['logging']['log_dir'] = f'outputs/logs/abl_{exp_id}'
    cfg['logging']['checkpoint_dir'] = f'outputs/checkpoints/abl_{exp_id}'

    set_seed(42)
    logger = setup_logger(cfg['logging']['log_dir'], exp_id)

    # 数据
    train_ds = ArgoverseTrajectoryDataset('data/processed/train.npz')
    val_ds = ArgoverseTrajectoryDataset('data/processed/val.npz')
    train_loader = create_dataloader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = create_dataloader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    mc = cfg['model']
    input_dim = mc.get('input_dim', 9)

    model = TrajectoryModel(
        input_dim=input_dim, hidden_dim=mc.get('hidden_dim', 128),
        num_mlp_layers=mc.get('num_mlp_layers', 3),
        dropout=mc.get('dropout', 0.1),
        num_modes=mc.get('num_modes', 3),
        pred_len=cfg['data'].get('pred_len', 30),
        pooling_type=mc.get('pooling', 'attention'),
        use_uncertainty=mc.get('use_uncertainty', True),
    )
    total_params, _ = count_parameters(model)
    logger.info(f'参数量: {total_params:,}')

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                       config=cfg, logger=logger, tb_writer=None)
    result = trainer.train()

    return {
        'name': name,
        'params': total_params,
        'best_val_min_ade': result['best_val_min_ade'],
        'best_epoch': result['best_epoch'] + 1,
        'train_losses': result['train_losses'],
        'val_ades': result['val_ades'],
    }

def main():
    results = []
    for name, overrides in ABLATIONS:
        try:
            r = run_one(name, overrides)
            results.append(r)
            print(f"  => {name}: Val minADE={r['best_val_min_ade']:.4f}")
        except Exception as e:
            print(f"  => {name}: 失败 - {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    with open('outputs/results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n消融实验完成! 共 {len(results)}/{len(ABLATIONS)} 个成功")
    for r in results:
        print(f"  {r['name']}: minADE={r['best_val_min_ade']:.4f}, params={r['params']:,}")

if __name__ == '__main__':
    main()
