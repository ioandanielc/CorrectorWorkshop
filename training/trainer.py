import importlib
import shutil
import time
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler_module
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

from utils.config import load_train_config, load_dataset_config, load_loss_config, load_model_config
from utils.logger import create_run_dir, setup_logger
from utils.visualizations import plot_comparison
from data.data_generator import PoissonDiskDataset
from data.data_processor import DataProcessor
from training.loss import classic_loss, rd_weighted_loss, coverage_loss, hybrid_loss


LOSS_FNS = {
    'classic_loss': classic_loss,
    'rd_weighted_loss': rd_weighted_loss,
    'coverage_loss': coverage_loss,
    'hybrid_loss': hybrid_loss,
}


def train(train_config_path, dataset_config_path, loss_config_path, model_config_path):
    train_cfg = load_train_config(train_config_path)
    dataset_cfg = load_dataset_config(dataset_config_path)
    loss_cfg = load_loss_config(loss_config_path)
    model_cfg = load_model_config(model_config_path)

    eval_cfg = train_cfg['eval']

    run_dir = create_run_dir()
    logger = setup_logger(run_dir)
    logger.info(f"Run directory: {run_dir}")

    for path in [train_config_path, dataset_config_path, loss_config_path, model_config_path]:
        shutil.copy(path, run_dir / "configs" / Path(path).name)
    logger.info("Configs copied to run directory")

    device = torch.device(train_cfg['device'])
    logger.info(f"Using device: {device}")

    dataset = PoissonDiskDataset(
        dim=dataset_cfg['dim'],
        cardinality=dataset_cfg['points_per_cloud'],
        rd=dataset_cfg['rd'],
        seed=dataset_cfg['seed'],
        noise_scale_min=dataset_cfg['noise_scale_min'],
        noise_scale_max=dataset_cfg['noise_scale_max'],
    )
    processor = DataProcessor()

    logger.info(f"Generating validation set ({eval_cfg['validation_size']} clouds)...")
    val_clean = dataset.generate_sample(eval_cfg['validation_size'])
    val_noisy = dataset.noise_sample(val_clean)
    np.save(run_dir / "validation_set.npy", val_noisy)
    logger.info("Validation set saved")

    model_module = importlib.import_module(model_cfg['model_file'].replace('/', '.'))
    CorrectorModel = model_module.CorrectorModel
    model = CorrectorModel(
        model_config=model_cfg,
        input_dim=dataset_cfg['dim'],
        initialization=train_cfg['initialization'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built: {n_params:,} parameters")

    optimizer_cls = getattr(optim, train_cfg['optimizer']['name'])
    optimizer = optimizer_cls(model.parameters(), **train_cfg['optimizer']['params'])

    scheduler_cls = getattr(lr_scheduler_module, train_cfg['lr_scheduler']['name'])
    scheduler = scheduler_cls(optimizer, **train_cfg['lr_scheduler']['params'])

    loss_fn = LOSS_FNS[loss_cfg['name']]
    rd = torch.tensor(dataset_cfg['rd'], dtype=torch.float32, device=device)

    loss_csv = open(run_dir / "loss.csv", "w")
    loss_csv.write(
        "iteration,loss,lr,"
        "mean_violation_pre,mean_violation,median_violation_pre,median_violation,"
        "illegal_pair_pct_pre,illegal_pair_pct,"
        "legal_cloud_pct_pre,legal_cloud_pct,"
        "mean_nn_dist_pre,mean_nn_dist,median_nn_dist_pre,median_nn_dist,"
        "displacement,displacement_rel_rd,displacement_median,displacement_rel_rd_median\n"
    )

    logger.info(f"Starting training for {train_cfg['num_iterations']} iterations")

    iter_timer = time.time()
    for iteration in range(1, train_cfg['num_iterations'] + 1):
        model.train()

        clean = dataset.generate_sample(train_cfg['batch_size'])
        noisy = dataset.noise_sample(clean)
        noisy_processed, _, _ = processor.make_invariant(noisy)
        x = torch.tensor(noisy_processed, dtype=torch.float32).to(device)

        displacement = model(x, rd=rd) if getattr(model, 'uses_rd', False) else model(x)
        corrected = x + displacement
        loss_val = loss_fn(x, corrected, rd, **loss_cfg['params'])

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        scheduler.step()

        if iteration % eval_cfg['log_interval'] == 0:
            with torch.no_grad():
                eye     = torch.eye(x.shape[1], dtype=torch.bool, device=device)
                off_diag = ~eye

                # ── Pre-correction (noisy input x) ─────────────────────────────
                pw_pre        = torch.cdist(x, x)
                pd_pre        = pw_pre[:, off_diag]                         # (B, N*(N-1))
                viol_pre      = torch.relu(rd - pd_pre)
                mean_viol_pre = viol_pre.mean().item()
                med_viol_pre  = viol_pre.median().item()
                ill_pct_pre   = (pd_pre < rd).float().mean().item() * 100
                leg_pct_pre   = (~(pd_pre < rd).any(dim=-1)).float().mean().item() * 100
                nn_pre        = pw_pre.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values
                mean_nn_pre   = nn_pre.mean().item()
                med_nn_pre    = nn_pre.median().item()

                # ── Post-correction ────────────────────────────────────────────
                pw_post        = torch.cdist(corrected, corrected)
                pd_post        = pw_post[:, off_diag]
                viol_post      = torch.relu(rd - pd_post)
                mean_violation = viol_post.mean().item()
                med_violation  = viol_post.median().item()
                illegal_pair_pct = (pd_post < rd).float().mean().item() * 100
                legal_cloud_pct  = (~(pd_post < rd).any(dim=-1)).float().mean().item() * 100
                nn_post        = pw_post.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values
                mean_nn_dist   = nn_post.mean().item()
                med_nn_dist    = nn_post.median().item()

                # ── Displacement ───────────────────────────────────────────────
                disp_per_pt    = displacement.norm(dim=-1)                  # (B, N)
                disp           = disp_per_pt.mean().item()
                disp_med       = disp_per_pt.median().item()
                disp_rel       = disp     / dataset_cfg['rd']
                disp_rel_med   = disp_med / dataset_cfg['rd']

            secs_per_iter = (time.time() - iter_timer) / eval_cfg['log_interval']
            iter_timer = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"iter={iteration:6d}  {secs_per_iter:.3f}s/iter  lr={current_lr:.2e}  "
                f"loss={loss_val.item():.6f}\n"
                f"  mean_violation   = {mean_viol_pre:.6f} -> {mean_violation:.6f}"
                f"   (median: {med_viol_pre:.6f} -> {med_violation:.6f})"
                f"   [0 = perfect]\n"
                f"  illegal_pairs    = {ill_pct_pre:.2f}% -> {illegal_pair_pct:.2f}%"
                f"   [0% = perfect]\n"
                f"  legal_clouds     = {leg_pct_pre:.2f}% -> {legal_cloud_pct:.2f}%"
                f"   [100% = perfect]\n"
                f"  mean_nn_dist     = {mean_nn_pre:.6f} -> {mean_nn_dist:.6f}"
                f"   (median: {med_nn_pre:.6f} -> {med_nn_dist:.6f})"
                f"   [>= rd={dataset_cfg['rd']}]\n"
                f"  displacement     = {disp:.6f} ({disp_rel:.3f}x rd)"
                f"   (median: {disp_med:.6f} = {disp_rel_med:.3f}x rd)"
            )
            loss_csv.write(
                f"{iteration},{loss_val.item():.6f},{current_lr:.2e},"
                f"{mean_viol_pre:.6f},{mean_violation:.6f},"
                f"{med_viol_pre:.6f},{med_violation:.6f},"
                f"{ill_pct_pre:.2f},{illegal_pair_pct:.2f},"
                f"{leg_pct_pre:.2f},{legal_cloud_pct:.2f},"
                f"{mean_nn_pre:.6f},{mean_nn_dist:.6f},"
                f"{med_nn_pre:.6f},{med_nn_dist:.6f},"
                f"{disp:.6f},{disp_rel:.6f},{disp_med:.6f},{disp_rel_med:.6f}\n"
            )
            loss_csv.flush()

        if iteration % eval_cfg['sample_interval'] == 0:
            _save_sample(model, processor, val_noisy, dataset_cfg, device, run_dir, iteration,
                         eval_cfg['num_visual_samples'])
            logger.debug(f"Sample saved at iteration {iteration}")

    loss_csv.close()
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    logger.info(f"Training complete. Model saved to {run_dir / 'model_final.pt'}")


def _save_sample(model, processor, val_noisy, dataset_cfg, device, run_dir, iteration, num_visual_samples):
    model.eval()
    with torch.no_grad():
        batch = val_noisy[:num_visual_samples]
        processed, _, _ = processor.make_invariant(batch)
        x = torch.tensor(processed, dtype=torch.float32).to(device)
        rd_tensor = torch.tensor(dataset_cfg['rd'], dtype=torch.float32, device=device)
        displacement = model(x, rd=rd_tensor) if getattr(model, 'uses_rd', False) else model(x)
        corrected = (x + displacement).cpu().numpy()

    plot_comparison(
        noisy_clouds=processed,
        corrected_clouds=corrected,
        rd=dataset_cfg['rd'],
        save_path=run_dir / "samples" / f"sample_{iteration:06d}.png",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config",   required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--loss-config",    required=True)
    parser.add_argument("--model-config",   required=True)
    args = parser.parse_args()
    train(
        train_config_path=args.train_config,
        dataset_config_path=args.dataset_config,
        loss_config_path=args.loss_config,
        model_config_path=args.model_config,
    )
