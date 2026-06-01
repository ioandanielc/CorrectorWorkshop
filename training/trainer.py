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
from utils.logger import create_run_dir, setup_logger, set_logger_eta
from utils.visualizations import plot_comparison, make_sample_gif
from data.data_generator import PoissonDiskDataset, PackedPoissonDiskDataset
from data.data_processor import DataProcessor
from training.loss import hybrid_loss


LOSS_FNS = {
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

    if dataset_cfg.get('type', 'standard') == 'packed':
        dataset = PackedPoissonDiskDataset(
            dim=dataset_cfg['dim'],
            packedness=dataset_cfg['packedness'],
            rd=dataset_cfg['rd'],
            seed=dataset_cfg['seed'],
            noise_scale_min=dataset_cfg['noise_scale_min'],
            noise_scale_max=dataset_cfg['noise_scale_max'],
        )
        logger.info(
            f"Packed dataset: packedness={dataset_cfg['packedness']}, "
            f"N_target={dataset.n_target}, rd={dataset_cfg['rd']}"
        )
    else:
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

    loss_csv = open(run_dir / "loss.csv", "w", buffering=1)   # line-buffered
    loss_csv.write(
        "iteration,loss,lr,"
        "mean_violation_pre,mean_violation_k1,median_violation_pre,median_violation_k1,"
        "illegal_pair_pct_pre,illegal_pair_pct_k1,"
        "mean_viol_per_cloud_pre,mean_viol_per_cloud_k1,viol_reduction_pct_k1,"
        "mean_nn_dist_pre,mean_nn_dist_k1,median_nn_dist_pre,median_nn_dist_k1,"
        "displacement_k1,displacement_rel_rd_k1,displacement_median_k1,displacement_rel_rd_median_k1,"
        "correction_eff_k1,efficiency_pct_k1,"
        "illegal_pair_pct_kK,mean_viol_per_cloud_kK,viol_reduction_pct_kK,efficiency_pct_kK\n"
    )

    logger.info(f"Starting training for {train_cfg['num_iterations']} iterations")

    iter_timer = time.time()
    for iteration in range(1, train_cfg['num_iterations'] + 1):
        model.train()

        clean = dataset.generate_sample(train_cfg['batch_size'])
        noisy = dataset.noise_sample(clean)
        noisy_processed, _, _ = processor.make_invariant(noisy)
        x = torch.tensor(noisy_processed, dtype=torch.float32).to(device)

        x_current  = x
        total_loss = torch.tensor(0.0, device=device)
        unroll_steps = train_cfg.get('unroll_steps', 1)
        for _ in range(unroll_steps):
            displacement = model(x_current, rd=rd) if getattr(model, 'uses_rd', False) else model(x_current)
            x_next       = x_current + displacement
            total_loss   = total_loss + loss_fn(x_current, x_next, rd, **loss_cfg['params'])
            x_current    = x_next.detach()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % eval_cfg['log_interval'] == 0:
            with torch.no_grad():
                eye      = torch.eye(x.shape[1], dtype=torch.bool, device=device)
                off_diag = ~eye

                # ── Pre-correction (noisy input x) ─────────────────────────────
                pw_pre        = torch.cdist(x, x)
                pd_pre        = pw_pre[:, off_diag]
                viol_pre      = torch.relu(rd - pd_pre)
                mean_viol_pre = viol_pre.mean().item()
                med_viol_pre  = viol_pre.median().item()
                ill_pct_pre   = (pd_pre < rd).float().mean().item() * 100
                ill_count_pre = (pd_pre < rd).float().sum(dim=-1).mean().item() / 2  # mean unique illegal pairs per cloud
                nn_pre        = pw_pre.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values
                mean_nn_pre   = nn_pre.mean().item()
                med_nn_pre    = nn_pre.median().item()

                def _post_stats(cloud, disp_tensor):
                    """Compute post-correction stats for a given corrected cloud."""
                    pw   = torch.cdist(cloud, cloud)
                    pd   = pw[:, off_diag]
                    viol = torch.relu(rd - pd)
                    nn   = pw.masked_fill(eye.unsqueeze(0), float('inf')).min(dim=-1).values
                    dpt  = disp_tensor.norm(dim=-1)
                    return dict(
                        mean_viol      = viol.mean().item(),
                        med_viol       = viol.median().item(),
                        ill_pct        = (pd < rd).float().mean().item() * 100,
                        ill_count      = (pd < rd).float().sum(dim=-1).mean().item() / 2,  # mean unique illegal pairs per cloud
                        mean_nn        = nn.mean().item(),
                        med_nn         = nn.median().item(),
                        disp           = dpt.mean().item(),
                        disp_med       = dpt.median().item(),
                    )

                # ── K=1 inference (single pass) ────────────────────────────────
                disp1      = model(x, rd=rd) if getattr(model, 'uses_rd', False) else model(x)
                corr1      = x + disp1
                s1         = _post_stats(corr1, disp1)

                # ── K=unroll_steps inference (multi-pass) ──────────────────────
                xk = x
                for _ in range(unroll_steps):
                    dk  = model(xk, rd=rd) if getattr(model, 'uses_rd', False) else model(xk)
                    xk  = xk + dk
                sk         = _post_stats(xk, xk - x)   # total displacement from original

                N               = x.shape[1]
                efficiency_ceil = 2.0 / (N - 1)

                def _eff(s):
                    viol_removed = mean_viol_pre - s['mean_viol']
                    eff = viol_removed / (s['disp'] + 1e-8)
                    return eff, (eff / efficiency_ceil) * 100

                eff1,  pct1  = _eff(s1)
                effk,  pctk  = _eff(sk)

                viol_red_k1 = (ill_count_pre - s1['ill_count']) / (ill_count_pre + 1e-9) * 100
                viol_red_kK = (ill_count_pre - sk['ill_count']) / (ill_count_pre + 1e-9) * 100

            secs_per_iter = (time.time() - iter_timer) / eval_cfg['log_interval']
            iter_timer = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            remaining_iters = train_cfg['num_iterations'] - iteration
            set_logger_eta(logger, remaining_iters * secs_per_iter)

            logger.info(
                f"iter={iteration:6d}  {secs_per_iter:.3f}s/iter  lr={current_lr:.2e}  "
                f"loss={total_loss.item():.6f}\n"
                f"  {'':20s}  {'K=1 (deploy)':>22s}   {'K='+str(unroll_steps)+' (train)':>22s}\n"
                f"  mean_violation   = {mean_viol_pre:.6f}  ->  {s1['mean_viol']:.6f}              {sk['mean_viol']:.6f}\n"
                f"  illegal_pairs    = {ill_pct_pre:.2f}%      ->  {s1['ill_pct']:.2f}%                 {sk['ill_pct']:.2f}%\n"
                f"  viol_per_cloud   = {ill_count_pre:.2f}       ->  {s1['ill_count']:.2f}                  {sk['ill_count']:.2f}\n"
                f"  viol_reduction   = {'':10s}     {viol_red_k1:.1f}%                    {viol_red_kK:.1f}%\n"
                f"  mean_nn_dist     = {mean_nn_pre:.6f}  ->  {s1['mean_nn']:.6f}              {sk['mean_nn']:.6f}\n"
                f"  displacement     = {'':10s}     {s1['disp']:.4f} ({s1['disp']/dataset_cfg['rd']:.3f}x rd)   "
                f"{sk['disp']:.4f} ({sk['disp']/dataset_cfg['rd']:.3f}x rd)\n"
                f"  correction_eff   = {'':10s}     {eff1:.4f} ({pct1:.1f}% ceil)         {effk:.4f} ({pctk:.1f}% ceil)"
            )
            # CSV: write K=1 columns (primary), plus K=K illegal% and legal% for tracking
            loss_csv.write(
                f"{iteration},{total_loss.item():.6f},{current_lr:.2e},"
                f"{mean_viol_pre:.6f},{s1['mean_viol']:.6f},"
                f"{med_viol_pre:.6f},{s1['med_viol']:.6f},"
                f"{ill_pct_pre:.2f},{s1['ill_pct']:.2f},"
                f"{ill_count_pre:.4f},{s1['ill_count']:.4f},{viol_red_k1:.2f},"
                f"{mean_nn_pre:.6f},{s1['mean_nn']:.6f},"
                f"{med_nn_pre:.6f},{s1['med_nn']:.6f},"
                f"{s1['disp']:.6f},{s1['disp']/dataset_cfg['rd']:.6f},"
                f"{s1['disp_med']:.6f},{s1['disp_med']/dataset_cfg['rd']:.6f},"
                f"{eff1:.6f},{pct1:.2f},"
                f"{sk['ill_pct']:.2f},{sk['ill_count']:.4f},{viol_red_kK:.2f},{pctk:.2f}\n"
            )
            loss_csv.flush()

            # expose for _save_sample (K=1)
            corrected = corr1

        if iteration % eval_cfg['sample_interval'] == 0:
            _save_sample(model, processor, val_noisy, dataset_cfg, device, run_dir, iteration,
                         eval_cfg['num_visual_samples'])
            logger.debug(f"Sample saved at iteration {iteration}")

    loss_csv.close()
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    logger.info(f"Training complete. Model saved to {run_dir / 'model_final.pt'}")

    gif_path = make_sample_gif(run_dir, fps=10, every_n=1)
    logger.info(f"Evolution GIF saved to {gif_path}")


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
