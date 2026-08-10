import importlib
import shutil
import time
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler_module
import matplotlib
matplotlib.use('Agg')
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))   # src/ — run from the project root

from utils.config import load_train_config, load_dataset_config, load_loss_config, load_model_config
from utils.logger import create_run_dir, setup_logger, set_logger_eta
from utils.visualizations.training_visualizations.visualizations import plot_comparison, make_sample_gif
from training.datagen import PoissonDiskDataset
from training.loss import sph_loss, rdsph_loss, mean_kg_norm


LOSS_FNS = {
    'sph_loss':    sph_loss,
    'rdsph_loss':  rdsph_loss,
}

SPH_LOSSES = ('sph_loss', 'rdsph_loss')   # kernel-gradient (symmetry) losses


def train(train_config_path, dataset_config_path, loss_config_path, model_config_path,
          seed=None):
    train_cfg = load_train_config(train_config_path)
    if seed is not None:
        train_cfg['seed'] = seed
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
        periodic=dataset_cfg.get('periodic', False),
    )

    # Periodic clouds live on the unit torus; box-aware models get box=1.0 below.
    # (model12 is translation-invariant by construction — no frame transform.)
    is_periodic = bool(dataset_cfg.get('periodic', False))

    logger.info(f"Generating validation set ({eval_cfg['validation_size']} clouds)...")
    val_clean = dataset.generate_sample(eval_cfg['validation_size'])
    val_noisy = dataset.noise_sample(val_clean)
    val_rd = dataset_cfg['rd']
    val_rd_tensor = torch.tensor(val_rd, dtype=torch.float32, device=device)
    np.save(run_dir / "validation_set.npy", val_noisy)
    logger.info("Validation set saved")

    # Seeds only torch, so model INITIALISATION varies while the data sequence stays
    # fixed by the dataset config's own seed — which is what isolates "is this result
    # an initialisation artifact" from "is this result a property of the recipe".
    # Absent = unseeded, the historical behaviour of every run before 2026-08-10.
    if train_cfg.get('seed') is not None:
        torch.manual_seed(int(train_cfg['seed']))
        logger.info(f"Torch seed: {train_cfg['seed']} (init only; data seed is the dataset config's)")

    model_module = importlib.import_module(model_cfg['model_file'].replace('/', '.'))
    CorrectorModel = model_module.CorrectorModel
    model = CorrectorModel(
        model_config=model_cfg,
        input_dim=dataset_cfg['dim'],
        initialization=train_cfg['initialization'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model built: {n_params:,} parameters")

    # box-aware models (model12) get the periodic box so wrap-seam pairs are visible
    model_kwargs = {'box': 1.0} if (is_periodic and getattr(model, 'uses_box', False)) else {}
    if model_kwargs:
        logger.info("Periodic model geometry: passing box=1.0 to forward()")

    optimizer_cls = getattr(optim, train_cfg['optimizer']['name'])
    optimizer = optimizer_cls(model.parameters(), **train_cfg['optimizer']['params'])

    scheduler_cls = getattr(lr_scheduler_module, train_cfg['lr_scheduler']['name'])
    scheduler = scheduler_cls(optimizer, **train_cfg['lr_scheduler']['params'])

    loss_fn = LOSS_FNS[loss_cfg['name']]
    rd = torch.tensor(dataset_cfg['rd'], dtype=torch.float32, device=device)

    # Cutoff radius: what the model "sees". Defaults to the constraint rd;
    # model_config.cutoff_rd decouples them (SPH losses: the kernel-gradient
    # asymmetry lives mostly in pairs ABOVE rd, which a violation-gated cutoff
    # would otherwise weight zero). The loss always uses the constraint rd.
    cutoff_rd = model_cfg.get('cutoff_rd')
    rd_model = (torch.tensor(cutoff_rd, dtype=torch.float32, device=device)
                if cutoff_rd is not None else rd)
    if cutoff_rd is not None:
        logger.info(f"Cutoff rd = {cutoff_rd} (constraint rd = {dataset_cfg.get('rd')})")

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

    batch_size   = train_cfg['batch_size']
    unroll_steps = train_cfg.get('unroll_steps', 1)
    # Dense edges are (B, N, N, ·): memory grows as batch_size * N^2. micro_batch_size
    # splits a batch into chunks that fit, leaving the effective batch — and so the
    # gradient noise — identical across cardinalities. Unset = one chunk = batch_size.
    micro_batch_size = int(train_cfg.get('micro_batch_size', batch_size))
    if micro_batch_size < batch_size:
        logger.info(f"Gradient accumulation: {batch_size} = "
                    f"{-(-batch_size // micro_batch_size)} x micro-batch {micro_batch_size}")

    best_val_loss = float('inf')
    best_iteration = None
    iter_timer = time.time()
    for iteration in range(1, train_cfg['num_iterations'] + 1):
        model.train()

        clean = dataset.generate_sample(batch_size)
        noisy = dataset.noise_sample(clean)
        x = torch.tensor(noisy, dtype=torch.float32).to(device)

        # Each unrolled step is detached from the next, so its loss is an independent
        # function of the parameters and sum_k backward(L_k) == backward(sum_k L_k).
        # Backpropagating inside the loop frees each step's activations immediately
        # instead of holding all unroll_steps graphs at once. Every micro-batch
        # contributes its share of the batch mean, so the accumulated gradient is the
        # full-batch gradient.
        optimizer.zero_grad()
        total_loss = 0.0
        for start in range(0, batch_size, micro_batch_size):
            xb    = x[start:start + micro_batch_size]
            share = xb.shape[0] / batch_size

            x_current = xb
            for _ in range(unroll_steps):
                displacement = model(x_current, rd=rd_model, **model_kwargs) if getattr(model, 'uses_rd', False) else model(x_current)
                x_next       = x_current + displacement
                step_loss    = loss_fn(x_current, x_next, rd, **loss_cfg['params']) * share
                step_loss.backward()
                total_loss  += step_loss.item()
                x_current    = x_next.detach()

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
                disp1      = model(x, rd=rd_model, **model_kwargs) if getattr(model, 'uses_rd', False) else model(x)
                corr1      = x + disp1
                s1         = _post_stats(corr1, disp1)

                # ── K=unroll_steps inference (multi-pass) ──────────────────────
                xk = x
                for _ in range(unroll_steps):
                    dk  = model(xk, rd=rd_model, **model_kwargs) if getattr(model, 'uses_rd', False) else model(xk)
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

                kg_line = ""
                if loss_cfg['name'] in SPH_LOSSES:
                    lp = loss_cfg['params']
                    kg_kwargs = dict(h_factor=lp.get('h_factor', 2.0), box=lp.get('box', 1.0))
                    kg_line = (f"\n  mean_|KG|        = {mean_kg_norm(x, **kg_kwargs).item():.6f}"
                               f"  ->  {mean_kg_norm(corr1, **kg_kwargs).item():.6f}"
                               f"              {mean_kg_norm(xk, **kg_kwargs).item():.6f}")

            rd_value = rd.item()
            secs_per_iter = (time.time() - iter_timer) / eval_cfg['log_interval']
            iter_timer = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            remaining_iters = train_cfg['num_iterations'] - iteration
            set_logger_eta(logger, remaining_iters * secs_per_iter)

            logger.info(
                f"iter={iteration:6d}  {secs_per_iter:.3f}s/iter  lr={current_lr:.2e}  "
                f"loss={total_loss:.6f}\n"
                f"  {'':20s}  {'K=1 (deploy)':>22s}   {'K='+str(unroll_steps)+' (train)':>22s}\n"
                f"  mean_violation   = {mean_viol_pre:.6f}  ->  {s1['mean_viol']:.6f}              {sk['mean_viol']:.6f}\n"
                f"  illegal_pairs    = {ill_pct_pre:.2f}%      ->  {s1['ill_pct']:.2f}%                 {sk['ill_pct']:.2f}%\n"
                f"  viol_per_cloud   = {ill_count_pre:.2f}       ->  {s1['ill_count']:.2f}                  {sk['ill_count']:.2f}\n"
                f"  viol_reduction   = {'':10s}     {viol_red_k1:.1f}%                    {viol_red_kK:.1f}%\n"
                f"  mean_nn_dist     = {mean_nn_pre:.6f}  ->  {s1['mean_nn']:.6f}              {sk['mean_nn']:.6f}\n"
                f"  displacement     = {'':10s}     {s1['disp']:.4f} ({s1['disp']/rd_value:.3f}x rd)   "
                f"{sk['disp']:.4f} ({sk['disp']/rd_value:.3f}x rd)\n"
                f"  correction_eff   = {'':10s}     {eff1:.4f} ({pct1:.1f}% ceil)         {effk:.4f} ({pctk:.1f}% ceil)"
                + kg_line
            )
            # CSV: write K=1 columns (primary), plus K=K illegal% and legal% for tracking
            loss_csv.write(
                f"{iteration},{total_loss:.6f},{current_lr:.2e},"
                f"{mean_viol_pre:.6f},{s1['mean_viol']:.6f},"
                f"{med_viol_pre:.6f},{s1['med_viol']:.6f},"
                f"{ill_pct_pre:.2f},{s1['ill_pct']:.2f},"
                f"{ill_count_pre:.4f},{s1['ill_count']:.4f},{viol_red_k1:.2f},"
                f"{mean_nn_pre:.6f},{s1['mean_nn']:.6f},"
                f"{med_nn_pre:.6f},{s1['med_nn']:.6f},"
                f"{s1['disp']:.6f},{s1['disp']/rd_value:.6f},"
                f"{s1['disp_med']:.6f},{s1['disp_med']/rd_value:.6f},"
                f"{eff1:.6f},{pct1:.2f},"
                f"{sk['ill_pct']:.2f},{sk['ill_count']:.4f},{viol_red_kK:.2f},{pctk:.2f}\n"
            )
            loss_csv.flush()

            # expose for _save_sample (K=1)
            corrected = corr1

        if iteration % eval_cfg['sample_interval'] == 0:
            _save_sample(model, val_noisy, val_rd, device, run_dir, iteration,
                         eval_cfg['num_visual_samples'],
                         rd_model=rd_model, model_kwargs=model_kwargs)
            logger.debug(f"Sample saved at iteration {iteration}")

            val_loss = _evaluate_val_loss(model, val_noisy, val_rd_tensor, rd_model,
                                          model_kwargs, loss_fn, loss_cfg,
                                          unroll_steps, train_cfg['batch_size'], device)
            if val_loss < best_val_loss:
                best_val_loss, best_iteration = val_loss, iteration
                torch.save(model.state_dict(), run_dir / "model_best.pt")
                logger.info(f"New best checkpoint at iter {iteration}: val_loss={val_loss:.6f}")

    loss_csv.close()
    torch.save(model.state_dict(), run_dir / "model_final.pt")
    logger.info(f"Training complete. Model saved to {run_dir / 'model_final.pt'}")
    if best_iteration is not None:
        logger.info(f"Best checkpoint: iter {best_iteration}  val_loss={best_val_loss:.6f}  "
                    f"-> {run_dir / 'model_best.pt'}")

    gif_path = make_sample_gif(run_dir, fps=10, every_n=1)
    logger.info(f"Evolution GIF saved to {gif_path}")


def _evaluate_val_loss(model, val_noisy, rd, rd_model, model_kwargs,
                       loss_fn, loss_cfg, unroll_steps, batch_size, device):
    """Deterministic K=unroll_steps loss on the full fixed validation set, chunked
    to the training batch size. Low-variance alternative to the per-iteration
    log metrics, which are computed on a fresh random training batch each time."""
    model.eval()
    total, n_chunks = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(val_noisy), batch_size):
            chunk = val_noisy[start:start + batch_size]
            x_current  = torch.tensor(chunk, dtype=torch.float32, device=device)
            chunk_loss = torch.tensor(0.0, device=device)
            for _ in range(unroll_steps):
                displacement = (model(x_current, rd=rd_model, **model_kwargs)
                                if getattr(model, 'uses_rd', False) else model(x_current))
                x_next = x_current + displacement
                chunk_loss = chunk_loss + loss_fn(x_current, x_next, rd, **loss_cfg['params'])
                x_current = x_next
            total += chunk_loss.item()
            n_chunks += 1
    model.train()
    return total / n_chunks


def _save_sample(model, val_noisy, rd_value, device, run_dir, iteration,
                 num_visual_samples, rd_model=None, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    model.eval()
    with torch.no_grad():
        batch = val_noisy[:num_visual_samples]
        x = torch.tensor(batch, dtype=torch.float32).to(device)
        rd_tensor = rd_model if rd_model is not None else torch.tensor(
            rd_value, dtype=torch.float32, device=device)
        displacement = model(x, rd=rd_tensor, **model_kwargs) if getattr(model, 'uses_rd', False) else model(x)
        corrected = (x + displacement).cpu().numpy()

    plot_comparison(
        noisy_clouds=batch,
        corrected_clouds=corrected,
        rd=rd_value,
        save_path=run_dir / "samples" / f"sample_{iteration:06d}.png",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config",   required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--loss-config",    required=True)
    parser.add_argument("--model-config",   required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="torch seed for model init; overrides the train config. "
                             "Use to replicate an arm across initialisations.")
    args = parser.parse_args()
    train(
        train_config_path=args.train_config,
        dataset_config_path=args.dataset_config,
        loss_config_path=args.loss_config,
        model_config_path=args.model_config,
        seed=args.seed,
    )
