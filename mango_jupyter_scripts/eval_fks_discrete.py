import os
import sys
sys.path.append("..")
sys.path.append("../mango")
sys.path.append("../mango/design_baselines/diff")
from contextlib import contextmanager
from helpers import  get_device
import torch
import torch.nn as nn
import numpy as np
import design_bench
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from mango.design_baselines.diff.nets import (
    DiffusionTest, DiffusionScore, Swish
)
import logging
@contextmanager
def fixed_random_seed(seed):
    """
    Context manager, used to fix random seeds and restore the random state when exiting.
    """

    random_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    
    try:
        yield 
    finally:
        torch.random.set_rng_state(random_state)
        
def _apply_clipping(x: torch.Tensor, clip_dic: dict) -> torch.Tensor:
    """Apply tensor clipping"""
    return torch.clamp(
        x,
        min=clip_dic['clip_min'].to(device=x.device, dtype=x.dtype),
        max=clip_dic['clip_max'].to(device=x.device, dtype=x.dtype)
    )

def _initialize_samples(task, num_samples, device, augment, clip_dic, is_discrete, seed=42, warm_init=False) -> torch.Tensor:
    """Initialize the sample tensor"""
    if warm_init:
        if is_discrete:
            X0_shape = task.x.shape[-1] * task.x.shape[-2]
        else:
            X0_shape = task.x.shape[-1]
        if augment:
            X0_shape = X0_shape + task.y.shape[-1]

        unique_values = np.unique(task.x)
        # 转换为 torch.tensor
        unique_tensor = torch.tensor(unique_values, dtype=torch.float32, device=device)
        # 生成随机索引
        indices = torch.randint(0, len(unique_values), (num_samples, X0_shape), device=device)
        # 使用索引获取值
        X0 = unique_tensor[indices] 
    else:

        x_dim = task.x.shape[-1] * (task.x.shape[-2] if is_discrete else 1)
        y_dim = task.y.shape[-1] if augment else 0
        with fixed_random_seed(seed):
            X0 = torch.randn(num_samples, x_dim + y_dim, device=device)

    return _apply_clipping(X0, clip_dic) if clip_dic['simple_clip'] else X0

 
@torch.no_grad()
def heun_sampler(
    task: design_bench.task.Task,
    sde: Union[DiffusionTest, DiffusionScore],
    gen_condition: torch.Tensor,
    num_samples: int = 256,
    num_steps: int = 1000,
    gamma: float = 1.0,
    device: torch.device = torch.device("cuda"),
    grad_mask: Optional[torch.Tensor] = None,
    seed: int = 1000,
    augment: bool = False,
    condition_training: bool = True,
    guidance: bool = False,
    clip_dic: Optional[dict] = None,
    inference_scaling_bool: bool = False,
    duplicated_time: int = 1,
    potential_type : str = "max",
    lmbda = 10.0,
    is_discrete: bool = False,
    guidance_scals1 = 0.0,
    guidance_scals2 = 1.0,
) -> Sequence[torch.Tensor]:

    # Initialize the tensor and configure
    clip_fn = lambda x: _apply_clipping(x, clip_dic) if clip_dic else x
    X0 = _initialize_samples(task, num_samples, device, augment, clip_dic, is_discrete, seed=seed, warm_init=False)
    delta = sde.gen_sde.T.item() / num_steps

    # Handle the generation conditions
    gen_condition = gen_condition.to(device, dtype=torch.float32)
    y = gen_condition.unsqueeze(0).expand(num_samples, -1)
    print_msg = f"Given y: {y[0].cpu().numpy()}" if condition_training else "Unconditional generation"
    # print(print_msg)

    # Estimate the time step
    time_steps = torch.linspace(1, 0, num_steps + 1, device=device) * sde.gen_sde.T.item()
    Xt = X0.clone()
    Xs = []

    # Initialize the scaling related variables
    product_of_potentials = torch.ones(num_samples, device=device)
    population_rs = torch.zeros(num_samples, device=device)

    # Main sampling cycle
    for i in range(num_steps):
        current_time = time_steps[i]
        t = torch.full_like(Xt[:, [0]], current_time).to(Xt.device)
        sigma = sde.gen_sde.sigma(t, Xt)
        Xt = Xt.detach().clone()#.requires_grad_(True)

        with torch.enable_grad():
            Xt_ = Xt.requires_grad_(True)
            mu = sde.gen_sde.mu(t, Xt_, y, guidance_bool=guidance, guidance_scals1=guidance_scals1, guidance_scals2=guidance_scals2)

        # Update Xt and add noise
        noise = torch.randn_like(Xt) * (delta**0.5) * sigma
        Xt = Xt + delta * mu + noise

        if inference_scaling_bool and (i + 1) % 5 == 0:
            t_next = torch.full_like(Xt[:, [0]], time_steps[i + 1]).squeeze(-1) 
            scores = sde.gen_sde.a(Xt, t_next, Xt[:, task.x.shape[-1]:])

            std_next = torch.sqrt(sde.gen_sde.base_sde.var(t_next))[0]
            mean_next = sde.gen_sde.base_sde.mean_weight(t_next)[0]
            # print(f"Mean: {mean_next.cpu().numpy()}")
            X0_pred = (Xt + std_next**3 * scores) / mean_next

            rs_candidates = torch.norm(X0_pred[..., task.x.shape[-1]:] - y, dim=-1)
            rs_candidates = torch.exp(-1 * rs_candidates)
            if potential_type == "max":
                w = torch.exp(lmbda * torch.maximum(rs_candidates, population_rs))
            elif potential_type == "add":
                rs_candidates = rs_candidates + population_rs
                w = torch.exp(lmbda * rs_candidates)
            elif potential_type == "diff":
                diffs = rs_candidates - population_rs
                w =  torch.exp(lmbda * diffs)
            else:
                ValueError(f"Invalid potential_type: {potential_type}")
 
            if (i + 1) == num_steps and potential_type in ("max", "add"):
                w = torch.exp(lmbda * rs_candidates) / product_of_potentials

            w = torch.nan_to_num(w, nan=0.0)
            w = torch.clamp(w, min=0.0)
            normalized_w = w + 1e-8
            normalized_w /= normalized_w.sum()

            if torch.any(w < 0) or not torch.all(torch.isfinite(w)):
                w = torch.ones_like(w) / w.size(0) 
                ValueError(f"Invalid probability distribution: {w.cpu().numpy()}")

            # adaptive_resampling = True
            # if adaptive_resampling:
            #     # compute effective sample size
            #     ess = 1.0 / (normalized_w.pow(2).sum())

            #     if ess > 0.5 * num_samples:
            #         continue

            indices = torch.multinomial(normalized_w, num_samples=num_samples, replacement=True)
            Xt = Xt[indices].detach().clone()
            population_rs = rs_candidates[indices].detach().clone()
            # update product_of_potentials 
            product_of_potentials = product_of_potentials[indices].detach().clone() * w[indices]
            product_of_potentials = torch.clamp(product_of_potentials, min=1e-8)

        Xt = clip_fn(Xt)
        Xs.append(Xt.detach().cpu())

    return Xs

 
 
@torch.no_grad()
def mango_eval(
    task: design_bench.task.Task,
    task_name: str,
    forwardmodel: nn.Module = None,
    inverse_model: nn.Module = None,
    ckpt_dir: Union[Path, str] = "./model/ours",
    logging_dir: Optional[Union[Path, str]] = None,
    num_samples: int = 256,
    num_steps: int = 1000,
    hidden_size: int = 2048,
    seed: int = 42,
    device: str = "auto",
    score_matching: bool = False,
    gamma: float = 1.0,
    clip_dic: dict = {'simple_clip': False, 'clip_min': 0.0, 'clip_max': 1.0},
    gen_condition=None,
    augment: bool = False,
    condition_training: bool = True,
    guidance: bool = False,
    is_discrete: bool = False,
    inference_scaling_bool: bool = False,
    duplicated_time: int = 1,
    guidance_scals1: float = 0.0,
    guidance_scals2: float = 1.0,
    description=None,
) -> None:
    """
    Ours  evaluation for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        logging_dir: optional directory to save logs and results to.
        num_samples: number of samples. Default 2048.
        num_steps: number of integration steps for sampling. Default 1000.
        hidden_size: hidden size of the model. Default 2048.
        seed: random seed. Default 42.
        device: device. Default CPU.
        score_matching: whether to perform score matching. Default False.
        gamma: drift parameter. Default 1.0.
    Returns:
        None.
    """
    device = get_device(device)
    # task = task
    # if task.is_discrete:
    #     task.map_to_logits()
    if gen_condition is None:
        gen_condition = torch.tensor([0]).clone().detach().to(device)
    elif gen_condition.device != device:
        gen_condition = gen_condition.to(device)
    # if task_name == os.environ["CHEMBL_TASK"]:
    #     task.map_normalize_y()

    inverse_model = inverse_model.to(device)
    inverse_model.eval()
    if forwardmodel is not None:
        surrogate = forwardmodel
        surrogate.eval()

    designs, preds, scores = [], [], []
    grad_mask = None

   

    diffusion = heun_sampler(
        task=task,
        sde=inverse_model,
        gen_condition=gen_condition,
        num_samples=num_samples,
        num_steps=num_steps,
        device=device,
        grad_mask=grad_mask,
        augment=augment,
        condition_training=condition_training,
        guidance=guidance,
        clip_dic=clip_dic,
        is_discrete=is_discrete,
        seed=seed,
        inference_scaling_bool=inference_scaling_bool,
        duplicated_time=duplicated_time,
        guidance_scals1=guidance_scals1,
        guidance_scals2=guidance_scals2,
    )
    # print(diffusion)
    idx = -1
    while diffusion[idx].isnan().any():
        idx -= 1
    X = diffusion[idx]
 
    if is_discrete:
        X1 = torch.floor(X[:,:task.x.shape[-1]].detach().cpu())
        X = torch.cat((X1, X[:,task.x.shape[-1]:].detach().cpu()), dim=1)
 
    print(f'the first 5 (no rank) design (normalized_x={task.is_normalized_x}, normalized_y={task.is_normalized_y}) is:\n', X[:1])
    if task.is_normalized_x:
        X[:,:task.x.shape[-1]] = task.denormalize_x(X[:,:task.x.shape[-1]])
    if task.is_normalized_y:
        X[:,task.x.shape[-1]:] = task.denormalize_y(X[:,task.x.shape[-1]:])
        
    if task.is_discrete:
        X_o = X[:, :task.x.shape[-1]*task.x.shape[-2]].view(X.size(0), -1, task.x.shape[-1])
    # print(f'the first 5 (no rank) design (normalized_x={task.is_normalized_x}, normalized_y={task.is_normalized_y}) is:\n', X[:1]) 
    designs = X[:,:task.x.shape[-1]].cpu().numpy()[np.newaxis, ...] if not task.is_discrete else X_o.cpu().numpy()[np.newaxis, ...]
 
    
    # task_original = design_bench.make(task_name) #some bug on task
    scores = task.predict(designs[0])[np.newaxis, ...]
    
    preds = surrogate.mlp(X[:,:task.x.shape[-1]]).cpu().numpy()[np.newaxis, ...] if not task.is_discrete else surrogate.mlp(X_o).cpu().numpy()[np.newaxis, ...]
    y_gen = X[:,task.x.shape[-1]:].cpu().numpy()[np.newaxis, ...] if not task.is_discrete else X[:,task.x.shape[-1]*task.x.shape[-2]:].cpu().numpy()[np.newaxis, ...]
    # preds = []

    # Save optimization results.
    if logging_dir is not None:
        if guidance:
            sign = 'with'
        else:
            sign = 'without'
            description = 'uncond'
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, f"all_solution_all_steps_{sign}_guidance_{description}_seed{seed}.npy"), diffusion)
        np.save(os.path.join(logging_dir, f"solution_{sign}_guidance_{description}_seed{seed}.npy"), designs)
        np.save(os.path.join(logging_dir, f"surrogate_scores_{sign}_guidance_{description}_seed{seed}.npy"), preds)
        np.save(os.path.join(logging_dir, f"generated_scores_{sign}_guidance_{description}_seed{seed}.npy"), y_gen)
        np.save(os.path.join(logging_dir, f"true_scores_{sign}_guidance_{description}_seed{seed}.npy"), scores)
        logging.info(f"Saved experiment results to {logging_dir}")
    logging.info("Optimization complete.")

    solution = {'x': designs[0], 'y': preds[0], 'y_scores':scores[0], 'y_gen':y_gen[0], 'algo': 'Ours'}
    return solution

