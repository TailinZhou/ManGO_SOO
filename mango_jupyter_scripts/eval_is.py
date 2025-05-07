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
from contextlib import contextmanager

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
        unique_tensor = torch.tensor(unique_values, dtype=torch.float32, device=device)
        indices = torch.randint(0, len(unique_values), (num_samples, X0_shape), device=device)
        X0 = unique_tensor[indices] 
    else:

        x_dim = task.x.shape[-1] * (task.x.shape[-2] if is_discrete else 1)
        y_dim = task.y.shape[-1] if augment else 0
        with fixed_random_seed(seed):
            X0 = torch.randn(num_samples, x_dim + y_dim, device=device)

    return _apply_clipping(X0, clip_dic) if clip_dic['simple_clip'] else X0

def _inference_scaling_step(Xt, sde, t, t_next, delta, sigma, y, task, duplicated_time, guidance_bool=False):
    """The multi-branch expansion steps when scaling"""
    # Expand the dimension for multi-candidate sampling
    batch_size = Xt.size(0)
    feat_dim = Xt.size(-1)
    

    # Calculate the mu. It should be noted that here we have utilized the backpropagation mechanism of PyTorch 
    #   to calculate the derivative of the two-norm instead of using the explicit calculation. In fact, they are equivalent.
    with torch.enable_grad():
        Xt_ = Xt.requires_grad_(True)
        mu = sde.gen_sde.mu(t, Xt_, y, guidance_bool=guidance_bool, guidance_scals1=0, guidance_scals2=1).clone().detach()   # [B,D]
    
    # Generate candidate samples
    dX = delta * mu.unsqueeze(1)          # [B,1,D]
    noise = torch.randn(batch_size, duplicated_time, feat_dim, 
                       device=Xt.device) * (delta**0.5) * sigma.unsqueeze(1)
    candidates = Xt.unsqueeze(1) + dX + noise  # [B,K,D]
 
    # Prepare the time step tensor 
    t_next_tensor = t_next.repeat_interleave(duplicated_time, dim=0).squeeze(-1)  # [B*K]
    # compute score
    flat_candidates = candidates.view(-1, feat_dim)  # [B*K,D]
    scores = sde.gen_sde.a(
        flat_candidates, 
        t_next_tensor,  
        flat_candidates[:, task.x.shape[-1]:]
    )
 
    scores = scores.view(batch_size, duplicated_time, feat_dim)  # [B,K,D]
    std_t_minus_1 = torch.sqrt(sde.gen_sde.base_sde.var(t_next))[0]
    # std_t_minus_1 = std_t_minus_1.unsqueeze(1).repeat(1, duplicated_time, 1)
    mean_t_minus_1 = sde.gen_sde.base_sde.mean_weight(t_next)[0]
    # mean_t_minus_1 = mean_t_minus_1.unsqueeze(1).repeat(1, duplicated_time, 1)
    X0_pred_from_candidates =   (candidates + std_t_minus_1**3 * scores)/mean_t_minus_1
 

    # Select the optimal candidate
    y_target = y.unsqueeze(1).expand(-1, duplicated_time, -1)
    y_preds = X0_pred_from_candidates[..., task.x.shape[-1]:]
    rewards = -torch.norm(y_preds - y_target, dim=-1)  # [B, K] 最大化负距离
    sample_indices  = torch.argmax(rewards, dim=1) #[B]

    # no difference between these two
    # rewards = torch.norm(y_preds - y_target, dim=-1)  # [B, K]
    # weights = torch.softmax(-rewards, dim=-1)  # [B, K]
    # sample_indices  = torch.multinomial(weights, num_samples=1).squeeze(-1)  # [B]
    # print(f"sample_indices: {sample_indices}")
    # return
 
    return candidates[torch.arange(batch_size), sample_indices]


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
    is_discrete: bool = False,
    guidance_scals1 = 0.0,
    guidance_scals2 = 1.0,
) -> Sequence[torch.Tensor]:
 
    # Initialization configuratio
    grad_mask = grad_mask.to(device) if grad_mask is not None else None
    clip_fn = lambda x: _apply_clipping(x, clip_dic) if clip_dic['simple_clip'] else x
    X0 = _initialize_samples(task, num_samples, device, augment, clip_dic, is_discrete, seed=seed, warm_init=False)

    # Generation of processing conditions
    gen_condition = gen_condition.to(device, dtype=torch.float32)
    y = gen_condition.unsqueeze(0).expand(num_samples, -1)
 
    # Initialize the time steps and states
    delta = sde.gen_sde.T.item() / num_steps
    ts = torch.linspace(1, 0, num_steps + 1, device=device) * sde.gen_sde.T.item()
    Xt = X0.clone()
    Xs = []

    # Main sampling cycle
    for i in range(num_steps):
        t = torch.full_like(Xt[:, [0]], ts[i])
        sigma = sde.gen_sde.sigma(t, Xt) #sqrt(beta(t))
        Xt = Xt.detach().clone().requires_grad_(True)
 
        # The correction steps of the Heun method
        if inference_scaling_bool and i% 5== 0 and i < num_steps - 1:
            t_minus_1 = torch.full_like(Xt[:, [0]], ts[i+1])
            Xt = _inference_scaling_step(
                Xt, sde, t, t_minus_1, delta, sigma, y, 
                task, duplicated_time, guidance_bool=guidance
            )
        else:
            with torch.enable_grad():
                Xt_ = Xt.requires_grad_(True)
                mu = sde.gen_sde.mu(t, Xt_, y, guidance_bool=guidance, guidance_scals1=guidance_scals1, guidance_scals2=guidance_scals2)
        
            noise = torch.randn_like(Xt) * (delta**0.5) * sigma
            Xt = Xt + delta * mu + noise
        
        Xt = clip_fn(Xt)
        Xs.append(Xt.cpu())

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
    # X = X[:, :task.x.shape[-1]]
    # if task.is_discrete:
    #     X = X.view(X.size(0), -1, task.x.shape[-1])
    if task.is_normalized_x:
        X[:,:task.x.shape[-1]] = task.denormalize_x(X[:,:task.x.shape[-1]])
    if task.is_normalized_y:
        X[:,task.x.shape[-1]:] = task.denormalize_y(X[:,task.x.shape[-1]:])
    # print(f'the first 5 (no rank) design (normalized_x={task.is_normalized_x}, normalized_y={task.is_normalized_y}) is:\n', X[:1]) 
    designs = X[:,:task.x.shape[-1]].cpu().numpy()[np.newaxis, ...]

    task_original = design_bench.make(task_name) #some bug on task
    scores = task_original.predict(designs[0])[np.newaxis, ...]
    
    preds = surrogate.mlp(X[:,:task.x.shape[-1]]).cpu().numpy()[np.newaxis, ...]
    y_gen = X[:,task.x.shape[-1]:].cpu().numpy()[np.newaxis, ...]
    # preds = []


    # Save optimization results.
    if logging_dir is not None:
        if guidance:
            sign = 'with'
        else:
            sign = 'without'
            description = 'uncond'
        os.makedirs(logging_dir, exist_ok=True)
        np.save(os.path.join(logging_dir, f"all_solution_all_steps_{sign}_guidance_{description}.npy"), diffusion)
        np.save(os.path.join(logging_dir, f"solution_{sign}_guidance_{description}.npy"), designs)
        np.save(os.path.join(logging_dir, f"surrogate_scores_{sign}_guidance_{description}.npy"), preds)
        np.save(os.path.join(logging_dir, f"generated_scores_{sign}_guidance_{description}.npy"), y_gen)
        np.save(os.path.join(logging_dir, f"true_scores_{sign}_guidance_{description}.npy"), scores)
        logging.info(f"Saved experiment results to {logging_dir}")
    logging.info("Optimization complete.")

    solution = {'x': designs[0], 'y': preds[0], 'y_scores':scores[0], 'y_gen':y_gen[0], 'algo': 'Ours'}
    return solution

