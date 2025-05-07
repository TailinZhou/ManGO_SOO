import os
import sys
import design_bench
sys.path.append("..")
sys.path.append("../mango")
sys.path.append("../mango/design_baselines/diff")
from typing import Union
from pathlib import Path
import logging
import torch.nn as nn
import pytorch_lightning as pl
from mango.design_baselines.diff.forward import ForwardModel
from mango.design_baselines.diff.trainer import RvSDataModule
def train_surrogate(
    task_name: str,
    ckpt_dir: Union[Path, str] = "../checkpoints/surrogate",
    batch_size: int = 128,
    num_workers: int = 1,
    seed: int = 42,
    device: str = "auto",
    num_epochs: int = 100,
    lr: float = 3e-4,
    hidden_size: int = 2048,
    dropout: float = 0.0,
) -> None:
    """
    Trains and validates a surrogate objective model for the design benchmark
    method for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        batch_size: batch size. Default 128.
        num_workers: number of workers. Default 0.
        seed: random seed. Default 42.
        device: device. Default CPU.
        num_epochs: number of training epochs. Default 100.
        lr: learning rate. Default 0.001.
        hidden_size: hidden size of the model. Default 2048.
        dropout: dropout probability. Default 0.0
    Returns:
        None.
    """
    task = design_bench.make(task_name)
    if task.is_discrete:
        task.map_to_logits()
    if task_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    dm = RvSDataModule(
        task=task,
        val_frac=0.1,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        temp="90"
    )
    model_save_path = os.path.join(ckpt_dir, f"{task_name}-surrogate-{num_epochs}epoch.ckpt")
    if os.path.exists(model_save_path):
        print(f"Loading model from {model_save_path}")
        model = ForwardModel.load_from_checkpoint(model_save_path, taskname=task_name, 
                                                  task=task, hidden_size=hidden_size, learning_rate=lr,
                                                    activation_fn=nn.LeakyReLU(negative_slope=0.2))
        return model
    model = ForwardModel(
        taskname=task_name,
        task=task,
        learning_rate=lr,
        hidden_size=hidden_size,
        activation_fn=nn.LeakyReLU(negative_slope=0.2),
    ) 

    devices = "".join(filter(str.isdigit, device))
    devices = [int(devices)] if len(devices) > 0 else "auto"
    accelerator = device.split(":")[0].lower()
    accelerator = "gpu" if accelerator == "cuda" else accelerator
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=ckpt_dir,
                filename=f"{task_name}-surrogate-{num_epochs}epoch" ,
            )
        ],
        max_epochs=num_epochs,
        logger=False
    )
    trainer.fit(model, dm)
    
    return model

 
 
def mango_train(
    task: design_bench.task.Task,
    task_name: str,
    ckpt_dir: Union[Path, str] = None,
    forward_model: nn.Module = None,
    inverse_model: nn.Module = None,
    batch_size: int = 128,
    num_workers: int = 1,
    seed: int = 42,
    device: str = "auto",
    num_epochs: int = 100,
    lr: float = 1e-3,
    hidden_size: int = 2048,
    dropout: float = 0.0,
    score_matching: bool = False,
    data_preserved_ratio: float = 1.0,
    clip_dic: dict = {'simple_clip': False, 'clip_min': 0.0, 'clip_max': 1.0},
    debais: bool = False,
    augment: bool = False,
    condition_training: bool = True,
) -> None:
    """
    ours_method training for model-based optimization (MBO).
    Input:
        task_name: name of the design-bench MBO task.
        ckpt_dir: directory to saved checkpoints.
        batch_size: batch size. Default 128.
        num_workers: number of workers. Default 0.
        seed: random seed. Default 42.
        device: device. Default CPU.
        num_epochs: number of training epochs. Default 100.
        lr: learning rate. Default 0.001.
        hidden_size: hidden size of the model. Default 2048.
        dropout: dropout probability. Default 0.0
        score_matching: whether to perform score matching. Default False.
    Returns:
        None.
    """

    if ckpt_dir is None:
        ckpt_dir = "./model/ours_method"
    os.makedirs(ckpt_dir, exist_ok=True)
    if task.is_discrete:
        task.map_to_logits()
    if task.dataset_name == os.environ["CHEMBL_TASK"]:
        task.map_normalize_y()

    if inverse_model is None:
        inverse_model = DiffusionSOO(
            taskname=task.dataset_name,
            task=task,
            learning_rate=lr,
            dropout_p=dropout,
            hidden_size=hidden_size,
            simple_clip=clip_dic['simple_clip'],
            clip_min= clip_dic['clip_min'],
            clip_max= clip_dic['clip_max'],
            debias=debais,
            forwardmodel=forward_model,
            augment=augment,
            condition_training=condition_training
        )


    dm = RvSDataModule(
        task=task,
        val_frac=0.1,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        temp="90",
        augment=augment,
    )


    accelerator = device.split(":")[0].lower()
    accelerator = "gpu" if accelerator == "cuda" else accelerator
    # pl.seed_everything(seed)
    trainer = pl.Trainer(
        accelerator=accelerator,
        strategy='dp',
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="elbo_estimator", #
                dirpath=ckpt_dir,
                filename=f"ours-{task.dataset_name}-seed{seed}-hidden{hidden_size}-score_matching{score_matching}"
            )
        ], 
        max_epochs=num_epochs,
        logger=False,
        gpus=1,
    )
    
    print(f"Training ours (x,y) pair diffusion model for {task_name} with seed {seed}")
    trainer.fit(inverse_model, dm)
    logging.info(f"Saved trained our diffusion model to {ckpt_dir}")
    
    return inverse_model
