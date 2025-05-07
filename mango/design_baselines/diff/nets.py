import sys
import os
import math

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Optional, Tuple, Type


@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset
    from design_bench.datasets.discrete.chembl_dataset import ChEMBLDataset
    from design_bench.datasets.discrete.gfp_dataset import GFPDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
    from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset

import numpy as np
import pytorch_lightning as pl

import torch
from torch import optim, nn, utils, Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

from util import TASKNAME2TASK

from lib.sdes import VariancePreservingSDE, PluginReverseSDE, ScorePluginReverseSDE
from unet import UNET_1D


def get_cosine_schedule_with_warmup(optimizer: Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class MLP(nn.Module):

    def __init__(
            self,
            input_dim=2,
            output_dim=1,
            index_dim=1,
            hidden_dim=128,
            condition_training=True,
            act=Swish(),
    ):
        super().__init__()
        self.input_dim = input_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim
        self.act = act
        self.y_dim = output_dim
        self.condition_training = condition_training
        if self.condition_training:
            print("initialization of Condition denoise network")
            self.main = nn.Sequential(
                nn.Linear(input_dim + index_dim + self.y_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, input_dim),
                # nn.Sigmoid() 
            )
        else:
            print("initialization of Uncondition denoise network")
            self.x_net = nn.Sequential(
                nn.Linear(input_dim + index_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, input_dim),
                # nn.Sigmoid() 
            )
            self.y_net = nn.Sequential(
                nn.Linear(input_dim + index_dim, hidden_dim),
                act,
                nn.Linear(hidden_dim, hidden_dim),
            )
        # self.main = nn.Sequential(
        #     nn.Linear(input_dim + index_dim + self.y_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, input_dim),
        # )
        

    def forward(self, input, t, y):
        # init
        sz = input.size()
        # print(input)
        input = input.view(-1, self.input_dim)
        t = t.view(-1, self.index_dim).float()
        # forward
        if self.condition_training:
            y = y.view(-1, self.y_dim).float()
            h = torch.cat([input, t, y], dim=1)
        else:
            h = torch.cat([input, t], dim=1)
        output = self.main(h)  # forward
        # print(output)
        return output.view(*sz)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    
    def forward(self, query, key, value):
        # query: (seq_len_query, batch_size, embed_dim)
        # key: (seq_len_key, batch_size, embed_dim)
        # value: (seq_len_key, batch_size, embed_dim)
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output, attn_weights

 

class CosineTimeEmbedding(nn.Module):
    def __init__(self, index_dim, embed_dim):
        super().__init__()
        self.index_dim = index_dim
        self.embed_dim = embed_dim
        

    def forward(self, t):
        device = t.device  # 获取输入张量的设备
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.embed_dim % 2 == 1:  # 如果embed_dim是奇数,在最后添加一个零
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        
        return emb

class CrossAttenMLP(nn.Module):
    def __init__(self,
            input_dim=2,
            output_dim=1,
            index_dim=1,
            hidden_dim=1024,
            embed_dim=128,
            condition_training=True,
            num_heads=8,
            act=nn.ReLU(),):
        super(CrossAttenMLP, self).__init__()

        self.condition_training = condition_training
        self.input_dim = input_dim
        self.y_dim = output_dim
        self.index_dim = index_dim
        self.hidden_dim = hidden_dim

        # 嵌入层，将x和y映射到相同的嵌入维度
        self.x_embed = nn.Linear(input_dim, embed_dim)
        self.y_embed = nn.Linear(output_dim, embed_dim)
        # self.t_embed = nn.Linear(index_dim, embed_dim)
        self.t_embed = CosineTimeEmbedding(index_dim, embed_dim)
        
        # 交叉注意力层
        self.cross_attention_y = CrossAttention(embed_dim, num_heads)
        self.cross_attention_x = CrossAttention(embed_dim, num_heads)
        
        # 结合后的处理层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 4),
            act,
            nn.Linear(embed_dim * 4, embed_dim),
        )
        # self.fusion = UNET_1D(input_dim=embed_dim * 2,layer_n=3,kernel_size=3,depth=3)
        #MLP
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            # nn.Linear(hidden_dim, hidden_dim),
            # act,
            nn.Linear(hidden_dim, input_dim if self.condition_training else input_dim+output_dim)
        )


        # 编码器
        # self.encoder = nn.Sequential(
        #     nn.Linear(embed_dim , hidden_dim),
        #     act,
        #     # nn.Linear(hidden_dim, hidden_dim),
        #     # act,
        #     nn.Linear(hidden_dim, embed_dim)
        # )
        # 重构层（可以根据具体任务调整）
        # self.decoder_x = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     act,
        #     nn.Linear(hidden_dim, input_dim)
        # )
        
        # self.decoder_y = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     act,
        #     nn.Linear(hidden_dim, output_dim)
        # )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, t, y):
        # x: (batch_size, x_dim)
        # y: (batch_size, y_dim)
        sz = x.size()
        # 嵌入
        if self.condition_training:
            x_emb = self.x_embed(x)  # (batch_size, embed_dim)
            y = y.view(-1, self.y_dim).float()
            y_emb = self.y_embed(y)  # (batch_size, embed_dim)
        else:
            x_emb = self.x_embed(x[:,:self.input_dim])
            y_emb = self.y_embed(x[:, self.input_dim:])
            # 转置以匹配MultiheadAttention的输入要求 (seq_len, batch_size, embed_dim)
        # t = t.view(-1, self.index_dim).float()
        t_emb = self.t_embed(t)  # (batch_size, embed_dim)
        x_emb = x_emb.unsqueeze(0)  # (1, batch_size, embed_dim)
        y_emb = y_emb.unsqueeze(0)  # (1, batch_size, embed_dim)
        # t_emb = t_emb.unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # 交叉注意力：让y作为查询，x作为键和值
        attn_output_y, attn_weights_y = self.cross_attention_y(y_emb, x_emb, x_emb)  # attn_output: (1, batch_size, embed_dim)
        # 交叉注意力：让x作为查询，y作为键和值
        attn_output_x, attn_weights_x = self.cross_attention_x(x_emb, y_emb, y_emb)  # attn_output: (1, batch_size, embed_dim)

        
        # 去除序列维度
        attn_output_y = attn_output_y.squeeze(0)  # (batch_size, embed_dim)
        attn_output_x = attn_output_x.squeeze(0)  # (batch_size, embed_dim)
        # y_emb = y_emb.squeeze(0)  # (batch_size, embed_dim)
        t_emb = t_emb.squeeze(0)  # (batch_size, embed_dim)
        
        # 拼接加权后的x和y的嵌入
        combined = torch.cat([attn_output_y, attn_output_x], dim=1)  # (batch_size, embed_dim * 2)

        # 融合
        fused_xy = self.layer_norm(self.fusion(combined)  + attn_output_y + attn_output_x)  # (batch_size, embed_dim)
        # 编码
        # combined_t = torch.cat([fused_xy, t_emb], dim=1)
        combined_t = fused_xy + t_emb
         
        output = self.MLP(combined_t)
 

        # token = self.encoder(combined_t)  # (batch_size, embed_dim)
        
        # if self.condition_training:
        #     # 重构x和y
        #     x_recon = self.decoder_x(token)  # (batch_size, x_dim)
        #     output = torch.cat([x_recon], dim=1)

        # else:
        #     # 重构x和y
        #     x_recon = self.decoder_x(token)  # (batch_size, x_dim)
        #     y_recon = self.decoder_y(token)  # (batch_size, y_dim)
        #     output = torch.cat([x_recon, y_recon], dim=1)



        return output.view(*sz)
    
class EnhancedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
    
    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)
        q = q + attn_out  # 残差连接
        q = self.norm(q)
        ffn_out = self.ffn(q)
        q = q + ffn_out  # 二次残差
        return self.norm(q)

class EnhancedCrossAttenMLP(nn.Module):
    def __init__(self,
                 input_dim=2,
                 output_dim=1,
                 index_dim=1,
                 hidden_dim=1024,
                 embed_dim=128,
                 condition_training=True,
                 num_heads=8,
                 act=nn.GELU()):
        super().__init__()

        self.condition_training = condition_training
        self.input_dim = input_dim
        self.y_dim = output_dim
        self.hidden_dim = hidden_dim

        # 增强嵌入层
        self.x_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim*4),
            act,
            nn.LayerNorm(embed_dim*4),
            nn.Linear(embed_dim*4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.y_embed = nn.Sequential(
            nn.Linear(output_dim, embed_dim*4),
            act,
            nn.LayerNorm(embed_dim*4),
            nn.Linear(embed_dim*4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.t_embed = CosineTimeEmbedding(index_dim, embed_dim)
        
        # 改进注意力模块
        self.cross_attention_y = EnhancedCrossAttention(embed_dim, num_heads)
        self.cross_attention_x = EnhancedCrossAttention(embed_dim, num_heads)
        
        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim*4),
            act,
            nn.LayerNorm(embed_dim*4),
            nn.Linear(embed_dim*4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 时间特征处理
        self.time_proj = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            act,
            nn.LayerNorm(embed_dim)
        )
        
        # 增强MLP
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act,
            nn.LayerNorm(hidden_dim),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim*2),
            act,
            nn.LayerNorm(hidden_dim*2),
            # nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, input_dim if self.condition_training else input_dim+output_dim)
        )

    def forward(self, x, t, y):
        # 嵌入
        # x_emb = self.x_embed(x)  # [B, E]
        # y_emb = self.y_embed(y)  # [B, E]
        t_emb = self.t_embed(t)  # [B, E]
        if self.condition_training:
            x_emb = self.x_embed(x)  # (batch_size, embed_dim)
            y = y.view(-1, self.y_dim).float()
            y_emb = self.y_embed(y)  # (batch_size, embed_dim)
        else:
            x_emb = self.x_embed(x[:,:self.input_dim])
            y_emb = self.y_embed(x[:, self.input_dim:])


        # 交叉注意力
        attn_y = self.cross_attention_y(
            y_emb.unsqueeze(0), 
            x_emb.unsqueeze(0), 
            x_emb.unsqueeze(0)
        ).squeeze(0)
        
        attn_x = self.cross_attention_x(
            x_emb.unsqueeze(0),
            y_emb.unsqueeze(0),
            y_emb.unsqueeze(0)
        ).squeeze(0)
        
        # 特征融合
        combined = torch.cat([attn_y, attn_x], dim=1)
        fused = self.fusion(combined)
        
        # 时间特征整合
        time_feat = torch.cat([fused, t_emb], dim=1)
        time_feat = self.time_proj(time_feat)
        
        # 最终预测
        output = self.MLP(time_feat)
        return output.view(*x.shape)


class DiffusionTest(pl.LightningModule):

    def __init__(
            self,
            taskname,
            task,
            hidden_size=2048,
            learning_rate=1e-3,
            activation_fn=Swish(),#=Swish(),
            beta_min=0.0001,#不是改这里，改jupyter notebook里的
            beta_max=0.02,
            dropout_p=0,
            simple_clip=False,
            clip_min=None,
            clip_max=None,
            T0=1,
            debias=False,
            vtype='rademacher',
            augment=False,
            condition_training=True,
            ):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.dim_y = 1#self.task.y.shape[-1]
        if not task.is_discrete:
            self.dim_x = self.task.x.shape[-1]
        else:
            self.dim_x = self.task.x.shape[-1] * self.task.x.shape[-2]
        self.dropout_p = dropout_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.simple_clip = simple_clip
        self.debias = debias

        self.clip_min = torch.tensor(task.x).min(axis=0)[0]
        self.clip_max = torch.tensor(task.x).max(axis=0)[0]
        # self.clip_min = torch.tensor(task.dataset.x_min) if clip_min is None  else clip_min
        # self.clip_max = torch.tensor(task.dataset.x_max) if clip_max is None else clip_max
        self.T0 = T0
        self.vtype = vtype

        self.learning_rate = learning_rate
        if augment:
            self.dim_x = self.dim_x + self.dim_y
        self.simple_clip = simple_clip
 

        # self.drift_q = MLP(input_dim=self.dim_x,
                        #    index_dim=1,
                        #     output_dim=self.dim_y,
                        #    hidden_dim=hidden_size,
                        #    act=activation_fn,
                        #    condition_training=condition_training)
        self.embed_dim = 128 
        # self.drift_q = CrossAttenMLP(input_dim= self.task.x.shape[-1] if not task.is_discrete else self.task.x.shape[-1] * self.task.x.shape[-2],
        #                 output_dim=self.dim_y,
        #                 index_dim=1,
        #                 hidden_dim=hidden_size,
        #                 embed_dim=self.embed_dim,
        #                 num_heads=8,
        #                 act=activation_fn,
        #                 condition_training=condition_training)

        self.drift_q = EnhancedCrossAttenMLP(input_dim= self.task.x.shape[-1] if not task.is_discrete else self.task.x.shape[-1] * self.task.x.shape[-2],
                        output_dim=self.dim_y,
                        index_dim=1,
                        hidden_dim=hidden_size,
                        embed_dim=self.embed_dim,
                        num_heads=8,
                        condition_training=condition_training)


        # self.drift_q = UNET_1D(1, 128, 7, 3)
        self.T = torch.nn.Parameter(torch.FloatTensor([self.T0]),
                                    requires_grad=False)

        self.inf_sde = VariancePreservingSDE(beta_min=self.beta_min,
                                             beta_max=self.beta_max,
                                             T=self.T)
        self.gen_sde = PluginReverseSDE(self.inf_sde,
                                        self.drift_q,
                                        self.T,
                                        vtype=self.vtype,
                                        debias=self.debias)

    # def configure_optimizers(self) -> optim.Optimizer:
    #     optimizer = torch.optim.Adam(self.gen_sde.parameters(),
    #                                  lr=self.learning_rate)
    #     return optimizer

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.gen_sde.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        optimizer = torch.optim.AdamW(self.gen_sde.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader()),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 关键参数：按训练步数更新
                "frequency": 1
            }
        }

    # def lr_scheduler_step(self, scheduler, metric):
    #     # 手动执行调度器步进
    #     scheduler.step()

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None,):
        # 使用**kwargs接收额外参数保证版本兼容性
        if metric is not None:  # 处理带监控指标的调度器
            scheduler.step(metric)
        else:  # 处理普通调度器
            scheduler.step()
    def training_step(self, batch, batch_idx, log_prefix="train_batch"):
        # x, y = batch
        x, y, w = batch
        self.clip_min.cuda()
        self.clip_max.cuda()
        if self.dropout_p == 0:
            """
            loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            """
            score_loss, guideloss = self.gen_sde.dsm_weighted(
                x, y, w,
                clip=self.simple_clip,
                c_min=self.clip_min.cuda(),
                c_max=self.clip_max.cuda()) # forward and compute loss
        else:
            # rand_mask = torch.rand(y.size())
            # mask = (rand_mask <= self.dropout_p)
            is_dropout = False
            if torch.rand(1) < self.dropout_p:
            # mask randomly chosen y values，set to -1 as uncondition
                y = 1.0*torch.ones_like(y)
                is_dropout = True
            """
            loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            """
            score_loss, guideloss = self.gen_sde.dsm_weighted(
                x,
                y,
                w,
                is_dropout=is_dropout,
                clip=self.simple_clip,
                c_min=self.clip_min,
                c_max=self.clip_max)   # forward and compute loss
        score_loss = score_loss.mean()
        guideloss = guideloss.mean()
        if log_prefix == "train_batch":
            self.log(f"{log_prefix}_score_loss", score_loss, prog_bar=True)
            if not self.drift_q.condition_training: # when not using our methen, guideloss does not exist
                self.log(f"{log_prefix}_guideloss", guideloss, prog_bar=True)
 

        loss = score_loss  #+ 10000* guideloss 
        return loss
    def validation_step(self, batch, batch_idx, log_prefix="val_batch"):
        x, y, w = batch
        loss_val = self.training_step(batch, batch_idx, log_prefix="val")
        # print(loss_val)
        # self.log(f"val_loss", loss_val, prog_bar=True, on_epoch=True)
        # self.log_dict({'val_loss': loss_val})
        loss_elbo = self.gen_sde.elbo_random_t_slice(x, y)
        self.log(f"elbo_estimator", loss_elbo )#, prog_bar=True
        return    {"val_loss": loss_val, "elbo_estimator": loss_elbo}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        #有一些同步上的问题，只能通过过滤掉一些不合理的值（小于平均值的）来解决
        # avg_loss = torch.stack([x['val_loss'][x['val_loss']>=x['val_loss'].mean()].mean() for x in outputs]) 
        try:
            avg_loss = torch.stack([x['val_loss'][0].mean() for x in outputs]) 
        except:
            avg_loss = torch.stack([x['val_loss'] for x in outputs])
        # print(avg_loss)
        avg_loss = avg_loss.mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        
        # avg_loss_elbo = torch.stack([x['elbo_estimator'][0].mean() for x in outputs]).mean()
        # self.log('avg_elbo_estimator', avg_loss_elbo, prog_bar=True)

class DiffusionScore(pl.LightningModule):

    def __init__(
            self,
            taskname,
            task,
            hidden_size=1024,
            learning_rate=1e-3,
            # activation_fn=nn.ReLU(),
            beta_min=0.1,
            beta_max=20.0,
            dropout_p=0,
            simple_clip=False,
            activation_fn=Swish(),
            T0=1,
            debias=False,
            vtype='rademacher'):
        super().__init__()
        self.taskname = taskname
        self.task = task
        self.learning_rate = learning_rate
        self.dim_y = self.task.y.shape[-1]
        self.dim_x = self.task.x.shape[-1]
        self.dropout_p = dropout_p
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.simple_clip = simple_clip
        self.debias = debias

        self.clip_min = torch.tensor(task.x).min(axis=0)[0]
        self.clip_max = torch.tensor(task.x).max(axis=0)[0]

        self.T0 = T0
        self.vtype = vtype

        self.learning_rate = learning_rate

        self.score_estimator = MLP(input_dim=self.dim_x,
                                   index_dim=1,
                                   hidden_dim=hidden_size,
                                   act=activation_fn)
        self.T = torch.nn.Parameter(torch.FloatTensor([self.T0]),
                                    requires_grad=False)

        self.inf_sde = VariancePreservingSDE(beta_min=self.beta_min,
                                             beta_max=self.beta_max,
                                             T=self.T)
        self.gen_sde = ScorePluginReverseSDE(self.inf_sde,
                                             self.score_estimator,
                                             self.T,
                                             vtype=self.vtype,
                                             debias=self.debias)

    # def configure_optimizers(self) -> optim.Optimizer:
    #     """Configures the optimizer used by PyTorch Lightning."""
    #     optimizer = torch.optim.Adam(self.gen_sde.parameters(), lr=self.learning_rate)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.gen_sde.parameters(),
                                     lr=self.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            # TODO: add to config
            num_warmup_steps=500,
            # num_training_steps=(len(train_dataloader) * config.num_epochs),
            num_training_steps=(10004 * 1000),
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "epoch"}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx, log_prefix="train"):
        # x, y = batch
        x, y, w = batch
        self.clip_min.cuda()
        self.clip_max.cuda()
        if self.dropout_p == 0:
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            loss = self.gen_sde.dsm_weighted(
                x, y, w,
                clip=self.simple_clip).mean()  # forward and compute loss
        else:
            rand_mask = torch.rand(y.size())
            mask = (rand_mask <= self.dropout_p)

            # mask randomly chosen y values
            y[mask] = 0.
            # loss = self.gen_sde.dsm(x, y).mean() # forward and compute loss
            loss = self.gen_sde.dsm_weighted(
                x,
                y,
                w,
                clip=self.simple_clip,
                c_min=self.clip_min,
                c_max=self.clip_max).mean()  # forward and compute loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # loss = self.training_step(batch, batch_idx, log_prefix="val")
        x, y, w = batch
        loss = self.gen_sde.elbo_random_t_slice(x, y)
        self.log(f"elbo_estimator", loss, prog_bar=True)
        return loss



