import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import torch.utils.checkpoint
from diffusers import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

logger = get_logger(__name__, log_level="INFO")

import os
import torch
import argparse

from utils.logger import *
from models import xclip
from datasets.CAP import DADA2KS1
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn

#loss
class NCELearnableTempLoss_vsc(nn.Module):
    """
    Compute contrastive loss: video-(sub,cap)
    """

    def __init__(self):
        super(NCELearnableTempLoss_vsc, self).__init__()
        self.temp=nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self,clip1,clip2,clip3,clip4):
        # assert text_feat.shape[0] == cap_feat.shape[0]
        logit_scale = self.temp.exp()
        logits1 = torch.einsum("bd,bkd->bk", clip1[0], logit_scale * clip1[1])
        logits2 = torch.einsum("bd,bkd->bk", clip2[0], logit_scale * clip2[1])
        logits3= torch.einsum("bd,bkd->bk", clip3[0], logit_scale * clip3[1])
        logits4 = torch.einsum("bd,bkd->bk", clip4[0], logit_scale * clip4[1])
        logits1_2=torch.einsum("bd,bkd->bk", clip1[0], logit_scale * clip2[1])
        logits1_4 = torch.einsum("bd,bkd->bk", clip1[0], logit_scale * clip4[1])
        logits2_1 = torch.einsum("bd,bkd->bk", clip2[0], logit_scale * clip1[1])
        logits2_3=torch.einsum("bd,bkd->bk", clip2[0], logit_scale * clip3[1])
        logits3_2=torch.einsum("bd,bkd->bk", clip3[0], logit_scale * clip3[1])
        logits3_4 = torch.einsum("bd,bkd->bk", clip3[0], logit_scale * clip4[1])
        logits4_1 = torch.einsum("bd,bkd->bk", clip4[0], logit_scale * clip1[1])
        logits4_3 = torch.einsum("bd,bkd->bk", clip4[0], logit_scale * clip3[1])
        diag = torch.eye(logits1 .shape[0], dtype=torch.bool).to(logits1 .device)
        logits1_pos=logits1[diag].reshape(logits1.shape[0], 1)
        logits2_pos = logits2[diag].reshape(logits2.shape[0], 1)
        logits3_pos = logits3[diag].reshape(logits3.shape[0], 1)
        logits4_pos = logits4[diag].reshape(logits4.shape[0], 1)
        logits1_2_pos = logits1_2[diag].reshape(logits1_2.shape[0], 1)
        logits1_4_pos = logits1_4[diag].reshape(logits1_4.shape[0], 1)
        logits2_1_pos = logits2_1[diag].reshape( logits2_1.shape[0], 1)
        logits2_3_pos = logits2_3[diag].reshape(logits2_3.shape[0], 1)
        logits3_2_pos = logits3_2[diag].reshape(logits3_2.shape[0], 1)
        logits3_4_pos = logits3_4[diag].reshape(logits3_4.shape[0], 1)
        logits4_1_pos = logits4_1[diag].reshape(logits4_1.shape[0], 1)
        logits4_3_pos = logits4_3[diag].reshape(logits4_3.shape[0], 1)

        v2t = torch.cat([ logits1_pos, logits1_2_pos,logits1_4_pos], dim=1)
        v2t1 = torch.cat([logits2_pos, logits2_1_pos ,  logits2_3_pos ], dim=1)
        v2t2 = torch.cat([logits3_pos, logits3_2_pos , logits3_4_pos], dim=1)
        v2t3 = torch.cat([logits4_pos,logits4_1_pos, logits4_3_pos ], dim=1)
        # v2t_2 = torch.cat([v2t_pos_2, v2t_neg, v2t_neg_2], dim=1)
        v2t_label = torch.zeros(v2t.shape[0], dtype=torch.long).to(v2t.device)
        #
        loss = (F.cross_entropy(v2t, v2t_label) + \
                + F.cross_entropy(v2t1, v2t_label) + \
                + F.cross_entropy(v2t2, v2t_label) + \
                +F.cross_entropy(v2t3, v2t_label)).mean()
        return loss




def main(
    pool: str,
    output_dir: str,
    root_path: str,
    pretrain_clip_path: str,
    validation_steps: int = 100,
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    NUM_FRAMES: int=16,
    max_grad_norm: float = 10.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    MODEL_ARCH: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
):
    # *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    log_dir=r""
    writer=SummaryWriter("log_0.1")
    # Make one log on every process with the configuration for debugging.
    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
#you have to make sure that the accelerator has the same device.
    device=torch.device("cuda",0)
    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        # OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    writer = SummaryWriter("log_0.1")

    model, _ = xclip.load(None, MODEL_ARCH,
                          device="cuda", jit=False,
                          T=NUM_FRAMES,
                          droppath=0.5,
                          use_checkpoint=False,
                          use_cache=False,
                          logger=logger,
                          )
    model.requires_grad_(True)
    model=model.to(device)

    # tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_clip_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrain_clip_path, subfolder="text_encoder")

    def generates_text(data):
        classes = torch.cat([tokenizer(c, max_length=77, padding="max_length",
                                       truncation=True,
                                       return_tensors="pt"
                                       ).input_ids[0] for c in data])
        return classes


    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    train_dataset = DADA2KS1(root_path=root_path, interval=1, phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        pin_memory=True, drop_last=True)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model,optimizer, train_dataloader, lr_scheduler,text_encoder = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler,text_encoder)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_depthprecision == "bf16":
        weight_dtype = torch.bfloat16
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers("accident prediction")
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    global_step = 0
    first_epoch = global_step // num_update_steps_per_epoch
    resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(0 ,10):
        model.train()
        train_loss = 0.0
        eps=1e-5

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            with accelerator.accumulate(model):
                device=torch.device("cuda",0)
                nv = batch["nv"].to(device,dtype=weight_dtype)
                rv = batch["rv"].to(device,dtype=weight_dtype)
                rv_reverse=batch["rv_reverse"].to(device,dtype=weight_dtype)
                av = batch["av"].to(device,dtype=weight_dtype)
                N_t = batch["N_t"]
                R_t = batch["R_t"]
                P_t = batch["P_t"]
                C_t = batch["C_t"]
                N_t = generates_text(N_t).to(device)
                R_t = generates_text(R_t).to(device)
                P_t = generates_text(P_t).to(device)
                C_t = generates_text(C_t).to(device)
                N_t=text_encoder(N_t)[0]
                R_t=text_encoder(R_t)[0]
                P_t=text_encoder(P_t)[0]
                C_t=text_encoder(C_t)[0]
                clip1 = model(nv, N_t)
                clip2 = model(rv, R_t)
                clip3 = model(rv_reverse, P_t)
                clip4 = model(av, C_t)
                lossss=NCELearnableTempLoss_vsc()
                losss=lossss(clip1,clip2,clip3,clip4)
                optimizer.zero_grad()
                accelerator.backward(losss)
                writer.add_scalar('Loss/train',losss,global_step)
                # print("steps.{}".format(losss), losss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad()
                logs = {"step_loss": losss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
#
#             # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
#
                if global_step % checkpointing_steps== 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        # save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
        if global_step >=max_train_steps:
            writer.close()
            break

# # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(text_encoder)
        accelerator.save(unet.state_dict(),save_path)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unwrap", type=str, default=None)
    parser.add_argument("--config", type=str, default="./configs/A_CLIP/A_CLIP.yaml")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    args = parser.parse_args()
    main(**OmegaConf.load(args.config))
