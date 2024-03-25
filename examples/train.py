import sys
import os
# proj_dir = '/home/jinyuyang/PACMAN_PROJECT/RESEARCH/YouRong'
proj_dir = '/home/jinyuyang/PACMAN_PROJECT/acdamic/YouRong/YouRong'
sys.path.append(proj_dir + r"")
import time
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

import numpy as np

from peft import LoraConfig, TaskType, get_peft_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy

import torch.distributed as dist
import torch.multiprocessing as mp

import logging



import oft

class TmpOutput:
    def __init__(self, x) -> None:
        self.logits = x

class MyModel(torch.nn.Module):
    def __init__(self, in_len):
        super().__init__()
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(in_len, in_len, bias=False) for i in range(10)]
        )
        for it, item in enumerate(self.linears):
            if it % 2 == 0:
                item.weight.requires_grad = False
            # print(item.weight.requires_grad)

    def forward(self, x, labels, use_cache):
        x = x.to(torch.float32)
        # print('torch.cuda.memory_allocated():', torch.cuda.memory_allocated())
        for linear in self.linears:
            x = linear(x)
            # print('torch.cuda.memory_allocated():', torch.cuda.memory_allocated())
            # print(x.shape)
        # x = x.reshape([x.shape[0], x.shape[1], 1]).expand(-1, -1, 32)
        return TmpOutput(x)


def run_train(num_iter, batch_size, seq_len, model, loss_func, optimizer, rank):
    t_overall_0 = time.time()
    for i in range(num_iter):
        torch.cuda.synchronize(rank)
        dist.barrier()
        t0 = time.time()
        input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(rank)
        output = model(
            input_ids,
            labels=input_ids,
            use_cache=False,  # reduce
        )
        input_ids = input_ids.long()
        loss = loss_func(
            # output_t,
            output.logits.view(-1, output.logits.size(-1)),
            # output.logits.view(-1),
            # input_ids,
            input_ids.view(-1),
        )
        del output
        torch.cuda.synchronize(rank)
        tb0 = time.time()
        dist.barrier()
        tmid = time.time()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if rank == 0:
        #     logging.info(loss.item())  # print loss for validation
        # below two are **important** for freeing the memory
        del loss
        torch.cuda.synchronize(rank)
        tb1 = time.time()
        dist.barrier()
        t1 = time.time()
        if rank == 0:
            logging.info(
                f"rank {rank} iter {i} fwd {tmid - t0} bwd {t1 - tmid} time {t1 - t0}"
            )
    t_overall_1 = time.time()
    if rank == 0 and num_iter > 2:
        logging.info(
            f"overall_tput {batch_size * num_iter / (t_overall_1 - t_overall_0)} batch_size {batch_size} seq_len {seq_len} num_iter {num_iter}"
        )

# def show_module_structure(module, indent=0, handles=[], modules=[], verbose=False):
#     indent_str = " " * indent
#     # next_indent_str = " " * (indent + 2)
#     has_module = False
#     for module_name, submodule in module.named_children():
#         if verbose:
#             grad_str = (
#                 submodule.requires_grad if hasattr(submodule, "requires_grad") else None
#             )
#             logging.info(f"{indent_str + module_name} {grad_str}")
#         if isinstance(submodule, FSDP):
#             handles.extend(submodule._handles)
#             modules.append(submodule)
#         handles, modules = show_module_structure(
#             submodule, indent + 2, handles, modules, verbose
#         )
#         has_module = True
#     if not has_module:
#         if verbose:
#             for param_name, param in module.named_parameters():
#                 # logging.info(indent_str + param_name, param.requires_grad)
#                 grad_str = (
#                     param.requires_grad if hasattr(param, "requires_grad") else None
#                 )
#                 logging.info(f"{indent_str + param_name} {grad_str}")
#     return handles, modules

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "25900"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def yr_main(rank, world_size, configs, log_queue, log_dir):

    oft.setup_worker_logging(rank, log_queue)

    setup(rank, world_size)


    if configs.validation:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True


    num_iter = configs.train.iter
    batch_size = configs.train.batch_size
    # need_max_batch_size = False
    # if batch_size == "max":
    #     batch_size = 1
    #     need_max_batch_size = True
    seq_len = configs.train.seq_len
    model_name = configs.model
    assert model_name in [
        "facebook_opt_125m",
        "facebook_opt_6.7b",
        "facebook_opt_30b",
        "simple",
        "/mnt/data/zhongrx/Llama-2-13b-hf",
        "/mnt/data/zhongrx/Llama-2-7b-hf",
        "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
        "/data/dataset/Llama-2-70b-hf-trans",
    ]


    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers

    if not configs.validation:
        # Skip model initilization
        transformers.PreTrainedModel._initialize_weights = lambda x, *configs, **kwargs: x
        torch.nn.init.normal_ = lambda x, *configs, **kwargs: x
        torch.nn.init.uniform_ = lambda x, *configs, **kwargs: x
        torch.nn.init.xavier_normal_ = lambda x, *configs, **kwargs: x
        torch.nn.init.xavier_uniform_ = lambda x, *configs, **kwargs: x
        torch.nn.init.kaiming_normal_ = lambda x, *configs, **kwargs: x
        torch.nn.init.kaiming_uniform_ = lambda x, *configs, **kwargs: x


    torch.cuda.set_device(rank)
    torch.distributed.barrier()
    tokenizer = None

    if model_name == "simple":
        model = MyModel(seq_len).to("cuda")
    else:
        if "opt" in model_name:
            model = oft.FTOPTForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, groups=configs.group_size
            )
        elif "Llama" in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to('cuda')

            model = FSDP(model,
                device_id=rank,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
            )


            # model.to_bettertransformer()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            model = get_peft_model(model, peft_config)


            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            assert False, f"{model_name} not supported"

    if tokenizer is not None and configs.with_data:
        # prepare dataset
        train_loader, val_loader, train_sampler, val_sampler = oft.prepare_dataset(
            rank, world_size, tokenizer, configs
        )
    

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    # Start training

    # if tokenizer is not None and configs.with_data:
    #     if rank == 0:
    #         logging.info("start training")
    #     # for epoch in range(args.train.epoch):
    #     #     train_acc = run_train_with_data(
    #     #         model,
    #     #         optimizer,
    #     #         rank,
    #     #         epoch,
    #     #         train_loader,
    #     #         train_sampler,
    #     #         configs.train.batch_size,
    #     #         log_dir.replace("main.log", ""),
    #     #     )
    #     #     if rank == 0:
    #     #         logging.info(f"epoch {epoch} train acc {train_acc}")
    #     #     if epoch % 10 == 0 and epoch != 0:
    #     #         run_eval(num_iter, batch_size, seq_len, model)
    # else:
    logging.info("before hook")

    def hook_fn(m, i, o):
        print(m)
        print("------------Input Grad------------")

        for grad in i:
            try:
                print(grad.shape)
            except AttributeError: 
                print ("None found for Gradient")

        print("------------Output Grad------------")
        for grad in o:  
            try:
                print(grad.shape)
            except AttributeError: 
                print ("None found for Gradient")
        print("\n")


    # torch.nn.modules.module.register_module_forward_hook(hook)
    hook = model.register_backward_hook(hook_fn)

    run_train(num_iter, batch_size, seq_len, model, loss_func, optimizer, rank)

    hook.remove()


@hydra.main(version_base=None, config_path="./configs", config_name="single-train-config")
def main(config: DictConfig):
# def main():
    filename = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/main.log"
    )
    torch.multiprocessing.set_start_method("spawn", force=True)
    log_queue = oft.logger.setup_primary_logging(filename)
    logging.info(config)
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(
        yr_main, args=(WORLD_SIZE, config, log_queue, filename), nprocs=WORLD_SIZE
    )
    logging.info(filename)

if __name__ == "__main__":
    main()
