import os
import sys
sys.path.append(r"./")

import shutil
import time
import numpy as np
from datetime import datetime

import hydra
from omegaconf import DictConfig
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.models
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

import torch.distributed.elastic.multiprocessing as mp
# import torch.multiprocessing as mp

import finetunehub as fth

class WorkerState:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """
    def __init__(self, model, optimizer):
        self.epoch = -1
        # self.best_acc1 = 0
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::

        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            # "best_acc1": self.best_acc1,
            # "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
    
    def save(self, f):
        torch.save(self.capture_snapshot(), f)
        print(f"=> saved checkpoint for epoch {self.epoch} at {f}")

def save_checkpoint(state: WorkerState, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job
    # gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    # if is_best:
    #     best = os.path.join(checkpoint_dir, "model_best.pth.tar")
    #     print(f"=> best model found at epoch {state.epoch} saving to {best}")
    #     shutil.copyfile(filename, best)

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
        rank = int(os.environ["LOCAL_RANK"])

        x = x.to(torch.float32).to('cuda:'+str(rank))
        # print('torch.cuda.memory_allocated():', torch.cuda.memory_allocated())
        for linear in self.linears:
            
            x = linear(x).to('cuda:'+str(rank))
            # print('torch.cuda.memory_allocated():', torch.cuda.memory_allocated())
            # print(x.shape)
        # x = x.reshape([x.shape[0], x.shape[1], 1]).expand(-1, -1, 32)
        return TmpOutput(x)



# def setup(rank, world_size):
#     # os.environ["MASTER_ADDR"] = "127.0.0.1"
#     # os.environ["MASTER_PORT"] = "25900"

#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

def run_train(i, batch_size, seq_len, model, loss_func, optimizer, rank, state):

    # torch.cuda.synchronize(rank)
    # dist.barrier()
    t0 = time.time()
    input_ids = torch.randint(0, 1024, (batch_size, seq_len)).to(rank)
    # input_ids = torch.randint(0, 1024, (batch_size, seq_len))
    output = model(
        input_ids,
        labels=input_ids,
        use_cache=False,  # reduce
    )
    input_ids = input_ids.long()
    # input_ids = input_ids.float()
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
    # dist.barrier()
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
    # dist.barrier()
    t1 = time.time()
    if rank == 0:
        logging.info(
            f"rank {rank} iter {i} fwd {tmid - t0} bwd {t1 - tmid} time {t1 - t0}"
        )
    


def procs_main(configs, log_queue, dirname):

    import gc
    del variable
    gc.collect()
    # setup(rank, world_size)
    # print("Rank is:", rank, sep=' ')
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(rank, world_size)
    fth.setup_worker_logging(rank, log_queue)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if configs.validation:
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

    num_iter = configs.train.iter
    batch_size = configs.train.batch_size

    seq_len = configs.train.seq_len
    model_name = configs.model
    assert model_name in [
        "facebook_opt_125m",
        "facebook_opt_6.7b",
        "facebook_opt_30b",
        "simple",
        "/mnt/data/zhongrx/Llama-2-13b-hf",
        "/mnt/data/zhongrx/Llama-2-7b-hf",
        "/data/dataset/Llama-2-70b-hf-trans",
        "/home/dataset/llama-2-hf-all/Llama-2-7b-hf",
        "/mnt/octave/mnt/data/zhongrx/Llama-2-7b-hf",
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
    
    # torch.distributed.barrier()
    tokenizer = None

    from peft import LoraConfig, TaskType, get_peft_model


    if model_name == "simple":
        model = MyModel(seq_len).to("cuda:"+str(rank))
    else:
        if "opt" in model_name:
            model = fth.FTOPTForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, groups=configs.group_size
            )
        elif "Llama" in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to("cuda:"+str(rank))

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
        train_loader, val_loader, train_sampler, val_sampler = fth.prepare_dataset(
            rank, world_size, tokenizer, configs
        )

    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_func = torch.nn.CrossEntropyLoss()

    state = WorkerState(model, optimizer)


    model.train()

    ckpt_filename = dirname+'/'+ str(rank) +'/ckpt'
    print(ckpt_filename)

    for i in range(num_iter):
        state.epoch = i
        run_train(i, batch_size, seq_len, model, loss_func, optimizer, rank, state)

        tic = time.perf_counter()
        # save_checkpoint(state, ckpt_filename)
        model.to('cpu')
        toc = time.perf_counter()
        save_time = toc - tic
        logging.info(f"Save checkpoint in {save_time:0.4f} seconds")
        model.to("cuda:"+str(rank))
        tic = time.perf_counter()
        logging.info(f"back checkpoint in {(tic-toc):0.4f} seconds")

    # if rank == 0 and num_iter > 2:
    #     logging.info(
    #         f"overall_tput {batch_size * num_iter / (t_overall_1 - t_overall_0)} batch_size {batch_size} seq_len {seq_len} num_iter {num_iter}"
    #     )

@hydra.main(version_base=None, config_path="./configs", config_name="single-train-config")
def main(config: DictConfig):

    rank = int(os.environ["LOCAL_RANK"])

    dirname = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    filename = (
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/main.log"
    )
    # # torch.multiprocessing.set_start_method("spawn", force=True)
    log_queue = fth.logger.setup_primary_logging(filename)
    logging.info(config)
    # WORLD_SIZE = torch.cuda.device_count()
    # # mp.spawn(
    # #     procs_main, args=(WORLD_SIZE, config, log_queue, filename), nprocs=WORLD_SIZE
    # # )


    # logging.info(filename)

    # dir_name = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dir_name = os.getcwd()+'/outputs/' + str(rank) + '/' + str(datetime.timestamp(datetime.now()))
    print(dir_name)

    os.makedirs(dir_name)

    procs_main(config, log_queue, dirname)

    logging.info(filename)

    # ctx = mp.start_processes(
    #     name="trainer",
    #     entrypoint=procs_main,
    #     args={0:0,1:1},
    #     envs={0: {"LOCAL_RANK": 0}, 1: {"LOCAL_RANK": 1}},
    #     log_dir=dir_name,
    #     # start_method='spawn',
    #     # redirects=mp.Std.ALL, # write all worker stdout/stderr to a log file
    #     # tee={0: mp.Std.ERR}, # tee only local rank 0's stderr to console
    #   )
    # ctx.wait()

    # torch.multiprocessing.set_start_method("spawn", force=True)
    # WORLD_SIZE = torch.cuda.device_count()
    # print(WORLD_SIZE)
    # mp.spawn(
    #     procs_main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE
    # )

if __name__ == "__main__":
    main()
