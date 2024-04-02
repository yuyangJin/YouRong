#!/usr/bin/env python3
import io
import os
import pprint
import sys
import etcd
from datetime import timedelta

import numpy 

import torch.distributed as dist
# from torch.distributed import TCPStore
import torch.distributed.elastic.rendezvous as rdzv
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import C10dRendezvousBackend
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import DynamicRendezvousHandler
from torch.distributed.elastic.rendezvous.etcd_rendezvous_backend import EtcdRendezvousBackend
from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
import torch.multiprocessing as mp


def trainer(args) -> str:
    rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend="gloo", rank=rank, world_size=4)
    print("rank " + str(rank) + " start train")
    
    f = open("device.txt", "r")
    lines = f.readlines()
    device_is_available = int(lines[0])
    f.close()

    num = 100
    if rank == 1 and device_is_available == 0:
        sys.exit(1)
    A = numpy.random.rand(num, num*2)
    B = numpy.random.rand(num*2, num*3)
    C = numpy.matmul(A, B)

    print(C[int(num/2), int(num*3/2)])

    return "finished"

def good_trainer(args) -> str:
    rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend="gloo", rank=rank, world_size=4)
    print("rank " + str(rank) + " start train")
    

    num = 100
    A = numpy.random.rand(num, num*2)
    B = numpy.random.rand(num*2, num*3)
    C = numpy.matmul(A, B)

    print(C[int(num/2), int(num*3/2)])

    return "finished"


def main():
    # dist.init_process_group(backend="gloo")
    # dist.barrier()

    nproc_per_process = 4


    # store = TCPStore("localhost")

    # backend = C10dRendezvousBackend(store, "my_run_id")
    #     rdzv_handler = DynamicRendezvousHandler.from_backend(
    #     run_id="my_run_id",
    #     store=store,
    #     backend=backend,
    #     min_nodes=2,
    #     max_nodes=4
    # )


    start_method='spawn'
    # shared_queue= mp.get_context(start_method).Queue()
    # rdzv = EtcdRendezvous(
    #     client=etcd.Client,
    #     prefix='/torchelastic/p2p',
    #     run_id=1234,
    #     num_min_workers=1,
    #     num_max_workers=2,
    #     timeout=10,
    #     last_call_timeout=30,
    # )

    # store = dist.TCPStore("127.0.0.1", 1234, 2, True)   

    print("========================= Start A Server ============================") 
    server_store = dist.TCPStore("127.0.0.1", 1234, 5, True, timedelta(seconds=30))
    print("========================= Start A Client ============================") 
    client_store = dist.TCPStore("127.0.0.1", 1234, 5, False, timedelta(seconds=10))
    print("========================= Connected ============================") 
    server_store.set("first_key", "first_value")
    res = client_store.get("first_key")
    print(res)

    backend = C10dRendezvousBackend(client_store, "my_run_id")


    # backend = EtcdRendezvousBackend(
    #     client = etcd.Client(),
    #     run_id = "my_run_id"
    # )

    rdzv_handler = DynamicRendezvousHandler.from_backend(
        run_id="my_run_id",
        store=client_store,
        backend=backend,
        min_nodes=1,
        max_nodes=2
    )
    spec = WorkerSpec(
                role="trainer",
                local_world_size=nproc_per_process,
                entrypoint=trainer,
                rdzv_handler=rdzv_handler,
                args=('foobar',)
            )
    agent = LocalElasticAgent(spec, start_method)
    results = agent.run()


    if results.is_failed():
        print("trainer failed")
    else:
        print(f"rank 0 return value: {results.return_values[0]}")
        # prints -> rank 0 return value: do train

if __name__ == '__main__':
    main()