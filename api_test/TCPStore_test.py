#!/usr/bin/env python3
import io
import os
import pprint
import sys
import etcd
from datetime import timedelta

import torch.distributed as dist
# from torch.distributed import TCPStore
from torch.distributed.elastic.rendezvous.dynamic_rendezvous import DynamicRendezvousHandler
from torch.distributed.elastic.rendezvous.etcd_rendezvous_backend import EtcdRendezvousBackend
from torch.distributed.elastic.agent.server import WorkerSpec
from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
import torch.multiprocessing as mp

def trainer(args) -> str:
    return "do train"

def main():

    server_store = dist.TCPStore("127.0.0.1", 1234, 2, True, timedelta(seconds=30))
    client_store = dist.TCPStore("127.0.0.1", 1234, 2, False)
    server_store.set("first_key", "first_value")
    res = client_store.get("first_key")
    print(res)


if __name__ == '__main__':
    main()