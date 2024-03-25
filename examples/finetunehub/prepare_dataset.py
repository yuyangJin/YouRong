import torch
import logging
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.elastic.utils.data import ElasticDistributedSampler
from tqdm import tqdm


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_dataset(rank, world_size, tokenizer, args):
    import datasets
    from trl.trainer import ConstantLengthDataset

    if args is not None:
        batch_size = args.train.batch_size
        seq_len = args.train.seq_len
    else:
        batch_size = 2
        seq_len = 128

    # dataset = datasets.load_dataset(args.dataset_name, data_dir="", split="train")
    dataset = datasets.load_from_disk("/home/jinyuyang/data/stack-exchange-paired.hf")
    num = 1000
    dataset = dataset.train_test_split(
        train_size=batch_size * num, test_size=batch_size * num, seed=42 + rank
    )
    train_data = dataset["train"]
    valid_data = dataset["test"]
    logging.info(f"train dataset size {len(train_data)}")
    logging.info(f"valid dataset size {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    logging.info(
        f"The character to token ratio of the dataset is: {chars_per_token:.2f}"
    )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=seq_len,
        chars_per_token=chars_per_token,
    )

    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=seq_len,
        chars_per_token=chars_per_token,
    )
    # train_sampler = DistributedSampler(
    #     train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    # )
    # valid_sampler = DistributedSampler(
    #     valid_dataset, rank=rank, num_replicas=world_size
    # )    
    train_sampler = ElasticDistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    valid_sampler = ElasticDistributedSampler(
        valid_dataset, rank=rank, num_replicas=world_size
    )
    # train_kwargs = {"batch_size": args.train.batch_size, "sampler": train_sampler}
    # test_kwargs = {"batch_size": args.train.batch_size, "sampler": valid_sampler}
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": batch_size}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **test_kwargs)
    return train_loader, valid_loader, train_sampler, valid_sampler


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logging.basicConfig(level=logging.INFO)
    model_name_or_path = "/mnt/data/zhongrx/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16
    ).to("cuda")
    model.to_bettertransformer()
    train_loader, valid_loader, train_sampler, valid_sampler = prepare_dataset(
        0, 1, tokenizer, None
    )
    cnt = 0
    for batch in train_loader:
        print(
            batch["input_ids"],
            batch["labels"],
            type(batch["input_ids"]),
            type(batch["labels"]),
            batch["input_ids"].shape,
            batch["labels"].shape,
        )
        print("input", tokenizer.decode(batch["input_ids"][0]))
        print("labels", tokenizer.decode(batch["labels"][0]))
        for key in batch.keys():
            batch[key] = batch[key].to("cuda")
        output = model(**batch, use_cache=False)
        print("loss", output["loss"])
        for key in output.keys():
            print(key, output[key].shape)
        cnt += 1
        # for key in batch.keys():
        #     print(key, batch[key].shape)
        #     print(batch[key])
        # break
        if cnt > 12:
            break
    print(len(train_loader))
