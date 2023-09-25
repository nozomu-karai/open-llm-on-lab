import contextlib
import time
from logging import getLogger

import hydra
import psutil
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from dataset import LimaTestDataset

logger = getLogger(__name__)
load_dotenv()


@torch.no_grad()
def bench_generate(cfg, model, tokenizer, dataloader, device):
    output_sequences = []
    n_generated_token = 0
    start = time.time()
    for batch in tqdm(dataloader):
        batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}
        output = model.generate(**batch, **cfg.generate)
        output_sequence = tokenizer.batch_decode(output, skip_special_tokens=True)
        output_sequences.extend(output_sequence)
        n_generated_token += output.numel() - batch["input_ids"].numel()
    end = time.time()
    elapsed = end - start
    logger.info(f"Inference Time: {elapsed:.2f} sec")
    logger.info(f"Generate speed: {n_generated_token / elapsed:.2f} token/sec")
    info_max_gpu_memory()
    logger.info(f"RAM usage: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB")

    return output_sequences


def info_max_gpu_memory():
    total_gpu_mem = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.max_memory_reserved(i) / 1024**3
        total_gpu_mem += gpu_mem
        logger.info(f"cuda({i}): {gpu_mem:.2f} GB")
    logger.info(f"total GPU memory: {total_gpu_mem:.2f} GB")


def maybe_autocast(dtype, device):
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    enable_autocast = device != torch.device("cpu")

    if enable_autocast:
        return torch.cuda.amp.autocast(dtype=dtype)
    else:
        return contextlib.nullcontext()


@hydra.main(version_base="1.1", config_path="../configs", config_name="generate.yaml")
def main(cfg):
    device = "cuda" if cfg.model.device_map != "cpu" else "cpu"
    set_seed(cfg.seed)

    logger.info("---- Load model ----")
    start = time.time()
    model_name = cfg.model.pretrained_model_name_or_path
    if cfg.model.torch_dtype == "float16":
        torch_dtype = torch.float16
    elif cfg.model.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.model.torch_dtype == "float32":
        torch_dtype = torch.float32
    else:
        assert False, f"Invalid torch_dtype: {cfg.model.torch_dtype}"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        load_in_4bit=cfg.model.load_in_4bit,
        load_in_8bit=cfg.model.load_in_8bit,
        device_map=cfg.model.device_map,
    )
    end = time.time()
    logger.info(f"Model Loading Time: {end - start:.2f} sec")
    info_max_gpu_memory()
    logger.info(f"RAM usage: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB")

    with maybe_autocast(dtype=torch_dtype, device=device):
        logger.info("---- Load dataset ----")
        dataset = LimaTestDataset(cfg, tokenizer=tokenizer)
        min_datasize = min(len(dataset), cfg.dataset.datasize)
        sub_dataset = Subset(dataset, list(range(min_datasize)))
        dataloader = DataLoader(sub_dataset, batch_size=cfg.dataset.batch_size, num_workers=cfg.dataset.num_workers)
        logger.info(f"Dataset size: {min_datasize}")

        logger.info("---- Inference ----")
        output_sequences = bench_generate(cfg, model, tokenizer, dataloader, device)
    print(output_sequences[0])


if __name__ == "__main__":
    main()
