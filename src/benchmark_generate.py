import time
from logging import getLogger

import hydra
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

logger = getLogger(__name__)
set_seed(0)


@torch.no_grad()
def bench_generate(cfg, model, tokenizer, prompt, device):
    start = time.time()
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=cfg.max_length,
    ).to(device)
    output = model.generate(**input_ids, **cfg.generate)
    output_sequence = tokenizer.batch_decode(output, skip_special_tokens=True)
    end = time.time()
    elapsed = end - start
    n_generated_token = output.numel() - input_ids["input_ids"].numel()
    logger.info(f"Inference Time: {elapsed:.2f} sec")
    logger.info(f"Generate speed: {n_generated_token / elapsed:.2f} token/sec")
    info_max_gpu_memory()

    return output_sequence


def info_max_gpu_memory():
    total_gpu_mem = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.max_memory_reserved(i) / 1024**3
        total_gpu_mem += gpu_mem
        logger.info(f"cuda({i}): {gpu_mem:.2f} GB")
    logger.info(f"total GPU memory: {total_gpu_mem:.2f} GB")


@hydra.main(version_base="1.1", config_path="../configs", config_name="generate.yaml")
def main(cfg):
    device = "cuda" if cfg.model.device_map != "cpu" else "cpu"

    logger.info("---- Load model ----")
    start = time.time()
    model_name = cfg.model.pretrained_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(**cfg.model)
    end = time.time()
    logger.info(f"Model Loading Time: {end - start:.2f} sec")
    logger.info(f"RAM usage: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB")
    info_max_gpu_memory()

    prompt = ["The highest mountain in the world is", "My name is"]

    logger.info("---- Inference ----")
    output_sequence = bench_generate(cfg, model, tokenizer, prompt, device)

    print(f"Input: {prompt}")
    print(f"Output: {output_sequence}")


if __name__ == "__main__":
    main()
