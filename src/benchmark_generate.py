from logging import getLogger

import hydra
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../configs", config_name="generate.yaml")
def main(cfg):
    logger.info("Load model")
    model_name = cfg.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    total_gpu_mem = 0
    for i in range(torch.cuda.device_count()):
        gpu_mem = torch.cuda.memory_allocated(i) / 1024**3
        total_gpu_mem += gpu_mem
        logger.info(f"cuda({i}): {gpu_mem} GB")
    logger.info(f"total gpu memory: {total_gpu_mem:.2f} GB")
    logger.info(f"memory usage: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB")

    promt = "The highest mountain in the world is"

    input_ids = tokenizer.encode(promt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=100, do_sample=True)
    output_sequence = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"Input: {promt}")
    print(f"Output: {output_sequence}")


if __name__ == "__main__":
    main()
