import os

import GPUtil as GPU
import humanize
import psutil


def print_gpu_info(gpu):
    print(
        "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(
            gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal
        )
    )


def main():
    process = psutil.Process(os.getpid())
    print(
        "Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
        " | Proc size: " + humanize.naturalsize(process.memory_info().rss),
    )
    gpus = GPU.getGPUs()
    for gpu in gpus:
        print_gpu_info(gpu)


if __name__ == "__main__":
    main()
