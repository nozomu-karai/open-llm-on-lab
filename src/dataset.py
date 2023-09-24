import json

from torch.utils.data import Dataset


class LimaTestDataset(Dataset):
    def __init__(self, cfg, tokenizer):
        self.cfg = cfg

        self.tokenizer = tokenizer
        self.dataset = self._load_data(cfg.dataset.data_path)

    def _load_data(self, data_path):
        dataset = []
        with open(data_path, "r") as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data["conversations"])

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids = self.tokenizer(
            data,
            return_tensors="pt",
            padding=self.cfg.dataset.padding,
            truncation=True,
            max_length=self.cfg.max_length,
        )

        return input_ids
