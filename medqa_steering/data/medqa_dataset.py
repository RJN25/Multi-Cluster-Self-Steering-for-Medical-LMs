from datasets import load_dataset
from torch.utils.data import Dataset

LETTER = ["A","B","C","D"]

class MedQADataset(Dataset):
    """
    Exposes items with:
      - stem: main question (sent1)
      - choices: list[str] length 4 (ending0..3)
      - label: int in {0,1,2,3}
      - qid: str id
    """
    def __init__(self, split="train", hf_name="GBaker/MedQA-USMLE-4-options-hf"):
        ds = load_dataset(hf_name, split=split)
        self.ids = ds["id"]
        self.stems = ds["sent1"]
        self.choices = list(zip(ds["ending0"], ds["ending1"], ds["ending2"], ds["ending3"]))
        self.labels = ds["label"]

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        return dict(
            qid=self.ids[i],
            stem=self.stems[i],
            choices=list(self.choices[i]),
            label=int(self.labels[i]),
        )
