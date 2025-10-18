from torch.utils.data import Dataset, DataLoader

class ProteinSetDataset(Dataset):
    """
    Expects each item as:
      {
        "proteins": FloatTensor (Ni, 1024),  # Ni can vary
        "label":    int in [0, 367]
      }
    """
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return it["proteins"], it["label"]

def collate_sets(batch):
    """
    batch: list of (proteins, label)
      proteins: (Ni, 1024)
    Returns:
      x:     (B, Nmax, 1024)
      mask:  (B, Nmax) boolean
      y:     (B,)
    """
    import torch
    prots, labels = zip(*batch)
    B = len(prots)
    Nmax = max(p.shape[0] for p in prots)
    D = prots[0].shape[1]

    x = torch.zeros(B, Nmax, D, dtype=prots[0].dtype)
    mask = torch.zeros(B, Nmax, dtype=torch.bool)
    y = torch.tensor(labels, dtype=torch.long)

    for i, p in enumerate(prots):
        n = p.shape[0]
        x[i, :n] = p
        mask[i, :n] = True
    return x, mask, y

