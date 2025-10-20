# Datasets
Information for dataset related code, logics and explanations.

## Columns: numeric cs categorical

### Numeric (standardize):
* Age, Purchase Amount (USD), Review Rating, Previous Purchases

### Categorical (embed or one-hot):
* Gender, Item Purchased, Category, Location, Size, Color, Season,
Subscription Status, Shipping Type, Discount Applied, Promo Code Used,
Payment Method, Frequency of Purchases

## Fitstatistics & vocab 

```
import csv, math, json
from collections import defaultdict

NUMERIC = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]
CATEG = ["Gender","Item Purchased","Category","Location","Size","Color","Season",
         "Subscription Status","Shipping Type","Discount Applied","Promo Code Used",
         "Payment Method","Frequency of Purchases"]

# Welford’s online mean/std
class OnlineScaler:
    def __init__(self): self.n=0; self.mean=0.0; self.M2=0.0
    def update(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.M2 += d * (x - self.mean)
    def stats(self):
        var = self.M2 / (self.n-1) if self.n>1 else 1.0
        std = math.sqrt(var) if var>0 else 1.0
        return {"mean": self.mean, "std": std}

def fit_transformer(csv_path, encoding="utf-8"):
    scalers = {col: OnlineScaler() for col in NUMERIC}
    vocab = {col: set() for col in CATEG}

    with open(csv_path, newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # numeric
            for col in NUMERIC:
                v = row.get(col, "")
                if v != "":
                    try: scalers[col].update(float(v))
                    except: pass
            # categorical
            for col in CATEG:
                v = (row.get(col, "") or "").strip() or "<UNK>"
                vocab[col].add(v)

    # freeze vocab -> index (reserve 0 for <UNK>)
    cat2idx = {}
    for col in CATEG:
        items = sorted(vocab[col] - {"<UNK>"})
        cat2idx[col] = {"<UNK>": 0}
        for i, val in enumerate(items, start=1):
            cat2idx[col][val] = i

    # freeze scalers
    scaler_stats = {col: scalers[col].stats() for col in NUMERIC}

    return {"numeric": scaler_stats, "categorical": cat2idx}

# Example:
# tf = fit_transformer("shopping_benavior_updated.csv")
# json.dump(tf, open("transform.json","w"))
```

## Stream rows and tensors with an `IterableDataset`

```
import torch
from torch.utils.data import IterableDataset

def make_row_transform(transform_frozen, target="Purchase Amount (USD)"):
    num_stats = transform_frozen["numeric"]
    cat_map = transform_frozen["categorical"]

    def to_example(row):
        # numeric -> z-score
        x_num = []
        for col in NUMERIC:
            v = row.get(col, "")
            if v == "":
                m, s = num_stats[col]["mean"], num_stats[col]["std"]
                x_num.append((0.0 - m) / s)  # or use m as fill
            else:
                val = float(v)
                m, s = num_stats[col]["mean"], num_stats[col]["std"]
                x_num.append((val - m) / s)

        # categorical -> index
        x_cat = []
        for col in CATEG:
            val = (row.get(col, "") or "").strip()
            idx = cat_map[col].get(val, 0)  # 0 = <UNK>
            x_cat.append(idx)

        # target (regression on Purchase Amount)
        y = float(row[target])

        x_num = torch.tensor(x_num, dtype=torch.float32)
        x_cat = torch.tensor(x_cat, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float32)
        return (x_num, x_cat), y

    return to_example

class StreamingCSVDataset(IterableDataset):
    def __init__(self, csv_path, row_transform, encoding="utf-8"):
        self.csv_path = csv_path
        self.row_transform = row_transform
        self.encoding = encoding

    def __iter__(self):
        import csv
        with open(self.csv_path, newline="", encoding=self.encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield self.row_transform(row)
```

## Use it in Jupyter

```python
# 1) Fit and save transforms (run once)
tf = fit_transformer("shopping_benavior_updated.csv")
import json; json.dump(tf, open("transform.json","w"))

# 2) Load & build dataset/loader
tf = json.load(open("transform.json"))
row_tf = make_row_transform(tf, target="Purchase Amount (USD)")

dataset = StreamingCSVDataset("shopping_benavior_updated.csv", row_tf)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# 3) Peek one batch
(X_num, X_cat), y = next(iter(loader))
print("X_num:", X_num.shape)   # [B, 4]
print("X_cat:", X_cat.shape)   # [B, 13]
print("y:", y.shape)           # [B]

# 4) Simple model skeleton (numerical + embeddings)
import torch.nn as nn

# build embedding sizes from vocab
embeddings = nn.ModuleList()
cat_dims = [len(tf["categorical"][c]) for c in CATEG]
emb_dims = [min(50, (d//2)+1) for d in cat_dims]  # rule-of-thumb

for d, e in zip(cat_dims, emb_dims):
    embeddings.append(nn.Embedding(num_embeddings=d, embedding_dim=e))

class TabModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs = embeddings
        in_num = len(NUMERIC)
        in_cat = sum(emb_dims)
        self.mlp = nn.Sequential(
            nn.Linear(in_num + in_cat, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)  # regression
        )

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:,i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x).squeeze(-1)

model = TabModel()
crit = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5) One tiny training step (just to show usage)
(X_num, X_cat), y = next(iter(loader))
pred = model(X_num, X_cat)
loss = crit(pred, y)
opt.zero_grad(); loss.backward(); opt.step()
print("loss:", float(loss))
```

### Notes & Choices


* Standardization: we applied z-score to the 4 numeric columns using online stats so it’s memory-safe.

* Categoricals: mapped to indices with <UNK>=0 for robustness.

* Reusability: we save the transform state (transform.json) and reuse it at inference to ensure the exact same preprocessing.

* Target: I used "Purchase Amount (USD)" as regression; switch to e.g. "Subscription Status" if you want classification (then map Yes/No to {1,0} and change final layer + loss).

## Split dataset iterator

```python
import random

class StreamingCSVDataset(IterableDataset):
    def __init__(self, csv_path, row_transform, split="train", ratio=0.8):
        self.csv_path = csv_path
        self.row_transform = row_transform
        self.split = split
        self.ratio = ratio

    def __iter__(self):
        import csv
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = random.random()
                if self.split == "train" and r < self.ratio:
                    yield self.row_transform(row)
                elif self.split == "test" and r >= self.ratio:
                    yield self.row_transform(row)

```

Usage:
```python 
train_ds = StreamingCSVDataset("shopping_benavior_updated.csv", row_tf, split="train", ratio=0.8)
test_ds = StreamingCSVDataset("shopping_benavior_updated.csv", row_tf, split="test", ratio=0.8)
```

