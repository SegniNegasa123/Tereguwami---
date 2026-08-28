"""
Training Pipeline for Few-Shot Metric Personalization (§8.6, §16)
Part of Tereguwami (ተርጓሚ) Model Training Infrastructure

Trains the Siamese / Prototypical embedding network using triplet loss / contrastive margin loss
so that instances of the same personal sign cluster tightly in the 128-dimensional L2-normalized metric space.
"""

import os
import json
import time
from typing import Dict, List, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class TripletSignDataset(Dataset):
        """Generates (anchor, positive, negative) landmark sequences for metric learning."""
        def __init__(self, num_triplets: int = 50, seq_len: int = 40, feature_dim: int = 1629):
            np.random.seed(101)
            self.triplets = []
            for _ in range(num_triplets):
                base_sign = np.random.randn(seq_len, feature_dim).astype(np.float32)
                anchor = base_sign + np.random.normal(0, 0.05, size=base_sign.shape).astype(np.float32)
                positive = base_sign + np.random.normal(0, 0.05, size=base_sign.shape).astype(np.float32)
                negative = np.random.randn(seq_len, feature_dim).astype(np.float32)
                self.triplets.append((anchor, positive, negative))

        def __len__(self) -> int:
            return len(self.triplets)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            a, p, n = self.triplets[idx]
            return torch.from_numpy(a), torch.from_numpy(p), torch.from_numpy(n)


def train_metric_learner(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    margin: float = 0.5,
    save_checkpoint: bool = True
) -> Dict[str, Any]:
    """Train metric projection head with Triplet Margin Loss."""
    if not HAS_TORCH:
        return {"status": "skipped", "reason": "PyTorch not available in environment"}

    from models.personalization.siamese_few_shot import SiameseEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseEncoder(input_dim=1629, hidden_dim=256, embedding_dim=128)
    model.to(device)

    dataset = TripletSignDataset(num_triplets=48)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        history.append({
            "epoch": epoch + 1,
            "triplet_loss": round(avg_loss, 4)
        })

    checkpoint_path = None
    if save_checkpoint:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(out_dir, "checkpoint_few_shot_metric.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "embedding_dim": 128,
            "final_loss": avg_loss
        }, checkpoint_path)

    return {
        "status": "completed",
        "epochs_trained": epochs,
        "final_loss": history[-1]["triplet_loss"],
        "checkpoint_path": checkpoint_path,
        "history": history
    }


if __name__ == "__main__":
    print("Initiating Few-Shot Metric Learner training...")
    res = train_metric_learner(epochs=2)
    print("Training finished:", res["status"])
    print("Final Triplet Loss:", res.get("final_loss"))
