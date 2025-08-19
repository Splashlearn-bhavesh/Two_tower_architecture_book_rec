import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
from tqdm import tqdm
import numpy as np


def recall_at_k(pred_ranks, k):
    """ pred_ranks: list of ranks for the true positive item (1 = best) """
    return np.mean([1 if r <= k else 0 for r in pred_ranks])


def ndcg_at_k(pred_ranks, k):
    """ NDCG@K: Normalized Discounted Cumulative Gain """
    return np.mean([1 / np.log2(r + 1) if r <= k else 0 for r in pred_ranks])


def train_two_tower(model,
                    train_loader,
                    val_loader,
                    epochs=10,
                    lr=1e-3,
                    weight_decay=1e-5,
                    patience=3,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    checkpoint_dir="checkpoints",
                    checkpoint_name="two_tower_best.pt",
                    eval_k_list=[5, 10]):

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)

        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["labels"].float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["labels"].size(0)
        train_loss /= len(train_loader.dataset)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        user_vecs, book_vecs, labels_list = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(batch)
                loss = criterion(outputs, batch["labels"].float())
                val_loss += loss.item() * batch["labels"].size(0)

                # Use model's helper methods to get embeddings
                u_vec = model.get_user_vec(batch)
                b_vec = model.get_book_vec(batch)

                user_vecs.append(u_vec.cpu())
                book_vecs.append(b_vec.cpu())
                labels_list.append(batch["labels"].cpu())

        val_loss /= len(val_loader.dataset)

        # ---------------- METRICS ----------------
        user_vecs = torch.cat(user_vecs)
        book_vecs = torch.cat(book_vecs)
        labels_list = torch.cat(labels_list)

        sim_matrix = torch.matmul(user_vecs, book_vecs.T)  # [num_users, num_books]
        pred_ranks = []
        for i in range(sim_matrix.size(0)):
            scores = sim_matrix[i]
            true_idx = (labels_list[i] == 1).nonzero(as_tuple=True)[0]
            if len(true_idx) == 0:
                continue
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == true_idx[0]).nonzero(as_tuple=True)[0].item() + 1
            pred_ranks.append(rank)

        metrics = {}
        for k in eval_k_list:
            metrics[f"Recall@{k}"] = recall_at_k(pred_ranks, k)
            metrics[f"NDCG@{k}"] = ndcg_at_k(pred_ranks, k)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        for k in eval_k_list:
            print(f"Recall@{k}: {metrics[f'Recall@{k}']:.4f} | NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")

        # ---------------- EARLY STOPPING ----------------
        if val_loss < best_val_loss:
            print(f"✅ Validation loss improved from {best_val_loss:.4f} → {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(checkpoint_dir, checkpoint_name))
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model
