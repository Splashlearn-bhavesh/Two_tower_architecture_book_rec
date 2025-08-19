import torch
from torch.utils.data import DataLoader
from src.utils.main_utils import load_json_file
from src.components.data_loader import BookDataset, collate_bookonly_fn
from src.components.model import TwoTowerModel
from tqdm import tqdm 
import faiss
import pickle

def build_and_save_book_embeddings_faiss(model, item_df, book_feature_cols,
                                         batch_size=512, device="cuda" if torch.cuda.is_available() else "cpu",
                                         index_path="book_index.faiss",
                                         id_map_path="book_ids.pkl"):

    fi = load_json_file('model_parmater.json')

    dataset = BookDataset( item_df, book_feature_cols)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_bookonly_fn)


    model = TwoTowerModel(fi['book_feature_count'], fi['user_feature_count'],
                    fi['emb_count'], fi['user_emb_count'],
                    book_feature_dim=fi['book_feature_dim'], user_feature_dim=fi['user_feature_dim'])
    model.eval()
    model = model.to(device)

    all_embeddings = []
    book_ids = []

    with torch.no_grad():
            for batch in tqdm(trainloader, desc="Encoding books"):
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Compute embeddings
                book_vec = model.get_book_vec(batch)

                all_embeddings.append(book_vec.cpu().numpy())
                book_ids.extend(batch["book_code"])  # keep mapping book_id → index

    all_embeddings = np.vstack(all_embeddings).astype("float32")  # [N_books, D]

    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (dot product)
    index.add(all_embeddings)

    # Save index & mapping
    faiss.write_index(index, index_path)
    with open(id_map_path, "wb") as f:
        pickle.dump(book_ids, f)

    print(f"✅ Saved FAISS index with {len(book_ids)} books at {index_path}")
    return index, book_ids