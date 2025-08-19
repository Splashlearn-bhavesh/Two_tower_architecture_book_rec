import torch
import torch.nn as nn

def masked_mean(embeddings, mask):
    """
    embeddings: [B, L, D]
    mask: [B, L] (1 where valid, 0 where padded)
    Returns: [B, D]
    """
    summed = (embeddings * mask.unsqueeze(-1)).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return summed / counts


class BookTower(nn.Module):
    def __init__(self, book_feature_count,emb_count, book_feature_dim):
        super().__init__()
        
        # Book ID embeddings
        self.theme_emb = nn.Embedding(book_feature_count['themes_count'], emb_count['themes_count'] , padding_idx=0)
        self.category_emb = nn.Embedding(book_feature_count['category_count'], emb_count['category_count'] , padding_idx=0)
        self.reading_skill_emb = nn.Embedding(book_feature_count['reading_skills_count'], emb_count['reading_skills_count'] , padding_idx=0)
        self.grades_emb = nn.Embedding(book_feature_count['grade_count'], emb_count['grade_count'] , padding_idx=0)
        self.book_code_emb = nn.Embedding(book_feature_count['book_count'], emb_count['book_count'] , padding_idx=0)
        
        self.embedding_dim = emb_count['themes_count'] + emb_count['category_count'] + emb_count['reading_skills_count'] + emb_count['grade_count'] + emb_count['book_count']
        # Projection layer
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_dim  + book_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
    
    def forward(self, theme_ids, theme_mask, 
                category_ids, category_mask, 
                reading_skill_ids, reading_skill_mask, 
                grades_ids, grades_mask, 
                book_code_ids, book_code_mask, 
                book_features):
        
        theme_vec = masked_mean(self.theme_emb(theme_ids), theme_mask)
        category_vec = masked_mean(self.category_emb(category_ids), category_mask)
        reading_skill_vec = masked_mean(self.reading_skill_emb(reading_skill_ids), reading_skill_mask)
        grades_vec = masked_mean(self.grades_emb(grades_ids), grades_mask)
        book_code_vec = masked_mean(self.book_code_emb(book_code_ids), book_code_mask)
        
        x = torch.cat([theme_vec, category_vec, reading_skill_vec, grades_vec, book_code_vec, book_features], dim=1)
        return self.fc(x)


class UserTower(nn.Module):
    def __init__(self, user_feature_count, user_emb_count, user_feature_dim):
        super().__init__()

        # All categorical features as embeddings
        self.last_book_emb = nn.Embedding(user_feature_count['book_count'], user_emb_count['book_count'], padding_idx=0)
        self.last_theme_emb = nn.Embedding(user_feature_count['themes_count'], user_emb_count['themes_count'], padding_idx=0)
        self.last_category_emb = nn.Embedding(user_feature_count['category_count'], user_emb_count['category_count'], padding_idx=0)
        self.last_reading_skill_emb = nn.Embedding(user_feature_count['reading_skills_count'], user_emb_count['reading_skills_count'], padding_idx=0)

        self.country_emb = nn.Embedding(user_feature_count['country_count'], user_emb_count['country_count'], padding_idx=0)
        self.state_emb = nn.Embedding(user_feature_count['state_count'], user_emb_count['state_count'], padding_idx=0)
        self.zipcode_emb = nn.Embedding(user_feature_count['zipcode_count'], user_emb_count['zipcode_count'], padding_idx=0)
        self.teacher_emb = nn.Embedding(user_feature_count['teacher_count'], user_emb_count['teacher_count'], padding_idx=0)
        self.school_emb = nn.Embedding(user_feature_count['school_count'], user_emb_count['school_count'], padding_idx=0)

        # Compute total embedding dim dynamically
        total_emb_dim = sum(user_emb_count.values())

        self.fc = nn.Sequential(
            nn.Linear(total_emb_dim + user_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self,
                last_book_ids, last_book_mask,
                last_theme_ids, last_theme_mask,
                last_category_ids, last_category_mask,
                last_reading_skills_id, last_reading_skills_mask,
                country_ids, country_mask,
                state_ids, state_mask,
                zipcode_ids, zipcode_mask,
                teacher_ids, teacher_mask,
                school_ids, school_mask,
                user_features):

        # Apply masked mean pooling to all categorical features
        last_book_vec = masked_mean(self.last_book_emb(last_book_ids), last_book_mask)
        last_theme_vec = masked_mean(self.last_theme_emb(last_theme_ids), last_theme_mask)
        last_category_vec = masked_mean(self.last_category_emb(last_category_ids), last_category_mask)
        last_reading_skill_vec = masked_mean(self.last_reading_skill_emb(last_reading_skills_id), last_reading_skills_mask)

        country_vec = masked_mean(self.country_emb(country_ids), country_mask)
        state_vec = masked_mean(self.state_emb(state_ids), state_mask)
        zipcode_vec = masked_mean(self.zipcode_emb(zipcode_ids), zipcode_mask)
        teacher_vec = masked_mean(self.teacher_emb(teacher_ids), teacher_mask)
        school_vec = masked_mean(self.school_emb(school_ids), school_mask)

        x = torch.cat([
            last_book_vec, last_theme_vec, last_category_vec, last_reading_skill_vec,
            country_vec, state_vec, zipcode_vec, teacher_vec, school_vec,
            user_features
        ], dim=1)

        return self.fc(x)
    

class TwoTowerModel(nn.Module):
    def __init__(self,
                 book_feature_count, user_feature_count,
                 emb_count, user_emb_count, book_feature_dim, user_feature_dim):
        super().__init__()

        self.book_tower = BookTower(book_feature_count, emb_count, book_feature_dim)
        self.user_tower = UserTower(user_feature_count, user_emb_count, user_feature_dim)

    def forward(self, batch):
        book_vec = self.get_book_vec(batch)
        user_vec = self.get_user_vec(batch)
        scores = (user_vec * book_vec).sum(dim=1)  # Dot product
        return scores

    def get_book_vec(self, batch):
        return self.book_tower(
            batch["theme_ids"], batch["theme_mask"],
            batch["category_ids"], batch["category_mask"],
            batch["reading_skill_ids"], batch["reading_skill_mask"],
            batch["grades_ids"], batch["grades_mask"],
            batch["book_code_ids"], batch["book_code_mask"],
            batch["book_features"]
        )

    def get_user_vec(self, batch):
        return self.user_tower(
            batch["last_book_ids"], batch["last_book_mask"],
            batch["last_theme_ids"], batch["last_theme_mask"],
            batch["last_category_ids"], batch["last_category_mask"],
            batch["last_reading_skills_id"], batch["last_reading_skills_mask"],
            batch["countries_ids"], batch["countries_mask"],
            batch["states_ids"], batch["states_mask"],
            batch["zipcode_ids"], batch["zipcode_mask"],
            batch["teacher_ids"], batch["teacher_mask"],
            batch["school_ids"], batch["school_mask"],
            batch["user_features"]
        )




