from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

class BookInteractionDataset(Dataset):
    def __init__(self, interactions_df, book_features_df, book_feature_cols, interaction_feature_cols):
        """
        interactions_df: includes user_id, book_code, label, and interaction-level features
        book_features_df: indexed by book_code, contains theme_ids, category_ids, and other features
        book_feature_cols: list of book feature column names
        interaction_feature_cols: list of interaction-level feature column names
        """
        self.interactions_df = interactions_df.reset_index(drop=True)
        self.book_features_df = book_features_df.set_index("book_code")
        self.book_feature_cols = book_feature_cols
        self.interaction_feature_cols = interaction_feature_cols

    def __len__(self):
        return len(self.interactions_df)

    def __getitem__(self, idx):
        row = self.interactions_df.iloc[idx]
        book_code = row["book_code"]

        # --- 1. Get book-level features ---
        book_info = self.book_features_df.loc[book_code]

        theme_ids = book_info["theme_ids"]  # already list[int]
        category_ids = book_info["category_ids"]  # already list[int]
        reading_skill_ids = book_info["reading_skill_ids"]  # already list[int]
        grades_ids = book_info['grades_ids']  # already list[int]
        book_code_ids= book_info['book_code_ids']  # already list[int]

        last_book_ids = row['book_code_ids']
        last_theme_ids = row['theme_ids']
        last_category_ids = row['category_ids']
        last_reading_skills_id = row['reading_skill_ids']

        countries_ids = row['countries_ids']
        states_ids = row['states_ids']
        zipcode_ids = row['zipcode_ids']
        teacher_code_ids = row['teacher_code_ids']
        school_code_ids = row['school_code_ids']

        # 'countries_ids','states_ids','zipcode_ids','teacher_ids','school_ids'
        

        book_features = np.array(book_info[self.book_feature_cols], dtype=np.float32)

        # --- 2. Get interaction-level features ---
        user_features = np.array(row[self.interaction_feature_cols], dtype=np.float32)

        # # --- 3. Merge into one "other_features" vector ---
        # other_features = torch.tensor(
        #     np.concatenate([book_features, interaction_features]),
        #     dtype=torch.float32
        # )

        return {
            "book_code": book_code,
            "theme_ids": torch.tensor(theme_ids, dtype=torch.long),
            "category_ids": torch.tensor(category_ids, dtype=torch.long),
            "reading_skill_ids":torch.tensor(reading_skill_ids, dtype=torch.long),
            "grades_ids" : torch.tensor(grades_ids , dtype=torch.long) ,
            "book_code_ids": torch.tensor(book_code_ids , dtype=torch.long),

            "last_book_ids" : torch.tensor(last_book_ids , dtype=torch.long),
            "last_theme_ids" : torch.tensor(last_theme_ids , dtype=torch.long),
            "last_category_ids" : torch.tensor(last_category_ids , dtype=torch.long),
            "last_reading_skills_id" : torch.tensor(last_reading_skills_id , dtype=torch.long),

            "countries_ids" : torch.tensor(countries_ids , dtype=torch.long),
            "states_ids" : torch.tensor(states_ids, dtype=torch.long),
            "zipcode_ids" : torch.tensor(zipcode_ids , dtype=torch.long),
            "teacher_code_ids" : torch.tensor(teacher_code_ids , dtype=torch.long),
            "school_code_ids" : torch.tensor(school_code_ids , dtype=torch.long),

            "book_features": torch.tensor(book_features, dtype=torch.float32),
            "user_features": torch.tensor(user_features, dtype=torch.float32),
            "label": torch.tensor(row["label"], dtype=torch.float32)
        }


def book_collate_fn(batch):
    # --------- Helper to pad & mask any list-of-tensors field ----------
    def pad_and_mask(key):
        seqs = [torch.as_tensor(item[key], dtype=torch.long) for item in batch]
        
        padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # [B, max_len]
        mask = (padded != 0).long()  # [B, max_len] boolean mask
        return padded, mask

    # Book-level multi-ID fields
    theme_ids, theme_mask = pad_and_mask("theme_ids")
    category_ids, category_mask = pad_and_mask("category_ids")
    reading_skill_ids, reading_skill_mask = pad_and_mask("reading_skill_ids")
    grades_ids, grades_mask = pad_and_mask("grades_ids")
    book_code_ids, book_code_mask = pad_and_mask("book_code_ids")

    # Interaction-level multi-ID fields
    last_book_ids, last_book_mask = pad_and_mask("last_book_ids")
    last_theme_ids, last_theme_mask = pad_and_mask("last_theme_ids")
    last_category_ids, last_category_mask = pad_and_mask("last_category_ids")
    last_reading_skills_id, last_reading_skills_mask = pad_and_mask("last_reading_skills_id")

    

    countries_ids, countries_mask = pad_and_mask("countries_ids")
    states_ids, states_mask = pad_and_mask("states_ids")
    zipcode_ids, zipcode_mask = pad_and_mask( "zipcode_ids")
    teacher_ids, teacher_mask = pad_and_mask("teacher_code_ids")
    school_ids, school_mask = pad_and_mask("school_code_ids")

    # Scalar / dense features
    book_features = torch.stack([torch.as_tensor(item["book_features"], dtype=torch.float32) for item in batch])
    user_features = torch.stack([torch.as_tensor(item["user_features"], dtype=torch.float32) for item in batch])
    labels = torch.stack([torch.as_tensor(item["label"], dtype=torch.float32) for item in batch])

    return {
        # --- Book-level IDs ---
        "theme_ids": theme_ids, "theme_mask": theme_mask,
        "category_ids": category_ids, "category_mask": category_mask,
        "reading_skill_ids": reading_skill_ids, "reading_skill_mask": reading_skill_mask,
        "grades_ids": grades_ids, "grades_mask": grades_mask,
        "book_code_ids": book_code_ids, "book_code_mask": book_code_mask,

        # --- Interaction-level IDs ---
        "last_book_ids": last_book_ids, "last_book_mask": last_book_mask,
        "last_theme_ids": last_theme_ids, "last_theme_mask": last_theme_mask,
        "last_category_ids": last_category_ids, "last_category_mask": last_category_mask,
        "last_reading_skills_id": last_reading_skills_id, "last_reading_skills_mask": last_reading_skills_mask,

        "countries_ids": countries_ids, "countries_mask": countries_mask,
        "states_ids": states_ids, "states_mask": states_mask,
        "zipcode_ids": zipcode_ids, "zipcode_mask": zipcode_mask,
        "teacher_ids": teacher_ids, "teacher_mask": teacher_mask,
        "school_ids": school_ids, "school_mask": school_mask,

        # --- Dense features & labels ---
       
        "book_features": book_features,
        "user_features": user_features,
        "labels": labels
    }

class BookDataset(Dataset):
    def __init__(self,  book_features_df, book_feature_cols):
        """
        interactions_df: includes user_id, book_code, label, and interaction-level features
        book_features_df: indexed by book_code, contains theme_ids, category_ids, and other features
        book_feature_cols: list of book feature column names
        interaction_feature_cols: list of interaction-level feature column names
        """
        
        self.book_features_df = book_features_df
        self.book_feature_cols = book_feature_cols
       

    def __len__(self):
        return len(self.book_features_df)

    def __getitem__(self, idx):
        row = self.book_features_df.iloc[idx]
        book_code = row["book_code"]

        # --- 1. Get book-level features ---
        # book_info = self.book_features_df.loc[book_code]

        theme_ids = row["theme_ids"]  # already list[int]
        category_ids = row["category_ids"]  # already list[int]
        reading_skill_ids = row["reading_skill_ids"]  # already list[int]
        grades_ids = row['grades_ids']  # already list[int]
        book_code_ids= row['book_code_ids']  # already list[int]


        # 'countries_ids','states_ids','zipcode_ids','teacher_ids','school_ids'
        

        book_features = np.array(row[self.book_feature_cols], dtype=np.float32)

        # --- 2. Get interaction-level features ---
        # user_features = np.array(row[self.interaction_feature_cols], dtype=np.float32)

        # # --- 3. Merge into one "other_features" vector ---
        # other_features = torch.tensor(
        #     np.concatenate([book_features, interaction_features]),
        #     dtype=torch.float32
        # )

        return {
            "book_code": book_code,
            "theme_ids": torch.tensor(theme_ids, dtype=torch.long),
            "category_ids": torch.tensor(category_ids, dtype=torch.long),
            "reading_skill_ids":torch.tensor(reading_skill_ids, dtype=torch.long),
            "grades_ids" : torch.tensor(grades_ids , dtype=torch.long) ,
            "book_code_ids": torch.tensor(book_code_ids , dtype=torch.long),
            "book_features": torch.tensor(book_features, dtype=torch.float32)
           
        }


def collate_bookonly_fn(batch):
    # --------- Helper to pad & mask any list-of-tensors field ----------
    def pad_and_mask(key):
        seqs = [torch.as_tensor(item[key], dtype=torch.long) for item in batch]
        
        padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # [B, max_len]
        mask = (padded != 0).long()  # [B, max_len] boolean mask
        return padded, mask

    # Book-level multi-ID fields
    theme_ids, theme_mask = pad_and_mask("theme_ids")
    category_ids, category_mask = pad_and_mask("category_ids")
    reading_skill_ids, reading_skill_mask = pad_and_mask("reading_skill_ids")
    grades_ids, grades_mask = pad_and_mask("grades_ids")
    book_code_ids, book_code_mask = pad_and_mask("book_code_ids")


    # Scalar / dense features
    book_features = torch.stack([torch.as_tensor(item["book_features"], dtype=torch.float32) for item in batch])

    book_codes = [item["book_code"] for item in batch]
    return {
        # --- Book-level IDs ---
        "book_code": book_codes,
        "theme_ids": theme_ids, "theme_mask": theme_mask,
        "category_ids": category_ids, "category_mask": category_mask,
        "reading_skill_ids": reading_skill_ids, "reading_skill_mask": reading_skill_mask,
        "grades_ids": grades_ids, "grades_mask": grades_mask,
        "book_code_ids": book_code_ids, "book_code_mask": book_code_mask,

        # --- Dense features & labels ---
       
        "book_features": book_features,
       
    }



class UserDataset(Dataset):
    def __init__(self, interactions_df, interaction_feature_cols):
        """
        interactions_df: includes user_id, book_code, label, and interaction-level features
        book_features_df: indexed by book_code, contains theme_ids, category_ids, and other features
        book_feature_cols: list of book feature column names
        interaction_feature_cols: list of interaction-level feature column names
        """
        self.interactions_df = interactions_df.reset_index(drop=True)
        # self.book_features_df = book_features_df.set_index("book_code")
        # self.book_feature_cols = book_feature_cols
        self.interaction_feature_cols = interaction_feature_cols

    def __len__(self):
        return len(self.interactions_df)

    def __getitem__(self, idx):
        row = self.interactions_df.iloc[idx]
        # book_code = row["book_code"]

        # --- 1. Get book-level features ---
        # book_info = self.book_features_df.loc[book_code]

        # theme_ids = book_info["theme_ids"]  # already list[int]
        # category_ids = book_info["category_ids"]  # already list[int]
        # reading_skill_ids = book_info["reading_skill_ids"]  # already list[int]
        # grades_ids = book_info['grades_ids']  # already list[int]
        # book_code_ids= book_info['book_code_ids']  # already list[int]

        last_book_ids = row['book_code_ids']
        last_theme_ids = row['theme_ids']
        last_category_ids = row['category_ids']
        last_reading_skills_id = row['reading_skill_ids']

        countries_ids = row['countries_ids']
        states_ids = row['states_ids']
        zipcode_ids = row['zipcode_ids']
        teacher_code_ids = row['teacher_code_ids']
        school_code_ids = row['school_code_ids']

        # 'countries_ids','states_ids','zipcode_ids','teacher_ids','school_ids'
        

        # book_features = np.array(book_info[self.book_feature_cols], dtype=np.float32)

        # --- 2. Get interaction-level features ---
        user_features = np.array(row[self.interaction_feature_cols], dtype=np.float32)

        # # --- 3. Merge into one "other_features" vector ---
        # other_features = torch.tensor(
        #     np.concatenate([book_features, interaction_features]),
        #     dtype=torch.float32
        # )

        return {
            # "book_code": book_code,
            # "theme_ids": torch.tensor(theme_ids, dtype=torch.long),
            # "category_ids": torch.tensor(category_ids, dtype=torch.long),
            # "reading_skill_ids":torch.tensor(reading_skill_ids, dtype=torch.long),
            # "grades_ids" : torch.tensor(grades_ids , dtype=torch.long) ,
            # "book_code_ids": torch.tensor(book_code_ids , dtype=torch.long),

            "last_book_ids" : torch.tensor(last_book_ids , dtype=torch.long),
            "last_theme_ids" : torch.tensor(last_theme_ids , dtype=torch.long),
            "last_category_ids" : torch.tensor(last_category_ids , dtype=torch.long),
            "last_reading_skills_id" : torch.tensor(last_reading_skills_id , dtype=torch.long),

            "countries_ids" : torch.tensor(countries_ids , dtype=torch.long),
            "states_ids" : torch.tensor(states_ids, dtype=torch.long),
            "zipcode_ids" : torch.tensor(zipcode_ids , dtype=torch.long),
            "teacher_code_ids" : torch.tensor(teacher_code_ids , dtype=torch.long),
            "school_code_ids" : torch.tensor(school_code_ids , dtype=torch.long),

            # "book_features": torch.tensor(book_features, dtype=torch.float32),
            "user_features": torch.tensor(user_features, dtype=torch.float32),
            # "label": torch.tensor(row["label"], dtype=torch.float32)
        }


def collate_useronly_fn(batch):
    # --------- Helper to pad & mask any list-of-tensors field ----------
    def pad_and_mask(key):
        seqs = [torch.as_tensor(item[key], dtype=torch.long) for item in batch]
        
        padded = pad_sequence(seqs, batch_first=True, padding_value=0)  # [B, max_len]
        mask = (padded != 0).long()  # [B, max_len] boolean mask
        return padded, mask

    # Book-level multi-ID fields
    # theme_ids, theme_mask = pad_and_mask("theme_ids")
    # category_ids, category_mask = pad_and_mask("category_ids")
    # reading_skill_ids, reading_skill_mask = pad_and_mask("reading_skill_ids")
    # grades_ids, grades_mask = pad_and_mask("grades_ids")
    # book_code_ids, book_code_mask = pad_and_mask("book_code_ids")

    # Interaction-level multi-ID fields
    last_book_ids, last_book_mask = pad_and_mask("last_book_ids")
    last_theme_ids, last_theme_mask = pad_and_mask("last_theme_ids")
    last_category_ids, last_category_mask = pad_and_mask("last_category_ids")
    last_reading_skills_id, last_reading_skills_mask = pad_and_mask("last_reading_skills_id")

    

    countries_ids, countries_mask = pad_and_mask("countries_ids")
    states_ids, states_mask = pad_and_mask("states_ids")
    zipcode_ids, zipcode_mask = pad_and_mask( "zipcode_ids")
    teacher_ids, teacher_mask = pad_and_mask("teacher_code_ids")
    school_ids, school_mask = pad_and_mask("school_code_ids")

    # Scalar / dense features
    # book_features = torch.stack([torch.as_tensor(item["book_features"], dtype=torch.float32) for item in batch])
    user_features = torch.stack([torch.as_tensor(item["user_features"], dtype=torch.float32) for item in batch])
    # labels = torch.stack([torch.as_tensor(item["label"], dtype=torch.float32) for item in batch])

    return {
        # --- Book-level IDs ---
        # "theme_ids": theme_ids, "theme_mask": theme_mask,
        # "category_ids": category_ids, "category_mask": category_mask,
        # "reading_skill_ids": reading_skill_ids, "reading_skill_mask": reading_skill_mask,
        # "grades_ids": grades_ids, "grades_mask": grades_mask,
        # "book_code_ids": book_code_ids, "book_code_mask": book_code_mask,

        # --- Interaction-level IDs ---
        "last_book_ids": last_book_ids, "last_book_mask": last_book_mask,
        "last_theme_ids": last_theme_ids, "last_theme_mask": last_theme_mask,
        "last_category_ids": last_category_ids, "last_category_mask": last_category_mask,
        "last_reading_skills_id": last_reading_skills_id, "last_reading_skills_mask": last_reading_skills_mask,

        "countries_ids": countries_ids, "countries_mask": countries_mask,
        "states_ids": states_ids, "states_mask": states_mask,
        "zipcode_ids": zipcode_ids, "zipcode_mask": zipcode_mask,
        "teacher_ids": teacher_ids, "teacher_mask": teacher_mask,
        "school_ids": school_ids, "school_mask": school_mask,

        # --- Dense features & labels ---
       
        # "book_features": book_features,
        "user_features": user_features,
        # "labels": labels
    }