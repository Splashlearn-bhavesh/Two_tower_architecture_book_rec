import os
import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from src.utils.main_utils import save_dict_to_json,  load_json_file


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_transformation')
logger.setLevel(logging.DEBUG)


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(log_dir, 'data_transformation.log'), mode='a')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def encode_column_with_sentence_transformer(df: pd.DataFrame, column: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Encodes a column of text into embeddings using a sentence-transformer model.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to encode.
        model_name (str): Pretrained sentence-transformers model name.

    Returns:
        np.ndarray: Array of shape (num_rows, embedding_dim)
    """
    model = SentenceTransformer(model_name)
    
    # Fill missing values
    texts = df[column].fillna("unk").astype(str).tolist()
    
    # Encode with model
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return np.array(embeddings)

def pre_train_emb_creation(book_df):

    book_df['title_plus_author'] = book_df.apply(lambda x:x['book_title'].lower()+' by '+x['authors'].lower(),axis=1)
    book_df['long_description'].fillna('unk',inplace=True)
    book_df['long_description'] = book_df.apply(lambda x:x['long_description'].lower(),axis=1)
    columns = ['book_isbn', 'title_plus_author', 'book_series', 'book_type', 'long_description','min_grade', 'max_grade',
        'readable_page_count','fiction_nonfiction', 'reading_skill_name','theme_name', 'category_name','language_book']

    book_df_final = book_df[columns]

    emb = encode_column_with_sentence_transformer(book_df_final,'title_plus_author')
    emb_df = pd.DataFrame(emb, columns=[f"emb_title_author_{i}" for i in range(emb.shape[1])])

    # Combine with book_id
    book_embedding_author_df = pd.concat([book_df_final, emb_df], axis=1)

    emb_desc = encode_column_with_sentence_transformer(book_embedding_author_df,'long_description')
    # Convert embeddings to DataFrame
    emb_desc_df = pd.DataFrame(emb_desc, columns=[f"emb_desc_{i}" for i in range(emb_desc.shape[1])])

    # Combine with book_id
    long_description_df = pd.concat([book_embedding_author_df, emb_desc_df], axis=1)
    emb_book_series = encode_column_with_sentence_transformer(long_description_df,'long_description')
    # Convert embeddings to DataFrame
    emb_book_series_df = pd.DataFrame(emb_book_series, columns=[f"emb_book_series_{i}" for i in range(emb_book_series.shape[1])])

    # Combine with book_id

    book_series_df = pd.concat([long_description_df,emb_book_series_df ], axis=1)

    return book_series_df , emb.shape[1] ,emb_desc.shape[1] ,emb_book_series.shape[1]

def clip(df,col,min,max):
    df[col] = np.clip(df[col],min,max)
    return df 

def scaling(df, col ,value):
    df[col] = df[col]/value
    return df

grade_list = ['pk', 'k', '1', '2', '3', '4', '5', '6', '7', '8']
grade_to_idx = {g: i for i, g in enumerate(grade_list)}

def get_range(min_g, max_g):
    start_idx = grade_to_idx[min_g]
    end_idx = grade_to_idx[max_g]
    return ','.join(grade_list[start_idx:end_idx + 1])

def get_category_mapping(item_df, input_col, out_put_col,json_file_name):

    # Step 1: Preprocess themes (split on commas)
    item_df['themes'] = item_df[input_col].fillna('').apply(
        lambda x: [t.strip().lower() for t in x.split(',') if t.strip()]
    )
    # Step 2: Build theme vocabulary
    from itertools import chain
    all_themes = sorted(set(chain.from_iterable(item_df['themes'])))
    theme_to_idx = {theme: idx for idx, theme in enumerate(all_themes)}
    if 'unk' not in theme_to_idx:
        theme_to_idx['unk'] = len(theme_to_idx)

    # Step 3: Map themes to indices
    item_df[out_put_col] = item_df['themes'].apply(
        lambda theme_list: [theme_to_idx[t] for t in theme_list if t in theme_to_idx]
    )
    save_dict_to_json(theme_to_idx,json_file_name)
    return item_df, len(theme_to_idx)

def get_category_mapping_book(item_df,input_col ,out_put_col,json_file_name):

    book_code_to_idx = {theme: idx for idx, theme in enumerate(list((item_df[input_col])))}
    if 'unk' not in book_code_to_idx:
        book_code_to_idx['unk'] = len(book_code_to_idx)

    # Step 3: Map themes to indices
    item_df[out_put_col] = item_df[input_col].apply(
        lambda theme_list: [book_code_to_idx[t] for t in [theme_list] if t in book_code_to_idx]
    )
    save_dict_to_json(book_code_to_idx,json_file_name)
    return item_df, len(book_code_to_idx)


# book_df_final['readable_page_count'] = np.clip(book_df_final['readable_page_count'],0,50)/50
# book_series_df.shape

def last_10_books_fast(df):
    df = df.copy()
    df['book_create_dt'] = pd.to_datetime(df['book_create_dt'])
    df = df.sort_values(['user_id', 'book_create_dt']).reset_index(drop=True)

    # Helper to join last 10 values for each row in a group
    def last_10_join(series):
        out = []
        hist = []
        for val in series:
            out.append(','.join(hist[-10:]) if hist else 'unk')
            hist.append(val)
        return pd.Series(out, index=series.index)

    # Precompute category/theme strings
    # df['cat_str'] = df['category_name'].apply(lambda x: ','.join(x))
    # print(df['cat_str'])
    # df['theme_str'] = df['theme_name'].apply(lambda x: ','.join(x))
    df['cat_str'] = df['category_name']
    # print(df['cat_str'])
    df['theme_str'] = df['theme_name']
    df['rs_str'] = df['reading_skill_name']


    # Vectorized per-group computation (one Python loop per group, not per row globally)
    grouped = df.groupby('user_id', group_keys=False)
    df['last_10_books'] = grouped['book_code'].transform(last_10_join)
    df['last_category_name'] = grouped['cat_str'].transform(last_10_join)
    df['last_theme_name'] = grouped['theme_str'].transform(last_10_join)
    df['last_reading_skill_name'] = grouped['rs_str'].transform(last_10_join)

    return df.drop(columns=['cat_str', 'theme_str','rs_str'])

def get_mapping_user(child_df , input_col, output_col , file_path ):
    category_to_idx = load_json_file(file_path)
    child_df['last_categories'] = child_df[input_col].fillna('').apply(
    lambda x: [t.strip().lower() for t in x.split(',') if t.strip()])
    child_df[output_col] = child_df['last_categories'].apply(
    lambda theme_list: [category_to_idx[t] for t in theme_list if t in category_to_idx] )

    return child_df, len(category_to_idx)

def get_mapping_book_user(child_df , input_col, output_col , file_path ):
    book_code_to_idx = load_json_file(file_path)
    child_df['last_books_list'] = child_df[input_col].fillna('').apply(
    lambda x: [t.strip().lower() for t in x.split(',') if t.strip()]
)

    child_df[output_col] = child_df['last_books_list'].apply(
    lambda theme_list: [book_code_to_idx[int(t)] if t!='unk' else book_code_to_idx[t]  for t in theme_list ])

    return child_df





def book_data_transformation(book_df):

    book_series_df, emb_shape,emb_desc_shape,emb_book_series_shape = pre_train_emb_creation(book_df)

    book_series_df = scaling(clip(book_series_df,'readable_page_count',0,50),'readable_page_count',50)

    book_series_df['book_type_binary'] = np.where(book_series_df.book_type == 'PDF',1,0)

    book_series_df['fiction_nonfiction'].fillna('unk',inplace =True)
    book_df_final_v1 = pd.get_dummies(book_series_df, columns=['fiction_nonfiction'], prefix='fn')

    book_df_final_v1 = pd.get_dummies(book_df_final_v1, columns=['language_book'], prefix='lang')

    book_df_final_v1["grades"] = book_df_final_v1.apply(
        lambda row: get_range(row["min_grade"], row["max_grade"]),axis=1)

    book_df_final_v1, theme_count = get_category_mapping(book_df_final_v1, 'theme_name', 'theme_ids','theme_to_idx.json')
    book_df_final_v1, category_count = get_category_mapping(book_df_final_v1, 'category_name', 'category_ids','category_to_idx.json')
    book_df_final_v1, reading_skills_count = get_category_mapping(book_df_final_v1, 'reading_skill_name', 'reading_skill_ids','reading_skill_to_idx.json')
    book_df_final_v1, grades_count = get_category_mapping(book_df_final_v1, 'grades', 'grades_ids','grades_to_idx.json')
    book_df_final_v1, book_count = get_category_mapping_book(book_df_final_v1,'book_isbn' ,'book_code_ids','book_code_to_idx.json')
    book_df_final_v1.rename(columns={'book_isbn':'book_code'},inplace=True)
    book_feature_count =  {
                    'themes_count':theme_count, 
                    'book_count': book_count, 
                    'grade_count': grades_count,
                    'reading_skills_count':reading_skills_count,
                    'category_count':category_count
                   }
    emb_count = {
                    'themes_count':8, 
                    'book_count': 16, 
                    'grade_count': 4,
                    'reading_skills_count':4,
                    'category_count':4
                }
    
    columns_author_title =[f"emb_title_author_{i}" for i in range(emb_shape)]
    columns_long_description = [f"emb_desc_{i}" for i in range(emb_desc_shape)]
    columns_book_series = [f"emb_book_series_{i}" for i in range(emb_book_series_shape)]
    columns_add = ['readable_page_count','book_type_binary', 'fn_Fiction', 'fn_Non-Fiction', 'fn_unk',
        'lang_English', 'lang_French', 'lang_Haitian French Creole',
        'lang_Mandarin', 'lang_Portuguese', 'lang_Spanish']

    columns_learn_emb = ['book_code','book_code_ids','grades_ids','reading_skill_ids', 'category_ids','theme_ids']

    book_feature_cols = columns_learn_emb + columns_author_title + columns_long_description + columns_book_series + columns_add  

    return book_df_final_v1[book_feature_cols], book_feature_count, emb_count 


def user_data_transformation(user_df,user_loc,user_platform): 
    # user_platform.rename(columns ={'isbn':'book_code'},inplace=True)
    df1 = user_platform.copy()
    user_platform = df1[['user_id', 'book_code', 'book_create_dt',
       'cumulative_web_during_school_hour',
       'cumulative_web_after_school_hour',
       'cumulative_apple_during_school_hour',
       'cumulative_apple_after_school_hour',
       'cumulative_android_during_school_hour', 'cumulative_android_after_school_hour',
       'cumulative_unk_during_school_hour',
       'cumulative_unk_after_school_hour']]
    
    user_platform['total'] = user_platform['cumulative_web_during_school_hour'] + user_platform['cumulative_web_after_school_hour'] + user_platform['cumulative_apple_during_school_hour']+ user_platform['cumulative_apple_after_school_hour']+ user_platform['cumulative_android_during_school_hour']+ user_platform['cumulative_android_after_school_hour']+ user_platform['cumulative_unk_during_school_hour']+ user_platform['cumulative_unk_after_school_hour']

    user_platform['cumulative_web_during_school_hour'] = user_platform['cumulative_web_during_school_hour']/user_platform['total']
    user_platform['cumulative_web_after_school_hour']  = user_platform['cumulative_web_after_school_hour'] /user_platform['total']
    user_platform['cumulative_apple_during_school_hour'] = user_platform['cumulative_apple_during_school_hour']/user_platform['total']
    user_platform['cumulative_apple_after_school_hour'] = user_platform['cumulative_apple_after_school_hour']/user_platform['total']
    user_platform['cumulative_android_during_school_hour'] = user_platform['cumulative_android_during_school_hour']/user_platform['total']
    user_platform['cumulative_android_after_school_hour'] = user_platform['cumulative_android_after_school_hour']/user_platform['total']
    user_platform['cumulative_unk_during_school_hour'] = user_platform['cumulative_unk_during_school_hour']/user_platform['total']
    user_platform['cumulative_unk_after_school_hour']  = user_platform['cumulative_unk_after_school_hour']/user_platform['total']
    
    user_platform_final = user_platform[['user_id', 'book_code',
       'cumulative_web_during_school_hour', 'cumulative_web_after_school_hour',
       'cumulative_apple_during_school_hour',
       'cumulative_apple_after_school_hour',
       'cumulative_android_during_school_hour', 'cumulative_android_after_school_hour',
       'cumulative_unk_during_school_hour', 'cumulative_unk_after_school_hour',
       ]]

    user_df.dropna(subset=['category_name'],inplace=True)


    user_df['category_name'] = user_df['category_name'].fillna('unk')
    user_df['theme_name'] = user_df['theme_name'].fillna('unk')
    user_df['reading_skill_name'] = user_df['reading_skill_name'].fillna('unk')

    user_df['total_pages']=user_df['total_pages'].fillna(user_df['total_pages'].median())
    user_df['max_read_pages']=user_df['max_read_pages'].fillna(user_df['max_read_pages'].median())


    user_df_v1 = user_df[['book_code', 'user_id','category_name','theme_name','reading_skill_name', 'book_create_dt', 'total_pages',
       'max_read_pages']].copy()

    user_df_v1['book_create_dt'] = pd.to_datetime(user_df_v1['book_create_dt'])

    user_loc_v1 = user_loc[['user_id','country', 'state', 'zipcode','klass_grade_name','teacher_id','school_id','class_activation_bucket']].copy()
    
    user_raw_df =  user_df_v1.merge(user_loc_v1, how ='left' ,on = 'user_id')

   #  user_platform_final['book_code'] = user_platform_final['book_code'].astype('str')

    user_raw_df_v1 = user_raw_df.merge(user_platform_final, how ='left' ,on = ['user_id','book_code'])

    cv = user_raw_df_v1 .copy()

    user_raw_df_v2 = last_10_books_fast(cv)
    user_raw_df_v2['class_activation_bucket'] = user_raw_df_v2['class_activation_bucket'].fillna('unk')
    user_raw_df_v2 = pd.get_dummies(user_raw_df_v2, columns=['klass_grade_name'], prefix='grade')
    user_raw_df_v2 = pd.get_dummies(user_raw_df_v2, columns=['class_activation_bucket'], prefix='class_activation_bucket')

    user_raw_df_v2['completion_rate'] = user_raw_df_v2['max_read_pages']/user_raw_df_v2['total_pages']
    user_raw_df_v2['label'] = np.where(user_raw_df_v2['completion_rate']>0.5,1,0)

    user_raw_df_v2, category_count  = get_mapping_user(user_raw_df_v2 , 'last_category_name', 'category_ids' , 'feature_mappings/category_to_idx.json')
    user_raw_df_v2, book_count = get_mapping_user(user_raw_df_v2 , 'last_10_books', 'book_code_ids' , 'feature_mappings/book_code_to_idx.json')
    # user_raw_df_v2, book_count = get_mapping_book_user(user_raw_df_v2 , 'last_10_books', 'book_code_ids' , 'feature_mappings/book_code_to_idx.json')
    user_raw_df_v2, reading_skills_count  = get_mapping_user(user_raw_df_v2 , 'last_reading_skill_name', 'reading_skill_ids' , 'feature_mappings/reading_skill_to_idx.json')
    user_raw_df_v2, theme_count = get_mapping_user(user_raw_df_v2 , 'last_theme_name', 'theme_ids' , 'feature_mappings/theme_to_idx.json')
    user_raw_df_v2, country_count = get_category_mapping(user_raw_df_v2, 'country', 'countries_ids','country_to_idx.json')
    user_raw_df_v2, state_count = get_category_mapping(user_raw_df_v2, 'state', 'states_ids','state_to_idx.json')
    user_raw_df_v2, zipcode_count = get_category_mapping(user_raw_df_v2, 'zipcode', 'zipcode_ids','zipcode_to_idx.json')
    user_raw_df_v2, teacher_count = get_category_mapping(user_raw_df_v2, 'teacher_id', 'teacher_code_ids','teacher_to_idx.json')
    user_raw_df_v2, school_count = get_category_mapping(user_raw_df_v2, 'school_id', 'school_code_ids','school_to_idx.json')

    user_columns = ['book_code', 'user_id', 'book_create_dt','book_code_ids','category_ids','reading_skill_ids','theme_ids','countries_ids', 'states_ids','zipcode_ids','teacher_code_ids','school_code_ids','state', 'zipcode',
       'teacher_id', 'school_id','cumulative_web_during_school_hour', 'cumulative_web_after_school_hour',
       'cumulative_apple_during_school_hour',
       'cumulative_apple_after_school_hour',
       'cumulative_android_during_school_hour',
       'cumulative_android_after_school_hour',
       'cumulative_unk_during_school_hour', 'cumulative_unk_after_school_hour',
        'grade_grade 1', 'grade_grade 2', 'grade_grade 3',
       'grade_grade 4', 'grade_grade 5', 'grade_kindergarten', 
       'class_activation_bucket_AC', 'class_activation_bucket_AC0',
       'class_activation_bucket_AC1', 'class_activation_bucket_AC2',
       'class_activation_bucket_AC3', 'class_activation_bucket_unk', 'last_10_books', 'last_category_name', 'last_theme_name',
       'last_reading_skill_name','label']

    user_feature_count =  {'themes_count':theme_count, 
                   'book_count': book_count, 
                   'reading_skills_count':reading_skills_count,
                   'category_count':category_count,
                   'country_count': country_count , 
                    'state_count': state_count ,
                    'zipcode_count': zipcode_count,
                    'teacher_count': teacher_count,
                    'school_count': school_count
                   }
    user_emb_count = {
                'themes_count':8, 
                'book_count': 16, 
                'reading_skills_count':4,
                'category_count':4,
                'country_count': 8 , 
                'state_count': 10,
                'zipcode_count': 14,
                'teacher_count': 16,
                'school_count': 16
                }

    return user_raw_df_v2[user_columns], user_feature_count, user_emb_count

