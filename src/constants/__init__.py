import os
from datetime import date

# For data ingestion
ITEM_QUERY = "sql_files/item_query.sql"
USER_QUERY = "sql_files/user_query.sql"
USER_CLASS_QUERY = "sql_files/class_query.sql"
USER_LOCATION_QUERY = "sql_files/user_location.sql"
USER_BOOK_PLATFORM = "sql_files/user_book_platform.sql"

DATA_INGESTION_OUTPUT_USER = "data/data_ingestion_output/user.csv"
DATA_INGESTION_OUTPUT_ITEM = "data/data_ingestion_output/item.csv"
DATA_INGESTION_OUTPUT_INTERACTION = "data/data_ingestion_output/interaction.csv"

BASE_FOLDER = "feature_mappings"

#for data validation 
VAL_USER_ALL_FEATURES =  ['user_id', 'program_selected','first_student_added_grade_name',  'source', 'options_text',
       'is_subject_personalised','subjects_personalised']
VAL_ITEM_ALL_FEATURES = ['game_code', 'characters', 'color theme', 'game category',
       'game mechanic', 'grade apt', 'interaction type', 'learning sub theme',
       'learning theme', 'level type', 'narrative theme', 'play type']
VAL_INTERACTION_COLUMNS = ['total_unique_playables_attempted',"time_spent_secs","time_spent_sec_all_content","total_game_days_played",'count_days_with_false_start', 'count_days_bounce','timestamp']

# Data transformation
USER_CATEGORICAL_FEATURES =[ 'program_selected',
       'first_student_added_grade_name', 'source', 'options_text',
       'subjects_personalised']
USER_ALL_FEATURES =  ['user_id', 'program_selected','first_student_added_grade_name',  'source', 'options_text',
       'is_subject_personalised','subjects_personalised']
USER_ONE_HOT_ENCODER_PATH ="data/transformation_data_encoders/user_one_hot_encoder.pkl"

ITEM_CATEGORICAL_FEATURES = ['characters', 'color theme', 'game category',
       'game mechanic', 'grade apt', 'interaction type', 'learning sub theme',
       'learning theme', 'level type', 'narrative theme', 'play type']
ITEM_ALL_FEATURES = ['game_code', 'characters', 'color theme', 'game category',
       'game mechanic', 'grade apt', 'interaction type', 'learning sub theme',
       'learning theme', 'level type', 'narrative theme', 'play type']
ITEM_ONE_HOT_ENCODER_PATH ="data/transformation_data_encoders/item_one_hot_encoder.pkl"

NUMERIC_INTERACTION_COLUMN = ['unique_playables_attempted_clipped', 'time_spent_share', 'net_days_played']
ALL_INTERACTION_COLUMN  = ['user_id','game_code','unique_playables_attempted_clipped', 'time_spent_share', 'net_days_played','timestamp']

TRANSFORMED_USER_DATA = 'data/data_ingestion_output/transformation_data/user_data.csv'
TRANSFORMED_ITEM_DATA = 'data/data_ingestion_output/transformation_data/item_data.csv'
TRANSFORMED_INTERACTION_DATA = 'data/data_ingestion_output/transformation_data/interaction_data.csv'

# Model hyperparameter
MAX_SAMPLES = 20
NO_COMPONENTS = 10
LEARNING_RATE = 0.005
LOSS = 'warp'
RANDOM_STATE = 42
MODEL_FILENAME = 'lightfm_model.pkl'
RERANKER_MODEL = 'reranked_games.csv'
BASE_PATH = 'models'
EVALUATION_K =  5
K_FOLD= 5
EPOCHS = 10
ITEM_ALPA = 0.0001
USER_ALPHA = 0.0001

#Model serving

ARTIFACTS = {
        "model": "models/lightfm_model.pkl",
    "encoder_user_one_hot": "data/transformation_data_encoders/user_one_hot_encoder.pkl",
    "reranker":"models/reranked_games.csv",
    "data": "data" 
}
CODE_PATH = [
    "training_pipeline.py",
    "prediction_pipeline.py",
    "src/lightfm_wrapper.py",
    "src",  
    "data",
    
]
PIP_REQUIREMENTS = "requirements.txt" 



# PIPELINE_NAME: str = ""
# ARTIFACT_DIR: str = "artifact"

# MODEL_FILE_NAME = "model.pkl"

# TARGET_COLUMN = "Response"
# CURRENT_YEAR = date.today().year
# PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# FILE_NAME: str = "data.csv"
# TRAIN_FILE_NAME: str = "train.csv"
# TEST_FILE_NAME: str = "test.csv"
# SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


# AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
# AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
# REGION_NAME = "us-east-1"


# """
# Data Ingestion related constant start with DATA_INGESTION VAR NAME
# """
# DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
# DATA_INGESTION_DIR_NAME: str = "data_ingestion"
# DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
# DATA_INGESTION_INGESTED_DIR: str = "ingested"
# DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# """
# Data Validation realted contant start with DATA_VALIDATION VAR NAME
# """
# DATA_VALIDATION_DIR_NAME: str = "data_validation"
# DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# """
# Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
# """
# DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
# DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
# DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# """
# MODEL TRAINER related constant start with MODEL_TRAINER var name
# """
# MODEL_TRAINER_DIR_NAME: str = "model_trainer"
# MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
# MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
# MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
# MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
# MODEL_TRAINER_N_ESTIMATORS=200
# MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
# MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
# MIN_SAMPLES_SPLIT_MAX_DEPTH: int = 10
# MIN_SAMPLES_SPLIT_CRITERION: str = 'entropy'
# MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 101

# """
# MODEL Evaluation related constants
# """
# MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
# MODEL_BUCKET_NAME = "my-model-mlopsproj"
# MODEL_PUSHER_S3_KEY = "model-registry"


# APP_HOST = "0.0.0.0"
# APP_PORT = 5000