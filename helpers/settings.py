from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME:str
    VERSION:str

    LOGGING_DIR:str = "../logs"
    EXPERIMENT_NAME:str

    DATASET_DIR:str 
    CSV_FILE:str = "metadata - standardized.csv"
    FUNDUS_DIR:str = "full-fundus"

    TRAIN_SIZE:float = 0.6
    INCLUDE_TEST:bool = False
    BATCH_SIZE:int = 16

    NUM_EPOCHS:int 

    model_config = SettingsConfigDict(
        env_file=".env"
    )