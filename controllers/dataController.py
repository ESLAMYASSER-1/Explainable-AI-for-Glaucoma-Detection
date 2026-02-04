import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


from utils import SMGD

import logging 
logging.getLogger(__name__)



class DataController:
    def __init__(self, data_dir:str, CSV_file:str, Fundus_dir:str):
        self.data_dir = data_dir
        self.CSV_file = CSV_file
        self.Fundus_dir = Fundus_dir
        self.DataFrame = None

        self.dataFrameHandle()

    def get_full_path(self, child):
        return self.data_dir + child
    
    def dataFrameHandle(self):
        df = pd.read_csv(self.get_full_path(self.CSV_file))
        df = df.loc[:, ["types", "fundus"]]
        df = df[df.types != -1]

        self.DataFrame = df

    def split_data(self, train_size:float = 0.6, test:bool = False):
        
        assert 0.01 < train_size < 0.99, "train split size should be in range [0.01, 0.99]."

        if test == False:
            x_train, x_val, y_train, y_val = train_test_split(
                                            self.DataFrame.fundus, 
                                            self.DataFrame.types, 
                                            train_size=train_size, 
                                            stratify=self.DataFrame.types
            )

            return {
                "X_train":x_train,
                "y_train":y_train,
                "X_val":x_val,
                "y_val":y_val
            }
        elif test == True:
            x_train, x, y_train, y = train_test_split(
                                            self.DataFrame.fundus, 
                                            self.DataFrame.types, 
                                            train_size=train_size, 
                                            stratify=self.DataFrame.types
            )
            x_val, x_test, y_val, y_test = train_test_split(
                                            x, 
                                            y, 
                                            train_size=0.5, 
                                            stratify=y
            )

            return {
                "X_train":x_train,
                "y_train":y_train,
                "X_val":x_val,
                "y_val":y_val,
                "X_test":x_test,
                "y_test":y_test,
            }
    
    def get_data_loader(self, X, y, batch_size, trasnformations=None, partition_type:str = "train"):
        assert partition_type in ["train", "val", "test"], "data loader types should be in ['train', 'val', 'test']."

        data = SMGD(self.get_full_path(self.Fundus_dir), X, y, trasnformations)

        if partition_type == "train":
            return DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
        
        return DataLoader(data, batch_size=batch_size, pin_memory=True, num_workers=4)


    
    
    

    