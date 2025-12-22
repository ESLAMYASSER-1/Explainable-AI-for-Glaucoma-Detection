import torchvision.transforms as trns


from controllers import DataController

import logging 
logging.getLogger(__name__)

class DataModel:
    def __init__(self, data_dir:str, CSV_file:str, Fundus_dir:str, train_size:float, include_test:bool):
        self.include_test = include_test
        self.DataController = DataController(data_dir=data_dir, CSV_file=CSV_file, Fundus_dir= Fundus_dir)
        self.Data_dict = self.DataController.split_data(train_size=train_size, test=include_test)

    def get_loaders(self, batch_size, trasnformations):
        train_loader = self.DataController.get_data_loader(
                                                self.Data_dict["X_train"],
                                                self.Data_dict["y_train"], 
                                                batch_size,
                                                trasnformations,
                                                partition_type= "train"
        )
        val_loader = self.DataController.get_data_loader(
                                                self.Data_dict["X_val"],
                                                self.Data_dict["y_val"], 
                                                batch_size//2,
                                                trasnformations,
                                                partition_type= "val"
        )

        test_loader = None
        if self.include_test:
            test_loader = self.DataController.get_data_loader(
                                                    self.Data_dict["X_test"],
                                                    self.Data_dict["y_test"], 
                                                    batch_size//2,
                                                    trasnformations,
                                                    partition_type= "test"
            )
        return train_loader, val_loader, test_loader


        
    
        
