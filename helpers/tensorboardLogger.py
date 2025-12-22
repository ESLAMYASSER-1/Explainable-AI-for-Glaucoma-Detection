from lightning.pytorch.loggers import TensorBoardLogger
from glob import glob
from pathlib import Path
def get_TBlogger(save_dir, experiment_name ):
    """Create tensorBoardLogger

    Args:
        save_dir (str): directory of logs 
        experiment_name (str): name of the experiment
        
        $ /save_dir/name/version/
    Returns:
        TensorBoardLogger: TensorBoardLogger 
    """
    # if not Path.exists(Path(save_dir)):
    #     Path.mkdir(Path(save_dir))
    
    # if not Path.exists(Path(save_dir)/experiment_name):
    #     Path.mkdir(Path(save_dir)/experiment_name)
    
    version = len(glob(f"{save_dir}/{experiment_name}/*"))+1
    return  TensorBoardLogger(
                                save_dir=save_dir,
                                name=experiment_name,
                                version = version ,
                                default_hp_metric =True,
    )