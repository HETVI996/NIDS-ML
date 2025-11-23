import sys
import pickle
from src.exception import CustomException

def sace_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Parameters:
    file_path (str): The path to the file where the object will be saved.
    obj (any): The Python object to be saved.
    """
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys) 
    
    
def load_object(file_path):
    """
    Load a Python object from a file using pickle.

    Parameters:
    file_path (str): The path to the file from which the object will be loaded.

    Returns:
    any: The Python object loaded from the file.
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)
