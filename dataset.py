import io
import os
import copy
import json
import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
  
class LogsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self,  use_tokenizer,path='dataset/bgl_logs_train.json',):

    # Check if path exists.
    with open(path, 'r') as file:
        self.data = json.load(file)


    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return len(self.data)
  
  def __getitem__(self, idx):
        return self.data[idx]
  

  
class AnomalousLogsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self,  use_tokenizer,path='dataset/bgl_logs_train.json',):

    # Check if path exists.
    with open(path, 'r') as file:
        self.data = json.load(file)
    self.data = [item for item in self.data if item['label'] == 'Anomalous']

    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return len(self.data)
  def __getitem__(self, idx):
        return self.data[idx]
  
class UnlabeledLogsDataset(Dataset):
  r"""PyTorch Dataset class for loading data.

  This is where the data parsing happens.

  This class is built with reusability in mind: it can be used as is as.

  Arguments:

    path (:obj:`str`):
        Path to the data partition.

  """

  def __init__(self,  use_tokenizer,path='dataset/bgl_logs_train.json',):

    # Check if path exists.
    with open(path, 'r') as file:
        self.data = json.load(file)
    self.data = [item for item in self.data if item['label'] == 'Anomalous']
    self.tokenizer=use_tokenizer

    return

  def __len__(self):
    r"""When used `len` return the number of examples.

    """
    
    return len(self.data)
  
  def __getitem__(self, idx):
        sentence = self.data[idx]['log']
        inputs = self.tokenizer.encode(sentence, add_special_tokens=True)
        return torch.tensor(inputs)
#   def __getitem__(self, item):
#     r"""Given an index return an example from the position.
    
#     Arguments:

#       item (:obj:`int`):
#           Index position to pick an example to return.

#     Returns:
#       :obj:`Dict[str, str]`: Dictionary of inputs that contain text and 
#       asociated labels.

#     """

#     return {'text':self.log[item],
#             'label':self.labels[item]}
  