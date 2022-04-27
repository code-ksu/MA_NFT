from pickle import load
import pandas
from torch.utils.data import IterableDataset, get_worker_info
from torch import as_tensor
from PIL import Image
import math
import os
import torch
import numpy as np

class PickleMultiDataset(IterableDataset):
    """A dataset that reads a pickle file and returns its content
    """

    def __init__(self, x_path, y_path, img_transform=None, tab_transform=None, target_transform=None):
        super(PickleMultiDataset).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.tab_transform = tab_transform
        
        with open(x_path,'rb') as path_name:
            self.x = load(path_name)
        with open(y_path,'rb') as path_name:
            self.y = load(path_name)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        
        data = self.x.iloc[index]
        img = self.get_preview_image(data)
        target = self.y.iloc[index]
        
        if self.img_transform is not None and img is not None:
            img = self.img_transform(img)
        #if self.tab_transform is not None:
        #    data = self.tab_transform(data.to_numpy())
        if self.target_transform is not None:
            target = self.target_transform(target)

        data_without_path = data[['param1', 'param2', 'param3']]
        
        data_without_path = as_tensor(data_without_path).reshape(-1) #as_tensor(data.to_numpy()) # torch.from_numpy
        return img, data_without_path, target

    def __len__(self):
        return len(self.x)
    
    def __iter__(self):
        end = len(self.x)
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = end
        else: # in a worker process split workload
            per_worker = int(math.ceil(end / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
        for i in range(iter_start, iter_end):
            img, data, target = self.__getitem__(i)
            if img is not None:
                yield img, data, target

    def get_preview_image(self, row):
        img_path = []
        preview = row['preview_path']
        if preview and preview.startswith('/scraper/data/preview/'):
            preview = os.path.join(os.sep, 'C:' + os.sep, 'nft_data', 'preview', preview[len('/scraper/data/preview/'):])
        if preview != None and pandas.isna(preview) == False:
            img_path.append(preview)
        
        id = row['id']
        image_folder = '..\\..\\opensea_scapper\\opensea_nft_scrapper\\data\\'
        img_path.append(image_folder + 'preview\\' + id + '_noext.png')
        #img_path.append(image_folder + 'img\\' + id + '_noext.png')
        #orginal = row['img_path']
        #if orginal != None and pandas.isna(orginal) == False:
        #    img_path.append(orginal)
        #    if orginal.count('\\') > 0:
        #        img_path.append(image_folder + 'img\\' + orginal[orginal.rindex('\\') + 1:])

        img = None
        for path in img_path:
            try:
                img = Image.open(path).convert('RGB')
                break
            except:
                print(f'IMG NOT FOUND in: {path}')

        return img
    

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
