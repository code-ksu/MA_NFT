from pickle import load
import pandas
from torch.utils.data import IterableDataset, get_worker_info
from torch import as_tensor, tensor
import torch
from PIL import Image
import math
import os
import numpy as np
from io import BytesIO 
import cairosvg

class PickleSeriesDataset(IterableDataset):
    """A dataset that reads a pickle file and returns its content
    """

    def __init__(self, x_path, y_path, transform=None):
        super(PickleSeriesDataset).__init__()
        self.transform = transform
        pickle_dir = 'D:\\Code\\datascience\\MA_NFT\\data\\pickle\\'
        
        def load_pickle(name):
            print('LOADING: ' + pickle_dir + name + '.pkl')
            with open(pickle_dir + name + '.pkl', 'rb') as f:
                return load(f)
    
        self.x = load_pickle(x_path)
        self.y = load_pickle(y_path)
        self.skipped = 0
            
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
        
        if self.transform is not None and img is not None:
            img = self.transform(img)

        return img, data, target

    def __len__(self):
        return len(self.x) - self.skipped
    
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
            
        self.skipped = 0
        for i in range(iter_start, iter_end):
            try:
                img, data, target = self.__getitem__(i)
                cats = self.get_cats(data)
                nums = self.get_nums(data)
                target = torch.tensor(target, dtype=torch.int64)

                if img is not None:
                    yield img, cats, nums, target
                else:
                    path = data['image_path']
                    self.skipped = self.skipped + 1
                    batch_worked = max(i - iter_start, 1)
                    missing = self.skipped / batch_worked * 100
                    print(f'{missing:.3f}% NOT FOUND: {path}')
            except Exception as e:
                self.skipped = self.skipped + 1
                batch_worked = max(i - iter_start, 1)
                missing = self.skipped / batch_worked * 100
                print(f'{missing:.3f}% NOT FOUND! {i} in {iter_start}-{iter_end} {end}')
                raise e
                
    
    def get_cats(self, row):
        cat_col = [
            'creator',
            'is_animation',
            'name',
            'collection_name',
            'contract_scheme',
            'sale_token',
            'instagram',
            'twitter',
            'collection_created_year',
            'unique_asset',
            'instagram_account',
            'twitter_account']
        return torch.tensor([self.get_cat_index(col, row) for col in cat_col], dtype=torch.int64)
    
    def get_cat_index(self, cat, row):
        value = row[cat]
        cat_list = self.x[cat].cat.categories.tolist()
        return cat_list.index(value)
    
    def get_nums(self, row):
        num_col = [
            'twitter_follower',
            'sale_time',
            'word_count_coll_desc',
            'z_twitter_follower']
        return torch.tensor([row[col] for col in num_col], dtype=torch.float)

    def get_preview_image(self, row):
        path = row['image_path']
        svg = path.endswith('.svg')
        if path.startswith('..\\..\\opensea_scapper\\opensea_nft_scrapper\\data\\'):
            path = path[len('..\\..\\opensea_scapper\\opensea_nft_scrapper\\data\\'):]
            path = 'E:\\data\\opensea\\' + path

        try:
            if svg: # SVGs are not supported by PIL so we convert them to PNGs
                out = BytesIO() 
                cairosvg.svg2png(url=path, write_to=out)
                return Image.open(out).convert('RGB')
            else: # everything else is supported by PIL
                return Image.open(path).convert('RGB')
        except:
            return None
        
    

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    