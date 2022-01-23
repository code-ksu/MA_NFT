from pickle import load
import pandas
from torch.utils.data import Dataset, get_worker_info
from torch import as_tensor
from PIL import Image
import math

class PickleSeriesDataset(Dataset):
    """A dataset that reads a pickle file and returns its content
    """

    def __init__(self, x_path, y_path, transform=None, target_transform=None):
        super(PickleSeriesDataset).__init__()
        self.transform = transform
        self.target_transform = target_transform
        
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

        img = self.get_preview_image(self.x.iloc[index])
        target = self.y.iloc[index]
        
        if self.transform is not None and img is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.x)

    def get_preview_image(self, row):
        img_path = []
        preview = row['preview_path']
        if preview != None and pandas.isna(preview) == False:
            img_path.append(preview)
        
        id = row['id']
        image_folder = '..\\..\\opensea_scapper\\opensea_nft_scrapper\\data\\'
        img_path.append(image_folder + 'preview\\' + id + '_noext.png')
        img_path.append(image_folder + 'img\\' + id + '_noext.png')
        orginal = row['img_path']
        if orginal != None and pandas.isna(orginal) == False:
            img_path.append(orginal)
            if orginal.count('\\') > 0:
                img_path.append(image_folder + 'img\\' + orginal[orginal.rindex('\\') + 1:])

        img = None
        for path in img_path:
            try:
                img = Image.open(path).convert('RGB')
                break
            except:
                pass
        if img == None:
            print(f'IMG NOT FOUND!!!! {id}')
        
        return img
    

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    