from torch.utils.data.dataloader import default_collate
from torch import as_tensor

def collate_none_values(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    #batch = filter (lambda x: x is not None and x[0] is not None, batch)
    #batch = [b for b in batch if b is not None and b[0] is not None]
    #batch = [(i, as_tensor( t )) for (i, t) in batch]
    #print(batch)
    return batch
    #return default_collate([(i, as_tensor(t)) for (i, t) in batch])
