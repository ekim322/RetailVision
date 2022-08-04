import torch
import torch.nn as nn

from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletNet(nn.Module):
    def __init__(self, embedding_size=64):
        super(TripletNet, self).__init__()
        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        resnet.fc = nn.Linear(2048, embedding_size)

        self.resnet = resnet

    def forward(self, imgs):
        embeddings = self.resnet(imgs)
        return embeddings


class TripletMiningLoss(nn.Module):
    def __init__(self):
        super(TripletMiningLoss, self).__init__()

        
    def forward(self, vector_batch, labels_batch):
        
        #dists = torch.sqrt(torch.sum((vector_batch[:, None, :] - vector_batch[:, :] + 1e-8) ** 2, -1))
        dists = torch.linalg.norm(vector_batch[:, None, :] - vector_batch[:, :] + 1e-8, dim=-1)
        
        pos_mask = self.get_positive_mask(labels_batch)
        neg_mask = self.get_negative_mask(labels_batch)
                
        positive_dists = torch.max(dists * pos_mask, 1)[0]
        
        # min masking doesn't work because we'll just take 0 when the neg_mask is applied
        # so we take the maximum distance value and set that in the inverse of the negative mask
        # then when we take the min we won't accidentialy take a value where two indices match
        global_max_value = torch.max(dists).item()
        negative_dists = torch.min(dists + (global_max_value * ~neg_mask), 1)[0]
        
        zero_tnsr = torch.Tensor([0.0])
        zero_tnsr = zero_tnsr.to(device)
        tl = torch.max(positive_dists - negative_dists + 0.5, zero_tnsr)
        
        return torch.mean(tl)
    
    def get_positive_mask(self, labels):
        # ones everywhere except for diagonal (same index) 
        diag_mask = torch.eye(labels.size(0)).bool()
        diag_mask = ~diag_mask
        diag_mask = diag_mask.to(device)

        # same label 
        equal_mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        
        # get the union of matching index and the diagonal mask
        mask = diag_mask & equal_mask
        
        return mask
    
    def get_negative_mask(self, labels):
        # get the makes for where labels don't match
        return torch.ne(labels.unsqueeze(0), labels.unsqueeze(1))