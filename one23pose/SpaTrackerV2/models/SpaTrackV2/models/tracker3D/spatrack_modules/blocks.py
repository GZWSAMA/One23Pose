import torch
import torch.nn as nn
import torch.nn.functional as F



class PointDinoV2(nn.Module):
    """
    PointDinoV2 is a 3D point tracking model that uses a backbone and head to extract features from points and track them.
    """
    def __init__(self, ):
        super(PointDinoV2, self).__init__()
        # self.backbone = PointDinoV2Backbone()
        # self.head = PointDinoV2Head()

