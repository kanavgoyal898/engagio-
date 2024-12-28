import torch

class Architecture(torch.nn.Module):

    def __init__(self, landmark_count, dimension_count, embedding_count, class_count):
        super().__init__()
        self.upsamp = torch.nn.Linear(dimension_count, embedding_count, bias=False)
        self.ffwd_1 = torch.nn.Linear(embedding_count, embedding_count)
        self.actv_1 = torch.nn.ReLU()
        self.bnrm_1 = torch.nn.BatchNorm1d(embedding_count)
        self.ffwd_2 = torch.nn.Linear(embedding_count, embedding_count)
        self.actv_2 = torch.nn.ReLU()
        self.bnrm_2 = torch.nn.BatchNorm1d(embedding_count)
        self.dnsamp = torch.nn.Linear(embedding_count, class_count)

    def forward(self, x, y=None):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, T, C = x.size()

        x = self.upsamp(x)                                  # B x T x E
        x = self.actv_1(self.ffwd_1(x))                    # B x T x E
        x = self.bnrm_1(x.view(B*T, -1)).view(B, T, -1)    # B x T x E
        x = self.actv_2(self.ffwd_2(x))                    # B x T x E
        x = self.bnrm_2(x.view(B*T, -1)).view(B, T, -1)    # B x T x E
        x = self.dnsamp(x)                                  # B x T x class_count

        if y is None:
            return x.view(B*T, -1)
        
        loss = None
        x = x.view(B*T, -1)
        y = y.view(B*T)
        loss = torch.nn.functional.cross_entropy(x, y)
        x = x.view(B, T, -1)
        return x, loss
