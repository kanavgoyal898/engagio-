import torch

class Head(torch.nn.Module):
    """
        Head module for the MultiHeadAttention
        in_embd: int, input embedding size
        head_size: int, size of the head
    """

    def __init__(self, in_embd, head_size):

        super().__init__()
        self.in_embd = in_embd
        self.head_size = head_size

        self.query = torch.nn.Linear(self.in_embd, self.head_size)
        self.key = torch.nn.Linear(self.in_embd, self.head_size)
        self.value = torch.nn.Linear(self.in_embd, self.head_size)

    def forward(self, x):
        B, T, C = x.shape                                       # (C == in_embd)

        q = self.query(x)                                       # (B, T, head_size)
        k = self.key(x)                                         # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) / (self.head_size**0.5)   # (B, T, T)
        wei = torch.softmax(wei, -1)                            # (B, T, T)

        v = self.value(x)                                       # (B, T, head_size)
        out = wei @ v                                           # (B, T, head_size)
        return out
    
    
class MultiHeadAttention(torch.nn.Module):

    """
        MultiHeadAttention module
        fan_in: int, input embedding size
        n_embd: int, multi-head key, query, value embedding size
        n_heads: int, number of heads
    """

    def __init__(self, in_embd, n_embd, n_heads):

        super().__init__()
        self.in_embd = in_embd
        self.n_embd = n_embd
        self.n_heads = n_heads

        assert self.n_embd % self.n_heads == 0, 'n_embd should be divisible by n_heads'
        self.head_size = self.n_embd // self.n_heads
        self.heads = torch.nn.ModuleList([Head(self.in_embd, self.head_size) for _ in range(self.n_heads)])
        self.proj = torch.nn.Linear(self.n_embd, self.in_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], -1)   # (B, T, head_size) * n_heads = (B, T, n_embd)
        out = self.proj(out)                                    # (B, T, in_embd)
        return out
    
    
class FeedForward(torch.nn.Module):

    """
        FeedForward module
        in_embd: int, input embedding size
        ffwd_mul: int, feed-forward multiplier
    """

    def __init__(self, in_embd, ffwd_mul):

        super().__init__()
        self.in_embd = in_embd
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.in_embd, ffwd_mul*self.in_embd),
            torch.nn.GELU(),
            torch.nn.Linear(ffwd_mul*self.in_embd, self.in_embd)
        )

    def forward(self, x):
        out = self.net(x)                                       # (B, T, in_embd)
        return out
    
    
class TransformerBlock(torch.nn.Module):

    """
        TransformerBlock module
        in_embd: int, input embedding size
        head_size: int, size of the transformer-block head
        n_heads: int, number of heads
        ffwd_mul: int, feed-forward multiplier
    """

    def __init__(self, in_embd, head_size, n_heads, ffwd_mul):
    
        super().__init__()
        self.in_embd = in_embd
        self.head_size = head_size
        self.n_heads = n_heads
        self.ffwd_mul = ffwd_mul

        self.attn = MultiHeadAttention(self.in_embd, self.head_size, self.n_heads)
        self.norm1 = torch.nn.LayerNorm(self.in_embd)
        self.ffwd = FeedForward(self.in_embd, self.ffwd_mul)
        self.norm2 = torch.nn.LayerNorm(self.in_embd)
        self.conv = torch.nn.Conv1d(self.in_embd, self.in_embd, 1)

    def forward(self, x):

        out = x                                                     # (B, T, in_embd)
        out = self.norm1(out + self.attn(out))                      # (B, T, in_embd)
        out = self.norm2(out + self.ffwd(out))                      # (B, T, in_embd)
        out = self.conv(out.transpose(-2, -1)).transpose(-2, -1)    # (B, T, in_embd)
        return out
    
    
class SignalEncoder(torch.nn.Module):

    """
        SignalEncoder module
        fan_in: int, input embedding size
        fan_out: int, output embedding size
        n_embd: int, signal embedding size
        head_size: int, size of the transformer-block head
        n_heads: int, number of heads
        n_blocks: int, number of transformer-blocks
        ffwd_mul: int, feed-forward multiplier
    """

    def __init__(self, fan_in, fan_out, n_embd, head_size, n_heads, n_blocks, ffwd_mul):  
        
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.n_embd = n_embd
        self.head_size = head_size
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.ffwd_mul = ffwd_mul

        self.context_embedding = torch.nn.Linear(self.fan_in, self.n_embd)
        self.blocks = torch.nn.Sequential(*[TransformerBlock(self.n_embd, self.head_size, self.n_heads, self.ffwd_mul) for _ in range(self.n_blocks)])
        self.ln_f = torch.nn.LayerNorm(self.n_embd)
        self.lm_head = torch.nn.Linear(self.n_embd, self.fan_out)
        
    def forward(self, x):
        B, T, C = x.shape                                       # (B, T, fan_in)

        x = self.context_embedding(x)                           # (B, T, n_embd)
        x = self.blocks(x)                                      # (B, T, n_embd)                                
        x = self.ln_f(x)                                        # (B, T, n_embd)
        x = self.lm_head(x)                                     # (B, T, fan_out)

        return x.view(B, T, self.fan_out)
class EngagementModel(torch.nn.Module):

    def __init__(self, fan_in_A, fan_in_N, fan_out_A, fan_out_N, fan_out, n_embd, head_size, n_heads, n_blocks, ffwd_mul):
        super().__init__()
        self.fan_out = fan_out
        self.signal_encoder_A = SignalEncoder(fan_in_A, fan_out_A, n_embd, head_size, n_heads, n_blocks, ffwd_mul)
        self.signal_encoder_N = SignalEncoder(fan_in_N, fan_out_N, n_embd, head_size, n_heads, n_blocks, ffwd_mul)
        self.lm_head = torch.nn.Sequential(
                            torch.nn.Linear(fan_out_A + fan_out_N, fan_out),
                            torch.nn.GELU()
                        )

    def forward(self, x_A, x_N, idx=None, weighted=False):
        x_A = self.signal_encoder_A(x_A)
        x_N = self.signal_encoder_N(x_N)
        
        out = torch.cat([x_A, x_N], -1)
        out = self.lm_head(out)
        B, T, C = out.shape

        loss = None
        if idx is not None:
            x = out.float()
            y = idx.unsqueeze(1).repeat(1, T, 1).float()
            x = x.view(-1, self.fan_out)
            y = y.view(-1).long()
            loss = torch.nn.functional.cross_entropy(x, y, 
                                                    weight=torch.tensor([1/61, 1/455, 1/4422, 1/3987], dtype=x.dtype, device=x.device) if weighted else None)
        
        return out, loss