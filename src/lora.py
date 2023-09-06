import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        Aq: nn.Module,
        Bq: nn.Module,
        Av: nn.Module,
        Bv: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.k = qkv.in_features
        self.Aq, self.Av = Aq, Av
        self.Bq, self.Bv = Bq, Bv

    def forward(self, x):
        # W + BAx has to be done for Q,V,K since it's in one big matrix Wqkv
        # return W + BAx
        W = self.qkv(x)  # B, N, 3*k
        W[:, :, :self.k] += self.Bq(self.Aq(x))   # For Q
        W[:, :, -self.k:] += self.Bv(self.Av(x))  # For V
        return W


class LoRATransformer(nn.Module):
    def __init__(self, transformer, r):
        super().__init__()

        # Freeze
        for param in transformer.parameters():
            param.requires_grad = False

        # Iterate over transformer
        for layer, block in enumerate(transformer.blocks):
            """
            From the paper:
            h = W0x + ∆W x = W0x + BAx
            
            Weight matrix W0 ∈ R^d×k, we constrain its update by representing the latter with a low-rank de-
            composition W0 + ∆W = W0 + BA, where B ∈ R^d×r , A ∈ R^r×k, and the rank r = min(d, k)
            
            With r = 4 and only the query (Wq) and value (Wv) projection matrices being adapted, the checkpoint
            size is reduced by roughly 10,000× (from 350GB to 35MB).
            """
            # print(block)
            k = block.attn.qkv.in_features

            # Use LoRALayer
            block.attn.qkv = LoRALayer(
                # Wqkv
                block.attn.qkv,
                # Wq [W = d x k, B = d x r, A = r x k]
                nn.Linear(k, r, bias=False),
                nn.Linear(r, k, bias=False),
                # Wv
                nn.Linear(k, r, bias=False),
                nn.Linear(r, k, bias=False),
            )

        self.lora = transformer
        self.lora.reset_classifier(num_classes=10)

    def forward(self, x):
        return self.lora(x)
