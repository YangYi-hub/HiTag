from __future__ import annotations

import re
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class _TransformerBlock(nn.Module):


    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        lx = self.ln_1(x)
        ax = self.attn(lx, lx, lx, need_weights=False, attn_mask=attn_mask)[0]
        x = x + ax
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerTextEncoder(nn.Module):

    def __init__(
        self,
        arch: str,
        vocab_size: int,
        context_length: int,
        grad_checkpointing: bool = False,
    ):

        super().__init__()
        self.vocab_size = vocab_size#49408
        self.context_length = context_length#77
        self.grad_checkpointing = grad_checkpointing#False

        self.layers = int(re.search(r"L(\d+)", arch).group(1))#12
        self.width = int(re.search(r"W(\d+)", arch).group(1))#512

        _attn = re.search(r"A(\d+)", arch)#None
        self.heads = int(_attn.group(1)) if _attn else self.width // 64#8

        self.token_embed = nn.Embedding(vocab_size, self.width)
        self.posit_embed = nn.Parameter(torch.empty(context_length, self.width))

        _resblocks = [
            _TransformerBlock(self.width, self.heads) for _ in range(self.layers)
        ]
        self.resblocks = nn.ModuleList(_resblocks)
        self.ln_final = nn.LayerNorm(self.width)

        attn_mask = torch.triu(
            torch.full((context_length, context_length), float("-inf")), diagonal=1
        )
        self.register_buffer("attn_mask", attn_mask.bool())

        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.posit_embed.data, std=0.01)

        out_proj_std = (2 * self.width * self.layers) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=self.width**-0.5)
            nn.init.normal_(block.attn.out_proj.weight, std=out_proj_std)
            nn.init.normal_(block.mlp[0].weight, std=(2 * self.width) ** -0.5)
            nn.init.normal_(block.mlp[2].weight, std=out_proj_std)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:


        max_len = text_tokens.shape[-1]#77
        _posit_embed = self.posit_embed[:max_len, :]
        _attn_mask = self.attn_mask[:max_len, :max_len]

        token_embeddings = self.token_embed(text_tokens) + _posit_embed

        textual_features = token_embeddings
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                textual_features = checkpoint(block, textual_features, _attn_mask)
            else:
                textual_features = block(textual_features, _attn_mask)

        textual_features = self.ln_final(textual_features)
        return textual_features
