import timm
import torch
from torch import nn


@timm.models.register_model
def vit_small_mocov3_patch16_224(**kwargs):

    return timm.models.vision_transformer._create_vision_transformer(
        "vit_small_patch16_224",
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        **kwargs,
    )


def build_timm_vit(
    arch: str,
    global_pool: str = "token",
    use_sincos2d_pos: bool = True,
    grad_checkpointing: bool = False,
):

    _supported = timm.list_models("vit_*")
    if arch not in _supported:
        raise ValueError(f"{arch} is not a supported ViT, choose: {_supported}")

    model = timm.create_model(
        arch,
        num_classes=0,
        global_pool=global_pool,
        class_token=global_pool == "token",
        norm_layer=nn.LayerNorm,
    )
    model.set_grad_checkpointing(grad_checkpointing)#false

    model.width = model.embed_dim#1024

    if use_sincos2d_pos:
        h, w = model.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)

        assert (
            model.embed_dim % 4 == 0
        ), "ViT embed_dim must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = model.embed_dim // 4

        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (10000.0**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        if global_pool == "token":
            pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
            pos_emb = torch.cat([pe_token, pos_emb], dim=1)


        model.pos_embed.data.copy_(pos_emb)
        model.pos_embed.requires_grad = False

    return model
