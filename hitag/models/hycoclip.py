from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from . import distributed as dist
from . import lorentz as L
from .text_encoders import TransformerTextEncoder

import json
import warnings
from .utils import *

warnings.filterwarnings("ignore")

class CLIPBaseline(nn.Module):

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):

        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):

        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):

        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)
        text_feats = self.textual(tokens)

        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:

        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        batch_size = image_feats.shape[0]
        targets = torch.arange(batch_size, device=image_logits.device)
        targets = targets + batch_size * self._rank

        loss = 0.5 * (
            F.cross_entropy(image_logits, targets)
            + F.cross_entropy(text_logits, targets)
        )
        output_dict = {
            "loss": loss,
            "logging": {"contrastive_loss": loss, "logit_scale": _scale},
        }
        return output_dict


class MERU(CLIPBaseline):

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):

        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std)

        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )

        self._curv_minmax = {
            "max": math.log(1.0 * 10),
            "min": math.log(1.0 / 10),
        }
        self.entail_weight = entail_weight

        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def encode_image(self, images: torch.Tensor, project: bool):

        image_feats = super().encode_image(images, project=False)

        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(image_feats, self.curv.exp())

        return image_feats

    def encode_text(self, tokens: list[torch.Tensor], project: bool):

        text_feats = super().encode_text(tokens, project=False)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(text_feats, self.curv.exp())

        return text_feats

    def forward(
        self, images: torch.Tensor,
        tokens: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        with torch.autocast(self.device.type, dtype=torch.float32):
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)

            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }


class HyCoCLIP(MERU):


    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):

        super().__init__(visual, textual, embed_dim, curv_init, learn_curv, entail_weight, pixel_mean, pixel_std)
        assert use_boxes, "HyCoCLIP requires box images and texts to function."

    def forward(
        self, images: torch.Tensor, box_images: torch.Tensor,
        tokens: list[torch.Tensor], box_tokens: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:


        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)
        _curv = self.curv.exp()

        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        with torch.autocast(self.device.type, dtype=torch.float32):
            image_logits = -L.pairwise_dist(image_feats, all_text_feats, _curv)
            text_logits = -L.pairwise_dist(text_feats, all_image_feats, _curv)
            box_image_logits = -L.pairwise_dist(box_image_feats, all_text_feats, _curv)
            box_text_logits = -L.pairwise_dist(box_text_feats, all_image_feats, _curv)

            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)
            )

            _angle = L.oxy_angle(text_feats, image_feats, _curv)
            _aperture = L.half_aperture(text_feats, _curv)

            _box_angle = L.oxy_angle(box_text_feats, box_image_feats, _curv)
            _box_aperture = L.half_aperture(box_text_feats, _curv)

            _cross_image_angle = L.oxy_angle(box_image_feats, image_feats, _curv)
            _box_image_aperture = L.half_aperture(box_image_feats, _curv)

            _cross_text_angle = L.oxy_angle(box_text_feats, text_feats, _curv)
            _box_text_aperture = L.half_aperture(box_text_feats, _curv)

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7   # inter-modal
            _local_aperture_thresh = 1.2    # intra-modal

            text_image_entailment_loss = torch.clamp(_angle - _global_aperture_thresh * _aperture, min=0).mean()
            box_text_image_entailment_loss = torch.clamp(_box_angle - _global_aperture_thresh * _box_aperture, min=0).mean()
            cross_image_entailment_loss = torch.clamp(_cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0).mean()
            cross_text_entailment_loss = torch.clamp(_cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0).mean()
            
            entailment_loss = 0.5 * (
                text_image_entailment_loss 
                + box_text_image_entailment_loss 
                + cross_image_entailment_loss 
                + cross_text_entailment_loss
            )

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "text_image_entailment_loss": text_image_entailment_loss,
                "box_text_image_entailment_loss": box_text_image_entailment_loss,
                "cross_image_entailment_loss": cross_image_entailment_loss,
                "cross_text_entailment_loss": cross_text_entailment_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }


def hycoclip(pretrained='', **kwargs):
    model = HyCoCLIP(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        print('msg', msg)
    return model