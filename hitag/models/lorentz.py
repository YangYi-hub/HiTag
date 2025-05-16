from __future__ import annotations

import math

import torch
from torch import Tensor
from loguru import logger


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:

    c_xyl = -curv * pairwise_inner(x, y, curv)
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)

    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:

    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))

    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:

    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    _half_aperture = torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))

    return _half_aperture


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))

    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)

    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))

    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def oxy_angle_eval(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))

    logger.info(f"x_time shape: {x_time.size()}")
    logger.info(f"y_time shape: {y_time.size()}")


    c_xyl = curv * (y @ x.T - y_time @ x_time.T)
    logger.info(f"c_xyl shape: {c_xyl.size()}")

    acos_numer = y_time + c_xyl * x_time.T
    logger.info(f"acos_numer shape: {acos_numer.size()}")
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    logger.info(f"acos_denom shape: {acos_denom.size()}")

    acos_input = acos_numer / (torch.norm(x, dim=-1, keepdim=True).T * acos_denom + eps)
    _angle = - torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))

    return _angle


def batch_oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8):

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True)) 
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    
    c_xyl = curv * (x @ y.T - x_time @ y_time.T)

    acos_numer = y_time.T + c_xyl * x_time 
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))  
    
    x_norm = torch.norm(x, dim=-1, keepdim=True) 
    
    acos_input = acos_numer / (x_norm * acos_denom + eps) 
    _angle = torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))  
    
    return _angle