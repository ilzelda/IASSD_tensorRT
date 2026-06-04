from contextlib import contextmanager

import onnxscript
import torch
from onnxscript import FLOAT, INT32
from onnxscript import values
from torch.library import custom_op


_EXPORT_PLACEHOLDERS_ENABLED = False
_IASSD_ONNX_OPSET = values.Opset('IASSD', 1)


@contextmanager
def export_placeholders_enabled():
    global _EXPORT_PLACEHOLDERS_ENABLED
    old_value = _EXPORT_PLACEHOLDERS_ENABLED
    _EXPORT_PLACEHOLDERS_ENABLED = True
    try:
        yield
    finally:
        _EXPORT_PLACEHOLDERS_ENABLED = old_value


def is_export_placeholders_enabled():
    return _EXPORT_PLACEHOLDERS_ENABLED


@custom_op('iassd::farthest_point_sampling', mutates_args=())
def farthest_point_sampling(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    raise RuntimeError('farthest_point_sampling은 ONNX export placeholder 전용입니다.')


@farthest_point_sampling.register_fake
def _(xyz, npoint: int):
    return torch.empty((xyz.shape[0], npoint), dtype=torch.int32, device=xyz.device)


@onnxscript.script()
def onnx_farthest_point_sampling(xyz: FLOAT[...], npoint: int) -> INT32[...]:
    return _IASSD_ONNX_OPSET.FarthestPointSampling(xyz, npoint_i=npoint)


@custom_op('iassd::gather_points', mutates_args=())
def gather_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    raise RuntimeError('gather_points는 ONNX export placeholder 전용입니다.')


@gather_points.register_fake
def _(features, idx):
    return torch.empty(
        (features.shape[0], features.shape[1], idx.shape[1]),
        dtype=features.dtype,
        device=features.device,
    )


@onnxscript.script()
def onnx_gather_points(features: FLOAT[...], idx: INT32[...]) -> FLOAT[...]:
    return _IASSD_ONNX_OPSET.GatherPoints(features, idx)


@custom_op('iassd::ball_query', mutates_args=())
def ball_query(xyz: torch.Tensor, new_xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    raise RuntimeError('ball_query는 ONNX export placeholder 전용입니다.')


@ball_query.register_fake
def _(xyz, new_xyz, radius: float, nsample: int):
    return torch.empty(
        (xyz.shape[0], new_xyz.shape[1], nsample),
        dtype=torch.int32,
        device=xyz.device,
    )


@onnxscript.script()
def onnx_ball_query(xyz: FLOAT[...], new_xyz: FLOAT[...], radius: float, nsample: int) -> INT32[...]:
    return _IASSD_ONNX_OPSET.BallQuery(xyz, new_xyz, radius_f=radius, nsample_i=nsample)


@custom_op('iassd::group_points', mutates_args=())
def group_points(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    raise RuntimeError('group_points는 ONNX export placeholder 전용입니다.')


@group_points.register_fake
def _(features, idx):
    return torch.empty(
        (features.shape[0], features.shape[1], idx.shape[1], idx.shape[2]),
        dtype=features.dtype,
        device=features.device,
    )


@onnxscript.script()
def onnx_group_points(features: FLOAT[...], idx: INT32[...]) -> FLOAT[...]:
    return _IASSD_ONNX_OPSET.GroupPoints(features, idx)


def build_iassd_onnx_registry():
    from torch.onnx._internal.exporter import _registration

    registry = _registration.ONNXRegistry.from_torchlib()
    registry.register_op(torch.ops.iassd.farthest_point_sampling.default, onnx_farthest_point_sampling)
    registry.register_op(torch.ops.iassd.gather_points.default, onnx_gather_points)
    registry.register_op(torch.ops.iassd.ball_query.default, onnx_ball_query)
    registry.register_op(torch.ops.iassd.group_points.default, onnx_group_points)
    return registry
