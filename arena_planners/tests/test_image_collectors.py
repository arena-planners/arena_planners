"""Tests for ImageCollector and CompressedImageCollector."""

from __future__ import annotations

import numpy as np
import pytest

sensor_msgs = pytest.importorskip("sensor_msgs.msg")
cv2 = pytest.importorskip("cv2")

from arena_planners.observations.data_sources.collectors import (  # noqa: E402
    CompressedImageCollector,
    ImageCollector,
)
from arena_planners.observations.pipeline import _TYPE_REGISTRY  # noqa: E402


def _make_image_msg(arr: np.ndarray, encoding: str) -> sensor_msgs.Image:
    msg = sensor_msgs.Image()
    msg.height = arr.shape[0]
    msg.width = arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.shape[1] * arr.dtype.itemsize * (arr.shape[2] if arr.ndim == 3 else 1)
    msg.data = arr.tobytes()
    return msg


def test_image_collector_rgb8_passthrough():
    c = ImageCollector("rgb", topic="/cam")
    arr = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    out = c._preprocess(_make_image_msg(arr, "rgb8"))
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, arr)


def test_image_collector_mono8_squeezes_channel():
    c = ImageCollector("g", topic="/cam")
    arr = np.arange(8 * 16, dtype=np.uint8).reshape(8, 16)
    out = c._preprocess(_make_image_msg(arr[..., None], "mono8"))
    assert out.shape == (8, 16)
    assert out.dtype == np.uint8


def test_image_collector_resize_uint8_downscale():
    c = ImageCollector("rgb", topic="/cam", output_size=[128, 128])
    arr = (np.random.default_rng(0).integers(0, 256, size=(480, 640, 3))).astype(np.uint8)
    out = c._preprocess(_make_image_msg(arr, "rgb8"))
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8


def test_image_collector_resize_depth_inter_nearest():
    """Depth must use INTER_NEAREST so output values are only ever exact input pixels."""
    c = ImageCollector("d", topic="/cam", output_size=[16, 16])
    # use distinct values so any interpolation would be detectable
    arr = (np.arange(32 * 32, dtype=np.uint16) * 7).reshape(32, 32)
    out = c._preprocess(_make_image_msg(arr[..., None], "16UC1"))
    assert out.shape == (16, 16)
    assert out.dtype == np.uint16
    valid = set(arr.flatten().tolist())
    assert set(out.flatten().tolist()).issubset(valid)


def test_image_collector_normalize_unit():
    c = ImageCollector("rgb", topic="/cam", normalize="unit")
    arr = np.full((4, 4, 3), 255, dtype=np.uint8)
    out = c._preprocess(_make_image_msg(arr, "rgb8"))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 1.0)


def test_image_collector_normalize_imagenet_returns_float32_shape():
    c = ImageCollector("rgb", topic="/cam", normalize="imagenet")
    arr = (np.random.default_rng(1).integers(0, 256, size=(8, 8, 3))).astype(np.uint8)
    out = c._preprocess(_make_image_msg(arr, "rgb8"))
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32


def test_image_collector_normalize_depth_mm_to_m():
    c = ImageCollector("d", topic="/cam", normalize="depth_mm_to_m")
    arr = np.full((4, 4), 2500, dtype=np.uint16)  # 2.5 m
    out = c._preprocess(_make_image_msg(arr[..., None], "16UC1"))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 2.5)


def test_image_collector_normalize_depth_clip():
    """Depth normalization clips to [0, 10] m to discard sensor max-range garbage."""
    c = ImageCollector("d", topic="/cam", normalize="depth_mm_to_m")
    arr = np.array([[50000, 0], [3000, 65535]], dtype=np.uint16)
    out = c._preprocess(_make_image_msg(arr[..., None], "16UC1"))
    np.testing.assert_allclose(out, np.array([[10.0, 0.0], [3.0, 10.0]], dtype=np.float32))


def test_image_collector_encoding_assert_pass():
    c = ImageCollector("rgb", topic="/cam", encoding="rgb8")
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    c._preprocess(_make_image_msg(arr, "rgb8"))


def test_image_collector_encoding_mismatch_raises():
    c = ImageCollector("rgb", topic="/cam", encoding="rgb8")
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="encoding mismatch"):
        c._preprocess(_make_image_msg(arr, "bgr8"))


def test_image_collector_unsupported_encoding_raises():
    c = ImageCollector("rgb", topic="/cam")
    msg = sensor_msgs.Image()
    msg.height = 4
    msg.width = 4
    msg.encoding = "yuv422"
    msg.data = b"\x00" * 32
    with pytest.raises(ValueError, match="unsupported encoding"):
        c._preprocess(msg)


def test_image_collector_empty_returns_empty():
    c = ImageCollector("rgb", topic="/cam")
    msg = sensor_msgs.Image()
    out = c._preprocess(msg)
    assert out.shape == (0,)


def test_compressed_image_decode_jpeg():
    c = CompressedImageCollector("rgb", topic="/cam/compressed")
    arr = (np.random.default_rng(2).integers(0, 256, size=(48, 64, 3))).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", arr)
    assert ok
    msg = sensor_msgs.CompressedImage()
    msg.format = "jpeg"
    msg.data = buf.tobytes()
    out = c._preprocess(msg)
    assert out.shape == (48, 64, 3)
    assert out.dtype == np.uint8


def test_compressed_image_decode_with_resize_and_normalize():
    c = CompressedImageCollector("rgb", topic="/cam/compressed", output_size=[16, 16], normalize="unit")
    arr = np.full((32, 32, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    msg = sensor_msgs.CompressedImage()
    msg.format = "png"
    msg.data = buf.tobytes()
    out = c._preprocess(msg)
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_compressed_image_empty_returns_empty():
    c = CompressedImageCollector("rgb", topic="/cam/compressed")
    msg = sensor_msgs.CompressedImage()
    out = c._preprocess(msg)
    assert out.shape == (0,)


def test_compressed_image_corrupt_raises():
    c = CompressedImageCollector("rgb", topic="/cam/compressed")
    msg = sensor_msgs.CompressedImage()
    msg.format = "jpeg"
    msg.data = b"\xff\xd8\xff\xe0not-actually-a-jpeg"
    with pytest.raises(ValueError, match="cv2.imdecode returned None"):
        c._preprocess(msg)


def test_normalize_unknown_mode_raises():
    c = ImageCollector("rgb", topic="/cam", normalize="bogus")
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown normalize mode"):
        c._preprocess(_make_image_msg(arr, "rgb8"))


def test_registry_contains_image_types():
    assert _TYPE_REGISTRY["sensor_msgs/Image"] is ImageCollector
    assert _TYPE_REGISTRY["sensor_msgs/CompressedImage"] is CompressedImageCollector
