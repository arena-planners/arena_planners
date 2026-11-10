// Copyright 2026 ductainguyen
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace arena_social_cost_layer
{

struct PersonalSpaceParams
{
  double base_sigma = 0.45;
  double alert_sigma = 0.75;
  double danger_sigma = 1.2;
  double base_amplitude = 0.5;
  double alert_amplitude = 0.75;
  double danger_amplitude = 1.0;
  double shape = 1.0;  // super-Gaussian exponent, 1.0 = Gaussian
};

/// Normalized (0,1] super-Gaussian cost at offset (dx, dy), shape 1.0 is a Gaussian.
inline double personalSpaceCost(double dx, double dy, double sigma, double amplitude, double shape)
{
  const double d2 = dx * dx + dy * dy;
  return amplitude * std::exp(-std::pow(d2 / (2.0 * sigma * sigma), shape));
}

/// Squared distance from (px,py) to the segment [(ax,ay),(bx,by)].
inline double pointToSegmentDistSq(
  double px, double py, double ax, double ay, double bx, double by)
{
  const double abx = bx - ax;
  const double aby = by - ay;
  const double ab2 = abx * abx + aby * aby;
  double t = 0.0;
  if (ab2 > 1e-9) {
    t = ((px - ax) * abx + (py - ay) * aby) / ab2;
    t = std::clamp(t, 0.0, 1.0);
  }
  const double cx = ax + t * abx;
  const double cy = ay + t * aby;
  const double dx = px - cx;
  const double dy = py - cy;
  return dx * dx + dy * dy;
}

/// Normalized (0,1] cost of (px,py) near the segment between two interaction participants.
inline double groupBetweenCost(
  double px, double py, double ax, double ay, double bx, double by,
  double weight, double sigma, double shape)
{
  const double d2 = pointToSegmentDistSq(px, py, ax, ay, bx, by);
  return weight * std::exp(-std::pow(d2 / (2.0 * sigma * sigma), shape));
}

/// Anisotropic Gaussian around a wheelchair (or a pushed pair), aligned with its heading:
/// sigma_front ahead of it, sigma_back behind it and sigma_side beside it, the front widening with
/// speed. Front and back above side make passing ahead of or behind the chair costlier than passing
/// beside it.
struct AnisotropicParams
{
  double sigma_front = 1.2;  // m ahead of the front at rest
  double sigma_back = 0.9;  // m behind the back
  double sigma_side = 0.45;  // m beside the body
  double front_speed_gain = 0.5;  // s, sigma_front grows by this times the speed
  double amplitude = 1.0;
  double shape = 1.0;  // super-Gaussian exponent, 1.0 = Gaussian
  double cutoff = 2.5;  // sigmas, nothing is painted further out
};

/// Offset of a point from a body that spans `length` m behind its front point (fx, fy) along
/// heading yaw: `along` > 0 ahead of the front, < 0 behind the back, 0 beside the body; `side`
/// across it.
struct BodyOffset
{
  double along;
  double side;
};

inline BodyOffset bodyOffset(
  double px, double py, double fx, double fy, double yaw, double length)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  const double u = (px - fx) * c + (py - fy) * s;
  const double v = -(px - fx) * s + (py - fy) * c;
  const double along = u > 0.0 ? u : (u < -length ? u + length : 0.0);
  return {along, v};
}

/// sigma_front at a speed (m/s), never below the one at rest.
inline double frontSigma(const AnisotropicParams & p, double speed)
{
  return p.sigma_front + p.front_speed_gain * std::max(speed, 0.0);
}

/// Distance from the body past which anisotropicCost is 0, for bounding boxes.
inline double anisotropicReach(const AnisotropicParams & p, double speed)
{
  return p.cutoff * std::max({frontSigma(p, speed), p.sigma_back, p.sigma_side});
}

/// Normalized cost at a body offset, 0 beyond `cutoff` sigmas.
inline double anisotropicCost(const BodyOffset & o, const AnisotropicParams & p, double speed)
{
  const double s_along = o.along > 0.0 ? frontSigma(p, speed) : p.sigma_back;
  const double q = (o.along * o.along) / (s_along * s_along) +
    (o.side * o.side) / (p.sigma_side * p.sigma_side);
  if (q > p.cutoff * p.cutoff) {
    return 0.0;
  }
  return p.amplitude * std::exp(-std::pow(q / 2.0, p.shape));
}

struct Box
{
  double min_x = std::numeric_limits<double>::max();
  double min_y = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double max_y = std::numeric_limits<double>::lowest();
};

/// Box around the segment [(ax,ay),(bx,by)] grown by reach on every side.
inline Box segmentBox(double ax, double ay, double bx, double by, double reach)
{
  Box box;
  box.min_x = std::min(ax, bx) - reach;
  box.min_y = std::min(ay, by) - reach;
  box.max_x = std::max(ax, bx) + reach;
  box.max_y = std::max(ay, by) + reach;
  return box;
}

/// Smallest box holding both, a default-constructed Box is the empty box.
inline Box unionBox(const Box & a, const Box & b)
{
  Box box;
  box.min_x = std::min(a.min_x, b.min_x);
  box.min_y = std::min(a.min_y, b.min_y);
  box.max_x = std::max(a.max_x, b.max_x);
  box.max_y = std::max(a.max_y, b.max_y);
  return box;
}

}  // namespace arena_social_cost_layer
