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
