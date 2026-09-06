// Copyright 2026 ductainguyen
//
// Pure cost-function math for SocialCostLayer, kept free of ROS/costmap types
// so it can be unit-tested without a live costmap or node (see test/test_cost_math.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace arena_social_cost_layer
{

// Mirrors arena_people_msgs/Pedestrian animation_state enum (IDLE=0, WALKING=1,
// RUNNING=2, PANIC=3, SURPRISED=4, CURIOUS=5, THREATENING=6).
enum AnimationState : uint8_t
{
  ANIM_IDLE = 0,
  ANIM_WALKING = 1,
  ANIM_RUNNING = 2,
  ANIM_PANIC = 3,
  ANIM_SURPRISED = 4,
  ANIM_CURIOUS = 5,
  ANIM_THREATENING = 6,
};

// Mirrors humansim's InteractionType (core/interaction_kinds.py). Index 2 is
// intentionally unused there (never reassign/reuse an index), kept unused here too.
enum InteractionType : uint8_t
{
  INTERACTION_TALK_TO = 0,
  INTERACTION_GROUP_CONVERSATION = 1,
  INTERACTION_SIT_ON = 3,
  INTERACTION_LIE_ON = 4,
  INTERACTION_USE = 5,
  INTERACTION_QUEUE_USE = 6,
  INTERACTION_WAVE_AT = 7,
  INTERACTION_BLOCK = 8,
  INTERACTION_SERVICE = 9,
  INTERACTION_HUG = 10,
  INTERACTION_SHAKE_HAND = 11,
};

struct PersonalSpaceParams
{
  double base_sigma = 0.45;   // meters, IDLE/WALKING/RUNNING
  double alert_sigma = 0.75;  // meters, SURPRISED/CURIOUS
  double danger_sigma = 1.2;  // meters, PANIC/THREATENING
  double base_amplitude = 0.5;
  double alert_amplitude = 0.75;
  double danger_amplitude = 1.0;
};

/// Gaussian falloff radius (sigma, meters) for a human's personal-space term,
/// scaled by their current animation_state - a panicking/threatening human
/// gets a wider berth than one idling or walking normally.
inline double personalSpaceSigma(uint8_t animation_state, const PersonalSpaceParams & p = {})
{
  switch (animation_state) {
    case ANIM_PANIC:
    case ANIM_THREATENING:
      return p.danger_sigma;
    case ANIM_SURPRISED:
    case ANIM_CURIOUS:
      return p.alert_sigma;
    default:
      return p.base_sigma;
  }
}

inline double personalSpaceAmplitude(uint8_t animation_state, const PersonalSpaceParams & p = {})
{
  switch (animation_state) {
    case ANIM_PANIC:
    case ANIM_THREATENING:
      return p.danger_amplitude;
    case ANIM_SURPRISED:
    case ANIM_CURIOUS:
      return p.alert_amplitude;
    default:
      return p.base_amplitude;
  }
}

/// Normalized (0,1] personal-space cost at offset (dx, dy) meters from a
/// pedestrian, given their animation_state.
inline double personalSpaceCost(
  double dx, double dy, uint8_t animation_state, const PersonalSpaceParams & p = {})
{
  const double sigma = personalSpaceSigma(animation_state, p);
  const double amplitude = personalSpaceAmplitude(animation_state, p);
  const double d2 = dx * dx + dy * dy;
  return amplitude * std::exp(-d2 / (2.0 * sigma * sigma));
}

/// Relative weight of the HHI group/O-space term by interaction type. Kept
/// separate from personal-space so the two can be combined (max) or summed by
/// the caller. HUG/GROUP_CONVERSATION ask for more clearance than a plain
/// TALK_TO/WAVE_AT; unlisted types fall back to a modest default.
inline double interactionTypeMultiplier(uint8_t interaction_type)
{
  switch (interaction_type) {
    case INTERACTION_HUG:
      return 1.5;
    case INTERACTION_GROUP_CONVERSATION:
      return 1.2;
    case INTERACTION_QUEUE_USE:
      return 1.0;
    case INTERACTION_TALK_TO:
      return 0.9;
    case INTERACTION_WAVE_AT:
      return 0.6;
    default:
      return 0.8;
  }
}

/// Squared distance from point (px,py) to the segment [(ax,ay),(bx,by)].
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

/// Normalized (0,1] cost of point (px,py) being inside/near the O-space
/// spanned between two interaction participants at (ax,ay) and (bx,by):
/// a Gaussian falloff off the connecting segment (not just off each
/// endpoint), so "passing between the two people" is penalized specifically,
/// scaled by `interactionTypeMultiplier`.
inline double groupBetweenCost(
  double px, double py, double ax, double ay, double bx, double by,
  uint8_t interaction_type, double sigma = 0.6)
{
  const double d2 = pointToSegmentDistSq(px, py, ax, ay, bx, by);
  const double amplitude = interactionTypeMultiplier(interaction_type);
  return amplitude * std::exp(-d2 / (2.0 * sigma * sigma));
}

}  // namespace arena_social_cost_layer
