// Copyright 2026 ductainguyen
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace arena_social_cost_layer
{

// mirrors arena_people_msgs/Pedestrian animation_state
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

// mirrors humansim's InteractionType (core/interaction_kinds.py), index 2 is retired there
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
  double base_sigma = 0.45;
  double alert_sigma = 0.75;
  double danger_sigma = 1.2;
  double base_amplitude = 0.5;
  double alert_amplitude = 0.75;
  double danger_amplitude = 1.0;
  double shape = 1.0;  // super-Gaussian exponent, 1.0 = Gaussian
};

/// Sigma of the personal-space term for an animation state.
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

/// Amplitude of the personal-space term for an animation state.
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

/// Normalized (0,1] personal-space cost at offset (dx, dy) from a pedestrian.
inline double personalSpaceCost(
  double dx, double dy, uint8_t animation_state, const PersonalSpaceParams & p = {})
{
  const double sigma = personalSpaceSigma(animation_state, p);
  const double amplitude = personalSpaceAmplitude(animation_state, p);
  const double d2 = dx * dx + dy * dy;
  return amplitude * std::exp(-std::pow(d2 / (2.0 * sigma * sigma), p.shape));
}

/// Weight of the group term by interaction type.
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
  uint8_t interaction_type, double sigma = 0.6, double shape = 1.0)
{
  const double d2 = pointToSegmentDistSq(px, py, ax, ay, bx, by);
  const double amplitude = interactionTypeMultiplier(interaction_type);
  return amplitude * std::exp(-std::pow(d2 / (2.0 * sigma * sigma), shape));
}

}  // namespace arena_social_cost_layer
