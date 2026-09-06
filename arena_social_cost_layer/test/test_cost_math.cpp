// Copyright 2026 ductainguyen
#include <gtest/gtest.h>

#include "arena_social_cost_layer/cost_math.hpp"

using arena_social_cost_layer::ANIM_IDLE;
using arena_social_cost_layer::ANIM_PANIC;
using arena_social_cost_layer::ANIM_WALKING;
using arena_social_cost_layer::groupBetweenCost;
using arena_social_cost_layer::INTERACTION_GROUP_CONVERSATION;
using arena_social_cost_layer::INTERACTION_HUG;
using arena_social_cost_layer::INTERACTION_TALK_TO;
using arena_social_cost_layer::INTERACTION_WAVE_AT;
using arena_social_cost_layer::interactionTypeMultiplier;
using arena_social_cost_layer::personalSpaceCost;
using arena_social_cost_layer::pointToSegmentDistSq;

TEST(PersonalSpaceCost, DecaysWithDistance)
{
  const double near = personalSpaceCost(0.1, 0.0, ANIM_WALKING);
  const double far = personalSpaceCost(3.0, 0.0, ANIM_WALKING);
  EXPECT_GT(near, far);
  EXPECT_GT(near, 0.0);
  EXPECT_NEAR(far, 0.0, 1e-3);
}

TEST(PersonalSpaceCost, PanicWidensAndRaisesCost)
{
  // Same offset, further out - a panicking/threatening human should still
  // carry meaningfully more cost there than a walking one (wider sigma).
  const double dx = 1.0;
  const double dy = 0.0;
  const double walking = personalSpaceCost(dx, dy, ANIM_WALKING);
  const double panic = personalSpaceCost(dx, dy, ANIM_PANIC);
  EXPECT_GT(panic, walking);
}

TEST(PersonalSpaceCost, IdleAndWalkingMatchBaseline)
{
  EXPECT_DOUBLE_EQ(
    personalSpaceCost(0.5, 0.5, ANIM_IDLE),
    personalSpaceCost(0.5, 0.5, ANIM_WALKING));
}

TEST(InteractionTypeMultiplier, HugExceedsWaveAndTalk)
{
  const double hug = interactionTypeMultiplier(INTERACTION_HUG);
  const double wave = interactionTypeMultiplier(INTERACTION_WAVE_AT);
  const double talk = interactionTypeMultiplier(INTERACTION_TALK_TO);
  const double group = interactionTypeMultiplier(INTERACTION_GROUP_CONVERSATION);
  EXPECT_GT(hug, wave);
  EXPECT_GT(hug, talk);
  EXPECT_GT(group, wave);
}

TEST(GroupBetweenCost, PeaksAtMidpointBetweenParticipants)
{
  // Two participants 2m apart on the x axis.
  const double ax = -1.0;
  const double ay = 0.0;
  const double bx = 1.0;
  const double by = 0.0;

  const double at_midpoint = groupBetweenCost(0.0, 0.0, ax, ay, bx, by, INTERACTION_TALK_TO);
  const double off_to_the_side = groupBetweenCost(0.0, 3.0, ax, ay, bx, by, INTERACTION_TALK_TO);
  const double past_the_pair = groupBetweenCost(5.0, 0.0, ax, ay, bx, by, INTERACTION_TALK_TO);

  EXPECT_GT(at_midpoint, off_to_the_side);
  EXPECT_GT(at_midpoint, past_the_pair);
  EXPECT_NEAR(past_the_pair, 0.0, 1e-2);
}

TEST(GroupBetweenCost, HugCostsMoreThanTalkAtSameGeometry)
{
  const double ax = -0.3;
  const double ay = 0.0;
  const double bx = 0.3;
  const double by = 0.0;
  const double talk = groupBetweenCost(0.0, 0.0, ax, ay, bx, by, INTERACTION_TALK_TO);
  const double hug = groupBetweenCost(0.0, 0.0, ax, ay, bx, by, INTERACTION_HUG);
  EXPECT_GT(hug, talk);
}

TEST(PointToSegmentDistSq, ZeroOnEndpointsAndSegment)
{
  EXPECT_NEAR(pointToSegmentDistSq(0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0, 1e-9);
  EXPECT_NEAR(pointToSegmentDistSq(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0, 1e-9);
  EXPECT_NEAR(pointToSegmentDistSq(0.5, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0, 1e-9);
  EXPECT_NEAR(pointToSegmentDistSq(0.5, 2.0, 0.0, 0.0, 1.0, 0.0), 4.0, 1e-9);
}
