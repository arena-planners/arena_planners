// Copyright 2026 ductainguyen
#include <gtest/gtest.h>

#include "arena_social_cost_layer/cost_math.hpp"

using arena_social_cost_layer::ANIM_IDLE;
using arena_social_cost_layer::ANIM_PANIC;
using arena_social_cost_layer::ANIM_WALKING;
using arena_social_cost_layer::Box;
using arena_social_cost_layer::groupBetweenCost;
using arena_social_cost_layer::INTERACTION_GROUP_CONVERSATION;
using arena_social_cost_layer::INTERACTION_HUG;
using arena_social_cost_layer::INTERACTION_TALK_TO;
using arena_social_cost_layer::INTERACTION_WAVE_AT;
using arena_social_cost_layer::interactionTypeMultiplier;
using arena_social_cost_layer::personalSpaceCost;
using arena_social_cost_layer::pointToSegmentDistSq;
using arena_social_cost_layer::segmentBox;
using arena_social_cost_layer::unionBox;

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

TEST(SegmentBox, CoversBothEndpointsGrownByReach)
{
  const Box box = segmentBox(-1.0, 2.0, 3.0, -0.5, 0.5);
  EXPECT_DOUBLE_EQ(box.min_x, -1.5);
  EXPECT_DOUBLE_EQ(box.min_y, -1.0);
  EXPECT_DOUBLE_EQ(box.max_x, 3.5);
  EXPECT_DOUBLE_EQ(box.max_y, 2.5);
}

TEST(SegmentBox, PointIsSquareOfReach)
{
  const Box box = segmentBox(1.0, -1.0, 1.0, -1.0, 2.0);
  EXPECT_DOUBLE_EQ(box.min_x, -1.0);
  EXPECT_DOUBLE_EQ(box.min_y, -3.0);
  EXPECT_DOUBLE_EQ(box.max_x, 3.0);
  EXPECT_DOUBLE_EQ(box.max_y, 1.0);
}

TEST(UnionBox, EmptyBoxIsIdentity)
{
  const Box box = segmentBox(0.0, 0.0, 1.0, 1.0, 0.0);
  const Box merged = unionBox(Box{}, box);
  EXPECT_DOUBLE_EQ(merged.min_x, 0.0);
  EXPECT_DOUBLE_EQ(merged.min_y, 0.0);
  EXPECT_DOUBLE_EQ(merged.max_x, 1.0);
  EXPECT_DOUBLE_EQ(merged.max_y, 1.0);
}

TEST(UnionBox, CoversBothBoxes)
{
  const Box a = segmentBox(-2.0, 0.0, -1.0, 0.5, 0.0);
  const Box b = segmentBox(4.0, -3.0, 5.0, 1.0, 0.0);
  const Box merged = unionBox(a, b);
  EXPECT_DOUBLE_EQ(merged.min_x, -2.0);
  EXPECT_DOUBLE_EQ(merged.min_y, -3.0);
  EXPECT_DOUBLE_EQ(merged.max_x, 5.0);
  EXPECT_DOUBLE_EQ(merged.max_y, 1.0);
}
