// Copyright 2026 ductainguyen
#include <gtest/gtest.h>

#include <cmath>

#include "arena_social_cost_layer/cost_math.hpp"

using arena_social_cost_layer::groupBetweenCost;
using arena_social_cost_layer::personalSpaceCost;
using arena_social_cost_layer::pointToSegmentDistSq;
using arena_social_cost_layer::PersonalSpaceParams;
using arena_social_cost_layer::Box;
using arena_social_cost_layer::segmentBox;
using arena_social_cost_layer::unionBox;
using arena_social_cost_layer::AnisotropicParams;
using arena_social_cost_layer::anisotropicCost;
using arena_social_cost_layer::anisotropicReach;
using arena_social_cost_layer::bodyOffset;

TEST(PersonalSpaceCost, DecaysWithDistance)
{
  const PersonalSpaceParams p{};
  const double near = personalSpaceCost(0.1, 0.0, p.base_sigma, p.base_amplitude, p.shape);
  const double far = personalSpaceCost(3.0, 0.0, p.base_sigma, p.base_amplitude, p.shape);
  EXPECT_GT(near, far);
  EXPECT_GT(near, 0.0);
  EXPECT_NEAR(far, 0.0, 1e-3);
}

TEST(PersonalSpaceCost, WiderSigmaAndHigherAmplitudeRaiseCost)
{
  const PersonalSpaceParams p{};
  const double base = personalSpaceCost(1.0, 0.0, p.base_sigma, p.base_amplitude, p.shape);
  const double danger = personalSpaceCost(1.0, 0.0, p.danger_sigma, p.danger_amplitude, p.shape);
  EXPECT_GT(danger, base);
}

TEST(PersonalSpaceCost, LargerShapeFlattensTheCore)
{
  const PersonalSpaceParams p{};
  const double gaussian = personalSpaceCost(0.3, 0.0, p.base_sigma, p.base_amplitude, 1.0);
  const double plateau = personalSpaceCost(0.3, 0.0, p.base_sigma, p.base_amplitude, 3.0);
  EXPECT_GT(plateau, gaussian);
}

TEST(GroupBetweenCost, PeaksAtMidpointBetweenParticipants)
{
  const double ax = -1.0;
  const double ay = 0.0;
  const double bx = 1.0;
  const double by = 0.0;
  const double at_midpoint = groupBetweenCost(0.0, 0.0, ax, ay, bx, by, 1.0, 0.6, 1.0);
  const double off_to_the_side = groupBetweenCost(0.0, 3.0, ax, ay, bx, by, 1.0, 0.6, 1.0);
  const double past_the_pair = groupBetweenCost(5.0, 0.0, ax, ay, bx, by, 1.0, 0.6, 1.0);
  EXPECT_GT(at_midpoint, off_to_the_side);
  EXPECT_GT(at_midpoint, past_the_pair);
  EXPECT_NEAR(past_the_pair, 0.0, 1e-2);
}

TEST(GroupBetweenCost, ScalesWithTheWeight)
{
  const double light = groupBetweenCost(0.0, 0.0, -0.3, 0.0, 0.3, 0.0, 0.6, 0.6, 1.0);
  const double heavy = groupBetweenCost(0.0, 0.0, -0.3, 0.0, 0.3, 0.0, 1.5, 0.6, 1.0);
  EXPECT_DOUBLE_EQ(heavy, light * 2.5);
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

TEST(AnisotropicCost, AheadAndBehindCostMoreThanBeside)
{
  const AnisotropicParams p{};
  const double yaw = 0.7;
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (const double d : {0.5, 1.0, 1.5}) {
    const double ahead = anisotropicCost(bodyOffset(d * c, d * s, 0.0, 0.0, yaw, 0.0), p, 0.0);
    const double behind = anisotropicCost(bodyOffset(-d * c, -d * s, 0.0, 0.0, yaw, 0.0), p, 0.0);
    const double left = anisotropicCost(bodyOffset(-d * s, d * c, 0.0, 0.0, yaw, 0.0), p, 0.0);
    const double right = anisotropicCost(bodyOffset(d * s, -d * c, 0.0, 0.0, yaw, 0.0), p, 0.0);
    EXPECT_GT(ahead, left) << "d=" << d;
    EXPECT_GT(behind, left) << "d=" << d;
    EXPECT_NEAR(left, right, 1e-12);
    EXPECT_GT(ahead, behind) << "the front reaches further than the back";
  }
}

TEST(AnisotropicCost, PeaksOnTheBodyAndStopsAtTheCutoff)
{
  const AnisotropicParams p{};
  EXPECT_DOUBLE_EQ(anisotropicCost(bodyOffset(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), p, 0.0), p.amplitude);
  const double edge = p.cutoff * p.sigma_side;
  EXPECT_GT(anisotropicCost(bodyOffset(0.0, 0.99 * edge, 0.0, 0.0, 0.0, 0.0), p, 0.0), 0.0);
  EXPECT_EQ(anisotropicCost(bodyOffset(0.0, 1.01 * edge, 0.0, 0.0, 0.0, 0.0), p, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(anisotropicReach(p, 0.0), p.cutoff * p.sigma_front);
}

TEST(AnisotropicCost, SpeedStretchesOnlyTheFront)
{
  const AnisotropicParams p{};
  const auto ahead = bodyOffset(1.5, 0.0, 0.0, 0.0, 0.0, 0.0);
  const auto behind = bodyOffset(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  EXPECT_GT(anisotropicCost(ahead, p, 1.0), anisotropicCost(ahead, p, 0.0));
  EXPECT_DOUBLE_EQ(anisotropicCost(behind, p, 1.0), anisotropicCost(behind, p, 0.0));
  EXPECT_GT(anisotropicReach(p, 1.0), anisotropicReach(p, 0.0));
}

TEST(AnisotropicCost, APushedPairIsFlatAlongItsSpan)
{
  // chair front at the origin heading +x, pusher 0.7 m behind:
  // beside the pair the cost only falls off sideways
  const AnisotropicParams p{};
  const double length = 0.7;
  const double beside_chair = anisotropicCost(bodyOffset(0.0, 0.6, 0.0, 0.0, 0.0, length), p, 0.0);
  const double beside_pusher = anisotropicCost(bodyOffset(-0.7, 0.6, 0.0, 0.0, 0.0, length), p,
    0.0);
  EXPECT_DOUBLE_EQ(beside_chair, beside_pusher);
  // behind the pusher it is measured from her, not from the chair
  const double behind_pusher = anisotropicCost(bodyOffset(-0.7 - 0.6, 0.0, 0.0, 0.0, 0.0, length),
    p, 0.0);
  const double behind_lone_chair = anisotropicCost(bodyOffset(-0.6, 0.0, 0.0, 0.0, 0.0, 0.0), p,
    0.0);
  EXPECT_DOUBLE_EQ(behind_pusher, behind_lone_chair);
  EXPECT_GT(behind_pusher, beside_pusher);
}
