// Copyright 2026 ductainguyen
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "arena_people_msgs/msg/pedestrians.hpp"
#include "arena_social_cost_layer/cost_math.hpp"
#include "nav2_costmap_2d/layer.hpp"
#include "nav2_costmap_2d/layered_costmap.hpp"
#include "rclcpp/rclcpp.hpp"

namespace arena_social_cost_layer
{

/// Nav2 costmap_2d::Layer that paints social cost from arena_peds
/// (arena_people_msgs/Pedestrians): a per-pedestrian personal-space Gaussian
/// modulated by animation_state, plus an HHI group/O-space term between
/// pedestrians sharing an ACTIVE interaction_id (see cost_math.hpp for the
/// underlying math, unit-tested independently of this ROS glue).
///
/// Deliberately does not touch the robot's goal or velocity commands - HRI
/// reactive behavior (e.g. reorienting toward a pointing human) is a
/// separate, follow-on component. This layer only shapes local-costmap cost.
class SocialCostLayer : public nav2_costmap_2d::Layer
{
public:
  SocialCostLayer() = default;

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  void onFootprintChanged() override {}
  bool isClearable() override {return false;}

private:
  void pedestriansCallback(const arena_people_msgs::msg::Pedestrians::SharedPtr msg);

  /// Maps a normalized (0,1] cost (from personalSpaceCost/groupBetweenCost) to
  /// a costmap byte: applies cost_scale_/cost_bias_ before clamping to [0,1]
  /// and scaling by max_cost_value_. cost_scale_ > 1 pushes mid-range values
  /// toward saturation sooner (widening the effectively-lethal-looking area
  /// beyond the raw 3-sigma envelope); cost_bias_ raises the floor so even the
  /// Gaussian's far tail (still inside the scanned reach_ window) registers.
  unsigned char toCellCost(double normalized_cost) const;

  rclcpp::Subscription<arena_people_msgs::msg::Pedestrians>::SharedPtr peds_sub_;
  arena_people_msgs::msg::Pedestrians::SharedPtr latest_peds_;
  std::mutex peds_mutex_;

  // Parameters (see config/social_cost_layer_params.yaml for a worked example).
  std::string peds_topic_{"arena_peds"};
  // Personal-space term's HARD outer clip (meters): cost is exactly 0 beyond
  // this for every animation state - independent of sigma, so widening this
  // does not dilute the high-cost core (see personal_space_sigma_ for that).
  double personal_space_radius_{1.0};
  // Personal-space Gaussian's base_sigma (meters, IDLE/WALKING/RUNNING);
  // SURPRISED/CURIOUS/PANIC/THREATENING widen from here by fixed ratios (see
  // kAlertSigmaRatio/kDangerSigmaRatio in onInitialize). Smaller = a tighter,
  // more concentrated high-cost core within personal_space_radius_.
  double personal_space_sigma_{0.35};
  // HHI group/O-space term's HARD outer clip (meters) off the connecting
  // segment - same role as personal_space_radius_, decoupled from group_sigma_.
  double group_extra_radius_{1.0};
  // HHI group/O-space Gaussian's sigma (meters) off the connecting segment.
  double group_sigma_{0.4};
  // Super-Gaussian shape exponent shared by both terms (see cost_math.hpp's
  // PersonalSpaceParams::shape). 1.0 = standard Gaussian bell; >1.0 flattens
  // the near-center plateau (most of the disk out to ~sigma reads as high
  // cost) and steepens the cutoff past it - use this to widen the "feels
  // maxed" area instead of cranking cost_scale_, which saturates the *whole*
  // profile roughly uniformly and washes out the gradient.
  double falloff_shape_{2.0};
  unsigned char max_cost_value_{200};   // stays below LETHAL_OBSTACLE(254)/INSCRIBED(253)
  double stale_timeout_s_{1.0};         // ignore Pedestrians older than this
  // Gain/floor on the normalized (0,1] cost, applied before max_cost_value_
  // scaling (see toCellCost). Raise cost_scale_ to make DWB/etc. actually
  // steer around people instead of just nudging - a costmap layer's cost only
  // competes for trajectory choice against controller-side critic scales
  // (e.g. dwb's BaseObstacle.scale), so a value here that "looks" high may
  // still lose to those; check the controller config too.
  double cost_scale_{1.0};
  double cost_bias_{0.0};
  bool enabled_{true};
  bool logged_first_message_{false};

  // Derived from personal_space_sigma_ in onInitialize.
  PersonalSpaceParams personal_space_params_{};
  double reach_{2.0};  // scan/paint window: max(personal_space_radius_, group_extra_radius_)
};

}  // namespace arena_social_cost_layer
