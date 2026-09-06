// Copyright 2026 ductainguyen
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "arena_people_msgs/msg/pedestrians.hpp"
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

  rclcpp::Subscription<arena_people_msgs::msg::Pedestrians>::SharedPtr peds_sub_;
  arena_people_msgs::msg::Pedestrians::SharedPtr latest_peds_;
  std::mutex peds_mutex_;

  // Parameters (see config/social_cost_layer_params.yaml for a worked example).
  std::string peds_topic_{"arena_peds"};
  double personal_space_radius_{2.5};   // meters; footprint radius for updateBounds/lookup
  double group_extra_radius_{1.5};      // extra meters beyond segment endpoints for group term
  unsigned char max_cost_value_{200};   // stays below LETHAL_OBSTACLE(254)/INSCRIBED(253)
  double stale_timeout_s_{1.0};         // ignore Pedestrians older than this
  bool enabled_{true};
  bool logged_first_message_{false};
};

}  // namespace arena_social_cost_layer
