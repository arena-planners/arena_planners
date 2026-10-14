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

/// Nav2 costmap layer painting personal-space and interaction-group cost from arena_peds.
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

  /// Maps a normalized cost through cost_scale_ and cost_bias_ onto [0, max_cost_value_].
  unsigned char toCellCost(double normalized_cost) const;

  rclcpp::Subscription<arena_people_msgs::msg::Pedestrians>::SharedPtr peds_sub_;
  arena_people_msgs::msg::Pedestrians::SharedPtr latest_peds_;
  std::mutex peds_mutex_;

  std::string peds_topic_{"arena_peds"};
  double personal_space_radius_{1.0};
  double personal_space_sigma_{0.35};
  double group_extra_radius_{1.0};
  double group_sigma_{0.4};
  double falloff_shape_{2.0};
  unsigned char max_cost_value_{200};
  double stale_timeout_s_{1.0};
  double cost_scale_{1.0};
  double cost_bias_{0.0};
  bool enabled_{true};
  bool logged_first_message_{false};

  PersonalSpaceParams personal_space_params_{};
  double reach_{2.0};  // max(personal_space_radius_, group_extra_radius_)
};

}  // namespace arena_social_cost_layer
