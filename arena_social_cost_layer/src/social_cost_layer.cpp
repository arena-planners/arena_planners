// Copyright 2026 ductainguyen
#include "arena_social_cost_layer/social_cost_layer.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "arena_social_cost_layer/cost_math.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace arena_social_cost_layer
{

void SocialCostLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node in SocialCostLayer::onInitialize"};
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("peds_topic", rclcpp::ParameterValue(std::string("arena_peds")));
  declareParameter("personal_space_radius", rclcpp::ParameterValue(2.5));
  declareParameter("group_extra_radius", rclcpp::ParameterValue(1.5));
  declareParameter("max_cost_value", rclcpp::ParameterValue(200));
  declareParameter("stale_timeout", rclcpp::ParameterValue(1.0));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".peds_topic", peds_topic_);
  node->get_parameter(name_ + ".personal_space_radius", personal_space_radius_);
  node->get_parameter(name_ + ".group_extra_radius", group_extra_radius_);
  int max_cost_param = max_cost_value_;
  node->get_parameter(name_ + ".max_cost_value", max_cost_param);
  max_cost_value_ = static_cast<unsigned char>(
    std::clamp(max_cost_param, 0, static_cast<int>(nav2_costmap_2d::LETHAL_OBSTACLE) - 1));
  node->get_parameter(name_ + ".stale_timeout", stale_timeout_s_);

  peds_sub_ = node->create_subscription<arena_people_msgs::msg::Pedestrians>(
    peds_topic_, rclcpp::SensorDataQoS(),
    std::bind(&SocialCostLayer::pedestriansCallback, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(), "SocialCostLayer '%s' initialized: enabled=%s, subscribing to '%s' "
    "(resolved: '%s')", name_.c_str(), enabled_ ? "true" : "false", peds_topic_.c_str(),
    peds_sub_->get_topic_name());

  current_ = true;
}

void SocialCostLayer::pedestriansCallback(const arena_people_msgs::msg::Pedestrians::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(peds_mutex_);
  latest_peds_ = msg;
  if (!logged_first_message_) {
    logged_first_message_ = true;
    if (auto node = node_.lock()) {
      RCLCPP_INFO(
        node->get_logger(), "SocialCostLayer '%s': first Pedestrians message received (%zu "
        "pedestrian(s)) on '%s'", name_.c_str(), msg->pedestrians.size(),
        peds_sub_->get_topic_name());
    }
  }
}

void SocialCostLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }

  arena_people_msgs::msg::Pedestrians::SharedPtr peds;
  {
    std::lock_guard<std::mutex> lock(peds_mutex_);
    peds = latest_peds_;
  }
  if (!peds) {
    return;
  }

  auto node = node_.lock();
  if (node) {
    const rclcpp::Time stamp(peds->header.stamp);
    const double age = (node->now() - stamp).seconds();
    if (age > stale_timeout_s_) {
      return;
    }
  }

  const double reach = personal_space_radius_ + group_extra_radius_;
  for (const auto & ped : peds->pedestrians) {
    const double x = ped.pose.position.x;
    const double y = ped.pose.position.y;
    *min_x = std::min(*min_x, x - reach);
    *min_y = std::min(*min_y, y - reach);
    *max_x = std::max(*max_x, x + reach);
    *max_y = std::max(*max_y, y + reach);
  }
}

void SocialCostLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  arena_people_msgs::msg::Pedestrians::SharedPtr peds;
  {
    std::lock_guard<std::mutex> lock(peds_mutex_);
    peds = latest_peds_;
  }
  if (!peds || peds->pedestrians.empty()) {
    return;
  }

  min_i = std::max(0, min_i);
  min_j = std::max(0, min_j);
  max_i = std::min(static_cast<int>(master_grid.getSizeInCellsX()), max_i);
  max_j = std::min(static_cast<int>(master_grid.getSizeInCellsY()), max_j);
  if (min_i >= max_i || min_j >= max_j) {
    return;
  }

  // Group pedestrians sharing an ACTIVE interaction_id, for the HHI
  // between-participants term. interaction_id < 0 means "no active
  // interaction" (see arena_people_msgs/Pedestrian.msg / AgentState.msg).
  std::vector<std::pair<int32_t, uint8_t>> group_ids;  // (interaction_id, interaction_type)
  std::vector<std::vector<size_t>> group_members;
  for (size_t idx = 0; idx < peds->pedestrians.size(); ++idx) {
    const auto & ped = peds->pedestrians[idx];
    if (ped.interaction_id < 0) {
      continue;
    }
    auto it = std::find_if(
      group_ids.begin(), group_ids.end(),
      [&](const auto & g) {return g.first == ped.interaction_id;});
    if (it == group_ids.end()) {
      group_ids.emplace_back(ped.interaction_id, ped.interaction_type);
      group_members.emplace_back(std::vector<size_t>{idx});
    } else {
      group_members[std::distance(group_ids.begin(), it)].push_back(idx);
    }
  }

  const double resolution = master_grid.getResolution();
  const double reach = personal_space_radius_ + group_extra_radius_;
  const int cell_reach = static_cast<int>(std::ceil(reach / resolution));

  for (const auto & ped : peds->pedestrians) {
    unsigned int mx, my;
    if (!master_grid.worldToMap(ped.pose.position.x, ped.pose.position.y, mx, my)) {
      continue;
    }
    const int i0 = std::max(min_i, static_cast<int>(mx) - cell_reach);
    const int i1 = std::min(max_i, static_cast<int>(mx) + cell_reach);
    const int j0 = std::max(min_j, static_cast<int>(my) - cell_reach);
    const int j1 = std::min(max_j, static_cast<int>(my) + cell_reach);

    for (int j = j0; j < j1; ++j) {
      for (int i = i0; i < i1; ++i) {
        double wx, wy;
        master_grid.mapToWorld(i, j, wx, wy);
        const double dx = wx - ped.pose.position.x;
        const double dy = wy - ped.pose.position.y;
        const double cost = personalSpaceCost(dx, dy, ped.animation_state);
        if (cost <= 0.0) {
          continue;
        }
        const unsigned char cell_cost =
          static_cast<unsigned char>(std::clamp(cost, 0.0, 1.0) * max_cost_value_);
        const unsigned char existing = master_grid.getCost(i, j);
        if (existing == nav2_costmap_2d::NO_INFORMATION || cell_cost > existing) {
          master_grid.setCost(i, j, cell_cost);
        }
      }
    }
  }

  // HHI between-participants term: for each group with >=2 members, paint the
  // O-space along every participant pair's connecting segment.
  for (size_t g = 0; g < group_members.size(); ++g) {
    const auto & members = group_members[g];
    if (members.size() < 2) {
      continue;
    }
    const uint8_t interaction_type = group_ids[g].second;
    for (size_t a = 0; a < members.size(); ++a) {
      for (size_t b = a + 1; b < members.size(); ++b) {
        const auto & pa = peds->pedestrians[members[a]].pose.position;
        const auto & pb = peds->pedestrians[members[b]].pose.position;

        unsigned int mx0, my0, mx1, my1;
        const bool ok0 = master_grid.worldToMap(pa.x, pa.y, mx0, my0);
        const bool ok1 = master_grid.worldToMap(pb.x, pb.y, mx1, my1);
        if (!ok0 || !ok1) {
          continue;
        }
        const int i0 = std::max(min_i, static_cast<int>(std::min(mx0, mx1)) - cell_reach);
        const int i1 = std::min(max_i, static_cast<int>(std::max(mx0, mx1)) + cell_reach);
        const int j0 = std::max(min_j, static_cast<int>(std::min(my0, my1)) - cell_reach);
        const int j1 = std::min(max_j, static_cast<int>(std::max(my0, my1)) + cell_reach);

        for (int j = j0; j < j1; ++j) {
          for (int i = i0; i < i1; ++i) {
            double wx, wy;
            master_grid.mapToWorld(i, j, wx, wy);
            const double cost = groupBetweenCost(
              wx, wy, pa.x, pa.y, pb.x, pb.y, interaction_type);
            if (cost <= 0.0) {
              continue;
            }
            const unsigned char cell_cost =
              static_cast<unsigned char>(std::clamp(cost, 0.0, 1.0) * max_cost_value_);
            const unsigned char existing = master_grid.getCost(i, j);
            if (existing == nav2_costmap_2d::NO_INFORMATION || cell_cost > existing) {
              master_grid.setCost(i, j, cell_cost);
            }
          }
        }
      }
    }
  }
}

void SocialCostLayer::reset()
{
  std::lock_guard<std::mutex> lock(peds_mutex_);
  latest_peds_.reset();
}

}  // namespace arena_social_cost_layer

PLUGINLIB_EXPORT_CLASS(arena_social_cost_layer::SocialCostLayer, nav2_costmap_2d::Layer)
