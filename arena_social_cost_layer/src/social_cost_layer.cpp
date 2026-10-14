// Copyright 2026 ductainguyen
#include "arena_social_cost_layer/social_cost_layer.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "arena_social_cost_layer/cost_math.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace arena_social_cost_layer
{

namespace
{

struct InteractionGroup
{
  uint8_t interaction_type;
  std::vector<size_t> members;
};

std::vector<InteractionGroup> interactionGroups(const arena_people_msgs::msg::Pedestrians & peds)
{
  std::vector<int32_t> ids;
  std::vector<InteractionGroup> groups;
  for (size_t idx = 0; idx < peds.pedestrians.size(); ++idx) {
    const auto & ped = peds.pedestrians[idx];
    if (ped.interaction_id < 0) {
      continue;
    }
    const auto it = std::find(ids.begin(), ids.end(), ped.interaction_id);
    if (it == ids.end()) {
      ids.push_back(ped.interaction_id);
      groups.push_back({ped.interaction_type, {idx}});
    } else {
      groups[std::distance(ids.begin(), it)].members.push_back(idx);
    }
  }
  return groups;
}

struct CellWindow
{
  int i0;
  int j0;
  int i1;
  int j1;
};

/// Cells of a world box inside the grid and [min_i,max_i) x [min_j,max_j), half-open.
CellWindow cellWindow(
  const nav2_costmap_2d::Costmap2D & grid, const Box & box,
  int min_i, int min_j, int max_i, int max_j)
{
  CellWindow w{};
  grid.worldToMapEnforceBounds(box.min_x, box.min_y, w.i0, w.j0);
  grid.worldToMapEnforceBounds(box.max_x, box.max_y, w.i1, w.j1);
  w.i0 = std::max(min_i, w.i0);
  w.j0 = std::max(min_j, w.j0);
  w.i1 = std::min(max_i, w.i1 + 1);
  w.j1 = std::min(max_j, w.j1 + 1);
  return w;
}

}  // namespace

void SocialCostLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error{"Failed to lock node in SocialCostLayer::onInitialize"};
  }

  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("peds_topic", rclcpp::ParameterValue(peds_topic_));
  declareParameter("personal_space_radius", rclcpp::ParameterValue(personal_space_radius_));
  declareParameter("personal_space_sigma", rclcpp::ParameterValue(personal_space_sigma_));
  declareParameter("group_extra_radius", rclcpp::ParameterValue(group_extra_radius_));
  declareParameter("group_sigma", rclcpp::ParameterValue(group_sigma_));
  declareParameter("falloff_shape", rclcpp::ParameterValue(falloff_shape_));
  declareParameter("max_cost_value", rclcpp::ParameterValue(static_cast<int>(max_cost_value_)));
  declareParameter("stale_timeout", rclcpp::ParameterValue(stale_timeout_s_));
  declareParameter("cost_scale", rclcpp::ParameterValue(cost_scale_));
  declareParameter("cost_bias", rclcpp::ParameterValue(cost_bias_));
  declareParameter("interaction_weights", rclcpp::ParameterValue(interaction_weights_));

  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".peds_topic", peds_topic_);
  node->get_parameter(name_ + ".personal_space_radius", personal_space_radius_);
  node->get_parameter(name_ + ".personal_space_sigma", personal_space_sigma_);
  node->get_parameter(name_ + ".group_extra_radius", group_extra_radius_);
  node->get_parameter(name_ + ".group_sigma", group_sigma_);
  node->get_parameter(name_ + ".falloff_shape", falloff_shape_);
  int max_cost_param = max_cost_value_;
  node->get_parameter(name_ + ".max_cost_value", max_cost_param);
  constexpr int kMaxCost = static_cast<int>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) - 1;
  max_cost_value_ = static_cast<unsigned char>(std::clamp(max_cost_param, 0, kMaxCost));
  node->get_parameter(name_ + ".stale_timeout", stale_timeout_s_);
  node->get_parameter(name_ + ".cost_scale", cost_scale_);
  node->get_parameter(name_ + ".cost_bias", cost_bias_);
  node->get_parameter(name_ + ".interaction_weights", interaction_weights_);

  const PersonalSpaceParams defaults{};
  const double sigma_scale = personal_space_sigma_ / defaults.base_sigma;
  personal_space_params_.base_sigma = personal_space_sigma_;
  personal_space_params_.alert_sigma = defaults.alert_sigma * sigma_scale;
  personal_space_params_.danger_sigma = defaults.danger_sigma * sigma_scale;
  personal_space_params_.shape = falloff_shape_;

  reach_ = std::max(personal_space_radius_, group_extra_radius_);

  auto sub_opt = rclcpp::SubscriptionOptions();
  sub_opt.callback_group = callback_group_;
  peds_sub_ = node->create_subscription<arena_people_msgs::msg::Pedestrians>(
    peds_topic_, rclcpp::SensorDataQoS(),
    std::bind(&SocialCostLayer::pedestriansCallback, this, std::placeholders::_1), sub_opt);

  RCLCPP_INFO(
    node->get_logger(), "SocialCostLayer '%s' initialized: enabled=%s, subscribing to '%s' "
    "(resolved: '%s')", name_.c_str(), enabled_ ? "true" : "false", peds_topic_.c_str(),
    peds_sub_->get_topic_name());

  current_ = true;
}

double SocialCostLayer::interactionWeight(uint8_t interaction_type) const
{
  if (interaction_type >= interaction_weights_.size()) {
    return 0.0;
  }
  return interaction_weights_[interaction_type];
}

double SocialCostLayer::personalSpaceSigma(uint8_t animation_state) const
{
  using Pedestrian = arena_people_msgs::msg::Pedestrian;
  switch (animation_state) {
    case Pedestrian::PANIC:
    case Pedestrian::THREATENING:
      return personal_space_params_.danger_sigma;
    case Pedestrian::SURPRISED:
    case Pedestrian::CURIOUS:
      return personal_space_params_.alert_sigma;
    default:
      return personal_space_params_.base_sigma;
  }
}

double SocialCostLayer::personalSpaceAmplitude(uint8_t animation_state) const
{
  using Pedestrian = arena_people_msgs::msg::Pedestrian;
  switch (animation_state) {
    case Pedestrian::PANIC:
    case Pedestrian::THREATENING:
      return personal_space_params_.danger_amplitude;
    case Pedestrian::SURPRISED:
    case Pedestrian::CURIOUS:
      return personal_space_params_.alert_amplitude;
    default:
      return personal_space_params_.base_amplitude;
  }
}

unsigned char SocialCostLayer::toCellCost(double normalized_cost) const
{
  const double scaled = std::clamp(normalized_cost * cost_scale_ + cost_bias_, 0.0, 1.0);
  return static_cast<unsigned char>(scaled * max_cost_value_);
}

void SocialCostLayer::pedestriansCallback(const arena_people_msgs::msg::Pedestrians::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(peds_mutex_);
  latest_peds_ = msg;
  if (auto node = node_.lock()) {
    last_peds_arrival_ = node->now();
  }
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

arena_people_msgs::msg::Pedestrians::SharedPtr SocialCostLayer::currentPedestrians()
{
  arena_people_msgs::msg::Pedestrians::SharedPtr peds;
  rclcpp::Time arrival;
  {
    std::lock_guard<std::mutex> lock(peds_mutex_);
    peds = latest_peds_;
    arrival = last_peds_arrival_;
  }
  if (!peds) {
    return nullptr;
  }
  if (auto node = node_.lock()) {
    const double age = (node->now() - arrival).seconds();
    if (age > stale_timeout_s_) {
      RCLCPP_WARN_THROTTLE(
        node->get_logger(), *node->get_clock(), 5000,
        "SocialCostLayer '%s': pedestrians last arrived %.1f s ago, past stale_timeout %.1f s, "
        "painting nothing", name_.c_str(), age, stale_timeout_s_);
      return nullptr;
    }
  }
  return peds;
}

void SocialCostLayer::updateBounds(
  double /*robot_x*/, double /*robot_y*/, double /*robot_yaw*/,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_) {
    return;
  }

  Box box{};
  if (const auto peds = currentPedestrians()) {
    for (const auto & ped : peds->pedestrians) {
      const auto & p = ped.pose.position;
      box = unionBox(box, segmentBox(p.x, p.y, p.x, p.y, reach_));
    }
    for (const auto & group : interactionGroups(*peds)) {
      if (interactionWeight(group.interaction_type) <= 0.0) {
        continue;
      }
      for (size_t a = 0; a < group.members.size(); ++a) {
        for (size_t b = a + 1; b < group.members.size(); ++b) {
          const auto & pa = peds->pedestrians[group.members[a]].pose.position;
          const auto & pb = peds->pedestrians[group.members[b]].pose.position;
          box = unionBox(box, segmentBox(pa.x, pa.y, pb.x, pb.y, group_extra_radius_));
        }
      }
    }
  }

  const Box request = unionBox(box, last_box_);
  last_box_ = box;
  *min_x = std::min(*min_x, request.min_x);
  *min_y = std::min(*min_y, request.min_y);
  *max_x = std::max(*max_x, request.max_x);
  *max_y = std::max(*max_y, request.max_y);
}

void SocialCostLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  const auto peds = currentPedestrians();
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

  for (const auto & ped : peds->pedestrians) {
    const auto & p = ped.pose.position;
    const CellWindow w = cellWindow(
      master_grid, segmentBox(p.x, p.y, p.x, p.y, reach_), min_i, min_j, max_i, max_j);

    const double clip_r2 = personal_space_radius_ * personal_space_radius_;

    for (int j = w.j0; j < w.j1; ++j) {
      for (int i = w.i0; i < w.i1; ++i) {
        double wx, wy;
        master_grid.mapToWorld(i, j, wx, wy);
        const double dx = wx - p.x;
        const double dy = wy - p.y;
        if (dx * dx + dy * dy > clip_r2) {
          continue;
        }
        const double cost = personalSpaceCost(
          dx, dy, personalSpaceSigma(ped.animation_state),
          personalSpaceAmplitude(ped.animation_state), personal_space_params_.shape);
        const unsigned char cell_cost = toCellCost(cost);
        const unsigned char existing = master_grid.getCost(i, j);
        if (existing == nav2_costmap_2d::NO_INFORMATION || cell_cost > existing) {
          master_grid.setCost(i, j, cell_cost);
        }
      }
    }
  }

  for (const auto & group : interactionGroups(*peds)) {
    const double weight = interactionWeight(group.interaction_type);
    if (weight <= 0.0) {
      continue;
    }
    const auto & members = group.members;
    for (size_t a = 0; a < members.size(); ++a) {
      for (size_t b = a + 1; b < members.size(); ++b) {
        const auto & pa = peds->pedestrians[members[a]].pose.position;
        const auto & pb = peds->pedestrians[members[b]].pose.position;
        const CellWindow w = cellWindow(
          master_grid, segmentBox(pa.x, pa.y, pb.x, pb.y, group_extra_radius_),
          min_i, min_j, max_i, max_j);

        const double clip_r2 = group_extra_radius_ * group_extra_radius_;

        for (int j = w.j0; j < w.j1; ++j) {
          for (int i = w.i0; i < w.i1; ++i) {
            double wx, wy;
            master_grid.mapToWorld(i, j, wx, wy);
            if (pointToSegmentDistSq(wx, wy, pa.x, pa.y, pb.x, pb.y) > clip_r2) {
              continue;
            }
            const double cost = groupBetweenCost(
              wx, wy, pa.x, pa.y, pb.x, pb.y, weight, group_sigma_, falloff_shape_);
            const unsigned char cell_cost = toCellCost(cost);
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
