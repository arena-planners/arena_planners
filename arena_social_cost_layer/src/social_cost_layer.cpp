// Copyright 2026 ductainguyen
#include "arena_social_cost_layer/social_cost_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
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
  int32_t interaction_id;
  uint8_t interaction_type;
  std::vector<size_t> members;
};

double yawOf(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

double speedOf(const arena_people_msgs::msg::Pedestrian & ped)
{
  return std::hypot(ped.twist.linear.x, ped.twist.linear.y);
}

/// Box around a body's span grown by reach.
Box bodyBox(const AnisotropicBody & body, double reach)
{
  const double bx = body.fx - body.length * std::cos(body.yaw);
  const double by = body.fy - body.length * std::sin(body.yaw);
  return segmentBox(body.fx, body.fy, bx, by, reach);
}

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
      groups.push_back({ped.interaction_id, ped.interaction_type, {idx}});
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
  declareParameter("anisotropic_classes", rclcpp::ParameterValue(anisotropic_classes_));
  declareParameter("agent_types_dir", rclcpp::ParameterValue(std::string{}));
  declareParameter("push_interaction_type", rclcpp::ParameterValue(push_interaction_type_));
  declareParameter("wheelchair_sigma_front",
      rclcpp::ParameterValue(anisotropic_params_.sigma_front));
  declareParameter("wheelchair_sigma_back", rclcpp::ParameterValue(anisotropic_params_.sigma_back));
  declareParameter("wheelchair_sigma_side", rclcpp::ParameterValue(anisotropic_params_.sigma_side));
  declareParameter(
    "wheelchair_front_speed_gain", rclcpp::ParameterValue(anisotropic_params_.front_speed_gain));
  declareParameter("wheelchair_amplitude", rclcpp::ParameterValue(anisotropic_params_.amplitude));
  declareParameter("wheelchair_cutoff", rclcpp::ParameterValue(anisotropic_params_.cutoff));

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
  node->get_parameter(name_ + ".anisotropic_classes", anisotropic_classes_);
  std::string agent_types_dir;
  node->get_parameter(name_ + ".agent_types_dir", agent_types_dir);
  if (agent_types_dir.empty()) {
    agent_types_dir = defaultAgentTypesDir();
  }
  classes_ = AgentClassResolver{agent_types_dir};
  node->get_parameter(name_ + ".push_interaction_type", push_interaction_type_);
  node->get_parameter(name_ + ".wheelchair_sigma_front", anisotropic_params_.sigma_front);
  node->get_parameter(name_ + ".wheelchair_sigma_back", anisotropic_params_.sigma_back);
  node->get_parameter(name_ + ".wheelchair_sigma_side", anisotropic_params_.sigma_side);
  node->get_parameter(
    name_ + ".wheelchair_front_speed_gain", anisotropic_params_.front_speed_gain);
  node->get_parameter(name_ + ".wheelchair_amplitude", anisotropic_params_.amplitude);
  node->get_parameter(name_ + ".wheelchair_cutoff", anisotropic_params_.cutoff);
  anisotropic_params_.shape = falloff_shape_;

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
    "(resolved: '%s'), agent types from '%s'", name_.c_str(), enabled_ ? "true" : "false",
    peds_topic_.c_str(), peds_sub_->get_topic_name(), agent_types_dir.c_str());

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

bool SocialCostLayer::isAnisotropic(const arena_people_msgs::msg::Pedestrian & ped)
{
  const std::string & cls = classes_.interactionClass(ped.agent_type);
  if (classes_.unresolved(ped.agent_type) && warned_types_.insert(ped.agent_type).second) {
    if (auto node = node_.lock()) {
      RCLCPP_WARN(
        node->get_logger(),
          "SocialCostLayer '%s': cannot read agent type '%s' (or a type it extends), "
        "treating it as a plain pedestrian", name_.c_str(), ped.agent_type.c_str());
    }
  }
  return !cls.empty() &&
         std::find(anisotropic_classes_.begin(), anisotropic_classes_.end(), cls) !=
         anisotropic_classes_.end();
}

std::vector<AnisotropicBody> SocialCostLayer::anisotropicBodies(
  const arena_people_msgs::msg::Pedestrians & peds, std::vector<bool> & covered,
  std::vector<int32_t> & pushes)
{
  const auto & list = peds.pedestrians;
  std::vector<bool> chair(list.size(), false);
  for (size_t idx = 0; idx < list.size(); ++idx) {
    chair[idx] = isAnisotropic(list[idx]);
  }
  covered.assign(list.size(), false);
  pushes.clear();
  std::vector<AnisotropicBody> bodies;

  // a pushed pair: one body from the pusher's back to the chair's front, along the chair's heading
  for (const auto & group : interactionGroups(peds)) {
    if (static_cast<int>(group.interaction_type) != push_interaction_type_ ||
      group.members.size() != 2 || chair[group.members[0]] == chair[group.members[1]])
    {
      continue;
    }
    const size_t rider = chair[group.members[0]] ? group.members[0] : group.members[1];
    const size_t pusher = rider == group.members[0] ? group.members[1] : group.members[0];
    const auto & r = list[rider].pose;
    const auto & p = list[pusher].pose.position;
    const double yaw = yawOf(r.orientation);
    const double behind = -((p.x - r.position.x) * std::cos(yaw) + (p.y - r.position.y) *
      std::sin(yaw));
    bodies.push_back({r.position.x, r.position.y, yaw, std::max(behind, 0.0),
        speedOf(list[rider])});
    covered[rider] = true;
    covered[pusher] = true;
    pushes.push_back(group.interaction_id);
  }
  for (size_t idx = 0; idx < list.size(); ++idx) {
    if (chair[idx] && !covered[idx]) {
      const auto & pose = list[idx].pose;
      bodies.push_back({pose.position.x, pose.position.y, yawOf(pose.orientation), 0.0,
          speedOf(list[idx])});
      covered[idx] = true;
    }
  }
  return bodies;
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
    std::vector<bool> covered;
    std::vector<int32_t> pushes;
    for (const auto & body : anisotropicBodies(*peds, covered, pushes)) {
      box = unionBox(box, bodyBox(body, anisotropicReach(anisotropic_params_, body.speed)));
    }
    for (size_t idx = 0; idx < peds->pedestrians.size(); ++idx) {
      if (covered[idx]) {
        continue;
      }
      const auto & p = peds->pedestrians[idx].pose.position;
      box = unionBox(box, segmentBox(p.x, p.y, p.x, p.y, reach_));
    }
    for (const auto & group : interactionGroups(*peds)) {
      if (interactionWeight(group.interaction_type) <= 0.0 ||
        std::find(pushes.begin(), pushes.end(), group.interaction_id) != pushes.end())
      {
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

  const auto raise = [&](int i, int j, double cost) {
      const unsigned char cell_cost = toCellCost(cost);
      const unsigned char existing = master_grid.getCost(i, j);
      if (existing == nav2_costmap_2d::NO_INFORMATION || cell_cost > existing) {
        master_grid.setCost(i, j, cell_cost);
      }
    };

  std::vector<bool> covered;
  std::vector<int32_t> pushes;
  for (const auto & body : anisotropicBodies(*peds, covered, pushes)) {
    const CellWindow w = cellWindow(
      master_grid, bodyBox(body, anisotropicReach(anisotropic_params_, body.speed)),
      min_i, min_j, max_i, max_j);
    for (int j = w.j0; j < w.j1; ++j) {
      for (int i = w.i0; i < w.i1; ++i) {
        double wx, wy;
        master_grid.mapToWorld(i, j, wx, wy);
        const double cost = anisotropicCost(
          bodyOffset(wx, wy, body.fx, body.fy, body.yaw, body.length), anisotropic_params_,
          body.speed);
        if (cost > 0.0) {
          raise(i, j, cost);
        }
      }
    }
  }

  for (size_t idx = 0; idx < peds->pedestrians.size(); ++idx) {
    if (covered[idx]) {
      continue;
    }
    const auto & ped = peds->pedestrians[idx];
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
        raise(
          i, j, personalSpaceCost(
            dx, dy, personalSpaceSigma(ped.animation_state),
            personalSpaceAmplitude(ped.animation_state), personal_space_params_.shape));
      }
    }
  }

  for (const auto & group : interactionGroups(*peds)) {
    const double weight = interactionWeight(group.interaction_type);
    if (weight <= 0.0 ||
      std::find(pushes.begin(), pushes.end(), group.interaction_id) != pushes.end())
    {
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
            raise(i, j,
                groupBetweenCost(wx, wy, pa.x, pa.y, pb.x, pb.y, weight, group_sigma_,
                falloff_shape_));
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
