// Copyright 2026 ductainguyen
#include "arena_social_cost_layer/agent_class.hpp"

#include <string>
#include <utility>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "yaml-cpp/yaml.h"

namespace arena_social_cost_layer
{

namespace
{

constexpr int kMaxExtends = 16;

bool isPath(const std::string & agent_type)
{
  const std::string ext = ".yaml";
  return agent_type.find('/') != std::string::npos ||
         (agent_type.size() >= ext.size() &&
         agent_type.compare(agent_type.size() - ext.size(), ext.size(), ext) == 0);
}

}  // namespace

AgentClassResolver::AgentClassResolver(std::string builtin_dir)
: builtin_dir_(std::move(builtin_dir)) {}

const std::string & AgentClassResolver::interactionClass(const std::string & agent_type)
{
  const auto it = cache_.find(agent_type);
  if (it != cache_.end()) {
    return it->second;
  }
  std::string cls;
  failed_[agent_type] = !resolve(agent_type.empty() ? "adult" : agent_type, 0, cls);
  return cache_.emplace(agent_type, std::move(cls)).first->second;
}

bool AgentClassResolver::unresolved(const std::string & agent_type) const
{
  const auto it = failed_.find(agent_type);
  return it != failed_.end() && it->second;
}

bool AgentClassResolver::resolve(const std::string & agent_type, int depth, std::string & cls) const
{
  if (depth > kMaxExtends) {
    return false;
  }
  std::string path = agent_type;
  if (!isPath(agent_type)) {
    if (builtin_dir_.empty()) {
      return false;
    }
    path = builtin_dir_ + "/" + agent_type + ".yaml";
  }
  try {
    const YAML::Node node = YAML::LoadFile(path);
    if (node["interaction_class"] && node["interaction_class"].IsScalar()) {
      cls = node["interaction_class"].as<std::string>();
      return true;
    }
    if (node["extends"] && node["extends"].IsScalar()) {
      return resolve(node["extends"].as<std::string>(), depth + 1, cls);
    }
  } catch (const YAML::Exception &) {
    return false;
  }
  cls.clear();  // the end of the chain without one: humansim's default, no class
  return true;
}

std::string defaultAgentTypesDir()
{
  try {
    return ament_index_cpp::get_package_share_directory("arena_humansim") + "/config/agent_types";
  } catch (const std::exception &) {
    return "";
  }
}

}  // namespace arena_social_cost_layer
