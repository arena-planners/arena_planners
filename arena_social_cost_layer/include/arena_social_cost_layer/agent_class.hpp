// Copyright 2026 ductainguyen
#pragma once

#include <string>
#include <unordered_map>

namespace arena_social_cost_layer
{

/// Ground-truth class of a pedestrian from its humansim agent type (`Pedestrian.agent_type`): the
/// type's `interaction_class`, following `extends` the way humansim does. A type is a builtin name,
/// read from `<builtin_dir>/<name>.yaml`, or a yaml path whose `extends` names a builtin; empty is
/// `adult`.
class AgentClassResolver
{
public:
  explicit AgentClassResolver(std::string builtin_dir = "");

  /// The type's interaction_class, "" when it has none or its chain cannot be read. Each type is
  /// read once.
  const std::string & interactionClass(const std::string & agent_type);

  /// Whether interactionClass(agent_type) could not read the type or a type it extends, for
  /// logging.
  bool unresolved(const std::string & agent_type) const;

private:
  bool resolve(const std::string & agent_type, int depth, std::string & cls) const;

  std::string builtin_dir_;
  std::unordered_map<std::string, std::string> cache_;
  std::unordered_map<std::string, bool> failed_;
};

/// Share directory of humansim's builtin agent types, "" when arena_humansim is not installed.
std::string defaultAgentTypesDir();

}  // namespace arena_social_cost_layer
