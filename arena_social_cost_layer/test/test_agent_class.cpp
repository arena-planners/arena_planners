// Copyright 2026 ductainguyen
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "arena_social_cost_layer/agent_class.hpp"

using arena_social_cost_layer::AgentClassResolver;

namespace fs = std::filesystem;

class AgentClassTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    dir_ = fs::temp_directory_path() / ("agent_class_" + std::to_string(::getpid()));
    fs::create_directories(dir_ / "builtin");
    write("builtin/adult.yaml", "name: adult\n");
    write("builtin/wheelchair_manual.yaml",
      "name: wheelchair_manual\nextends: adult\ninteraction_class: wheelchair\n");
    write("builtin/caregiver.yaml", "name: caregiver\nextends: adult\n");
  }

  void TearDown() override {fs::remove_all(dir_);}

  std::string write(const std::string & rel, const std::string & text)
  {
    std::ofstream(dir_ / rel) << text;
    return (dir_ / rel).string();
  }

  std::string builtin() const {return (dir_ / "builtin").string();}

  fs::path dir_;
};

TEST_F(AgentClassTest, BuiltinNamesReadTheirClass)
{
  AgentClassResolver r{builtin()};
  EXPECT_EQ(r.interactionClass("wheelchair_manual"), "wheelchair");
  EXPECT_EQ(r.interactionClass("caregiver"), "");
  EXPECT_FALSE(r.unresolved("caregiver"));
  EXPECT_EQ(r.interactionClass(""), "") << "empty is adult";
  EXPECT_FALSE(r.unresolved(""));
}

TEST_F(AgentClassTest, AScenarioTypeInheritsTheClassThroughExtends)
{
  AgentClassResolver r{builtin()};
  const auto rider = write("rider_a.yaml",
    "name: rider_a\nextends: wheelchair_manual\nmode: behavior_tree\n");
  const auto nurse = write("nurse_a.yaml", "name: nurse_a\nextends: caregiver\n");
  EXPECT_EQ(r.interactionClass(rider), "wheelchair");
  EXPECT_EQ(r.interactionClass(nurse), "");
  EXPECT_FALSE(r.unresolved(nurse));
}

TEST_F(AgentClassTest, TheTypesOwnClassWinsOverItsParents)
{
  AgentClassResolver r{builtin()};
  const auto walker = write("walker.yaml",
    "name: walker\nextends: wheelchair_manual\ninteraction_class: pedestrian\n");
  EXPECT_EQ(r.interactionClass(walker), "pedestrian");
}

TEST_F(AgentClassTest, UnreadableTypesAndCyclesAreUnresolved)
{
  AgentClassResolver r{builtin()};
  EXPECT_EQ(r.interactionClass("no_such_type"), "");
  EXPECT_TRUE(r.unresolved("no_such_type"));
  const auto missing = (dir_ / "missing.yaml").string();
  EXPECT_EQ(r.interactionClass(missing), "");
  EXPECT_TRUE(r.unresolved(missing));
  write("builtin/loop_a.yaml", "name: loop_a\nextends: loop_b\n");
  write("builtin/loop_b.yaml", "name: loop_b\nextends: loop_a\n");
  EXPECT_EQ(r.interactionClass("loop_a"), "");
  EXPECT_TRUE(r.unresolved("loop_a"));
}

TEST_F(AgentClassTest, NoBuiltinDirResolvesOnlyPathsWithTheirOwnClass)
{
  AgentClassResolver r{""};
  const auto own = write("own.yaml", "name: own\ninteraction_class: wheelchair\n");
  EXPECT_EQ(r.interactionClass(own), "wheelchair");
  EXPECT_EQ(r.interactionClass("wheelchair_manual"), "");
  EXPECT_TRUE(r.unresolved("wheelchair_manual"));
}
