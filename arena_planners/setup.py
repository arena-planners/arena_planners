from setuptools import find_packages, setup

package_name = "arena_planners"

setup(
    name=package_name,
    packages=find_packages(where=".", include=[f"{package_name}*"]),
    package_dir={"": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "cohan_peds_bridge = arena_planners.cohan_peds_bridge:main",
        ],
    },
)
