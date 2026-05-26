from setuptools import find_packages, setup

package_name = "arena_planners"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(where=".", include=[f"{package_name}*"]),
    package_dir={"": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [],
    },
)
