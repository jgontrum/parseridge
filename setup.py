from setuptools import setup

setup(
    name="parseridge",
    version="0.0.1",
    author="Johannes Gontrum",
    author_email="j@gontrum.me",
    include_package_data=True,
    license="Apache Licence 2.0",
    entry_points={"console_scripts": ["parseridge = parseridge.main:start"]},
)
