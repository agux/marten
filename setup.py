from setuptools import setup, find_packages

setup(
    name="marten",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "marten=marten.cli.main:main",  # 'marten' command will call 'main' function from 'cli.main' module
        ],
    },
)
