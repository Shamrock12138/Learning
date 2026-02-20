from setuptools import setup, find_packages

setup(
  name="myTools",
  version="0.1.0",
  packages=find_packages(exclude=['Projects*']),
  package_dir={'': '.'},  # 包的根目录就是当前目录
  install_requires=[],  # 添加依赖
  author="shamrock",
  description="我的函数库",
)
