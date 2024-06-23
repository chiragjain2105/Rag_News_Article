from setuptools import find_packages,setup
from typing import List

setup(
    name="Rag_News_Article",
    version="0.0.1",
    author="Chirag Jain",
    author_email="jainchiragcj2105@gmail.com",
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)