from setuptools import setup, find_packages


setup(
    name="swanlab_logger",
    version="0.1.0",
    description="Unified logging adapter supporting both TensorBoard and SwanLab",
    long_description=open("README.md", "r", encoding="utf-8").read() if __import__('os').path.exists("README.md") else "Unified logging adapter supporting both TensorBoard and SwanLab",
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        # SwanLab is an optional dependency, handled in the code
    ],
    extras_require={
        "swanlab": ["swanlab>=0.1.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)