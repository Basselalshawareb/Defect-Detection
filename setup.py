from setuptools import find_packages, setup


if __name__=="__main__":
    setup(
        name="Defect",
        packages=find_packages(include=['wrappers','packages'])
    )