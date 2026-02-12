from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:

    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        # "-e ." is valid in requirements.txt for pip, but
        # it is not a valid entry for install_requires in setup().
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        # Filter out any empty strings that may appear
        requirements = [req for req in requirements if req]

        return requirements
    
setup(
    name="customer-churn-prediction-model",
    version="0.0.1",
    author="Vivek Boddul",
    author_email="boddulvivek7474@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
