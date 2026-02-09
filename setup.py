from setuptools import setup,find_packages
from typing import List 

def get_requirements(file_path:str)->List[str]:
    
    with open(file_path,'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        return requirements
    
setup(
    name="customer-churn-prediction-model",
    version="0.0.1",
    author="Vivek Boddul",
    author_email="boddulvivek7474@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
