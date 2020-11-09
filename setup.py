from setuptools import setup, find_packages


setup(
    name='attention_benchmarking',
    version='0.0.1',
    url='https://github.com/AndriyMulyar/2020-603-Project-Mulyar',
    description='',
    packages=find_packages(),
    install_requires=[
        'tokenizers',
        'datasets',
        'torch'
    ],
    include_package_data=True,
)
