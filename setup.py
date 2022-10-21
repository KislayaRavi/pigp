from distutils.core import setup

setup(
    name = "pigp",
    version = "0.0.1",
    packages = ["pigp"],
    license = 'MIT',
    long_description = open("README.md").read(),
    install_requires = ['gpflow','tensorflow','scipy','numpy','matplotlib'],
)