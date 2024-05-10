import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.3"  # Update version number if needed

REPO_NAME = "BTC_PRICE_PREDICTION_MODEL"  # Update repository name
AUTHOR_USER_NAME = "amenallahbenothmen"
SRC_REPO = "LSTM_BTC_Prediction"  # Update package name
AUTHOR_EMAIL = "amenallah.benothmen@supcom.tn"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for LSTM app",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Update content type if needed
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    # Add any additional options or dependencies as needed
)
