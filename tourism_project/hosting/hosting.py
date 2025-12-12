'''
upload following 3 files to hugging face spaces repository:
 1. requirements.txt: dependencies file
 2. app.py: front-end app script
 3. dockerfile: docker container actions file
'''
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# hugging face login profile id
hf_login_id = "JaiBhatia020373"

# set name of the new repository on the Hugging face hub
repo_name = "tourism-package-prediction"

# repository type - data repository
repo_type = "spaces"

repo_id = hf_login_id + "/" + repo_name

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists and if not then create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# upload specified data folder from google colab to hugging face repository
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type=repo_type)
