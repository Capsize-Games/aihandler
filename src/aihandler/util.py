import os
import importlib
import requests
from aihandler.qtvar import ExtensionVar


def get_extensions_from_url(app):
    """
    extension CSV format:
        name,description,repo,version,reviewed,official
        LoRA,Adds support for LoRA,Capsize-Games/airunner-lora,1.0.0,true,true
    """
    url = "https://gist.githubusercontent.com/w4ffl35/4379990be80b1140ad8de348a3e2c37a/raw/e67c4cb6744f76361385463987d2cfcfef74ddce/extensions.txt"
    available_extensions = []
    try:
        response = requests.get(url)
        if response.status_code == 200:
            extensions = response.text.splitlines()
            headers = extensions.pop(0)
            headers = headers.split(",")
            for extension in extensions:
                extension = extension.split(",")
                available_extensions.append(ExtensionVar(
                    app,
                    name=extension[headers.index("name")],
                    description=extension[headers.index("description")],
                    repo=extension[headers.index("repo")],
                    version=extension[headers.index("version")],
                    reviewed=extension[headers.index("reviewed")] == "true",
                    official=extension[headers.index("official")] == "true",
                    enabled=True
                ))
    except requests.exceptions.RequestException as e:
        print("Unable to load extensions")
    return available_extensions


def download_extension(repo, extension_path):
    """
    Downloads the extension from the repo and installs it into the extensions folder.
    :param repo:
    :param extension_path:
    :return:
    """
    try:
        # download the extension
        github_url = f"https://github.com/{repo}"
        # download the latest release zip and extract it into the extensions folder
        # get the latest release
        response = requests.get(f"{github_url}/releases/latest")
        if response.status_code == 200:
            # get the latest release zip
            latest_release_url = response.url
            response = requests.get(f"{latest_release_url}.zip")
            if response.status_code == 200:
                # extract the zip into the extensions folder
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(extension_path)
    except requests.exceptions.RequestException as e:
        print("Unable to download extension")


def import_extension_class(extension_repo, extension_path, file_name, class_name):
    if not os.path.exists(extension_path):
        download_extension(extension_repo, extension_path)
    for f in os.listdir(extension_path):
        if os.path.isfile(os.path.join(extension_path, f)) and f == file_name:
            module_name = file_name[:-3]
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(extension_path, file_name))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, class_name)

