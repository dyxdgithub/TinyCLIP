"""Download all n2 one-label-per-parent images from the ModelScope source.

This entry point uses the same ModelScope WebDataset implementation as the
sampled downloader, but targets the complete n2 one-label-per-parent manifest.
It writes images, a status CSV, and a resumable SQLite checkpoint separately
from the sampled-300-per-class task.
"""

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
N2_DIR = SCRIPT_DIR.parent / "Hierarchy" / "n2"
IMPLEMENTATION_PATH = SCRIPT_DIR / "download_n2_sampled_images_from_modelscope.py"


def load_downloader_module():
    specification = importlib.util.spec_from_file_location(
        "n2_modelscope_downloader", IMPLEMENTATION_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError("Unable to load downloader implementation: {}".format(IMPLEMENTATION_PATH))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    downloader = load_downloader_module()
    downloader.DEFAULT_INPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv"
    downloader.DEFAULT_CLASS_TREE = N2_DIR / "Last_Two_Layers_Multi_Members_n2.json"
    downloader.DEFAULT_OUTPUT_DIR = N2_DIR / "images_modelscope_one_label_per_parent"
    downloader.DEFAULT_STATUS_OUTPUT = N2_DIR / "Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent_modelscope_download_status.csv"
    downloader.main()


if __name__ == "__main__":
    main()
