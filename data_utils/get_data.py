import argparse
import os
import shutil

import kagglehub


def download_dataset(dataset_name: str, output_dir: str):
    print(f"Downloading {dataset_name} ...")
    path = kagglehub.dataset_download(dataset_name)
    final_path = os.path.join(output_dir, os.path.basename(path))

    shutil.move(path, final_path)
    print(f"Moved {dataset_name} to: {final_path}\n")


def cli():
    parser = argparse.ArgumentParser(
        description="Download multiple datasets into a specified folder"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Path to the folder where datasets should be downloaded",
    )
    args = parser.parse_args()
    return args


def main():
    args = cli()
    base_dir = args.output_dir

    os.makedirs(base_dir, exist_ok=True)

    datasets = {
        "shivapan/nih-x-ray-dataset": "nih-x-ray-dataset",
        "ashery/chexpert": "chexpert",
        # "raddar/padchest-chest-xrays-sample": "padchest",
    }

    for dataset_name, subfolder in datasets.items():
        dataset_dir = os.path.join(base_dir, subfolder)
        os.makedirs(dataset_dir, exist_ok=True)
        download_dataset(dataset_name, dataset_dir)


if __name__ == "__main__":
    main()
