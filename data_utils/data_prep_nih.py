# TODO: merge this with the data_prep_chexpert.py to have a unified data prep script

from __future__ import annotations

import argparse
import os
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

PathPair = Tuple[Path, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NIH Chest X-ray dataset")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--workers", type=int, default=max(cpu_count() - 2, 1))
    return parser.parse_args()


def process_image_star(args: Tuple[PathPair, Tuple[int, int]]) -> Optional[str]:
    task, img_size = args
    return process_image(task, img_size)


def process_image(task: PathPair, img_size: Tuple[int, int]) -> Optional[str]:
    src, dst = task
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = img.resize(img_size, Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="JPEG", quality=95)
    except Exception as exc:
        return str(exc)
    return None


def collect_image_tasks(src_root: Path, dst_root: Path) -> List[PathPair]:
    tasks: List[PathPair] = []
    for path in src_root.rglob("*"):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            dst = dst_root / f"{path.stem}.jpg"
            tasks.append((path, dst))
    return tasks


def run_image_pool(
    tasks: Sequence[PathPair],
    img_size: Tuple[int, int],
    workers: int,
    desc: str,
) -> None:
    payload = [(task, img_size) for task in tasks]

    with Pool(workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(process_image_star, payload),
                total=len(payload),
                desc=desc,
            )
        )


def binarize_labels(df: pd.DataFrame) -> List[str]:
    labels = sorted(set("|".join(df["Finding Labels"]).split("|")) - {"No Finding"})
    for label in labels:
        df[label] = df["Finding Labels"].apply(lambda x: int(label in x.split("|")))
    return labels


def assign_folds(df: pd.DataFrame, n_folds: int, group_col: str) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1
    gkf = GroupKFold(n_splits=n_folds)

    for fold, (_, val_idx) in enumerate(gkf.split(df, groups=df[group_col])):
        df.loc[val_idx, "fold"] = fold

    return df


def process_fold(
    df: pd.DataFrame,
    fold: int,
    output_dir: Path,
    img_size: Tuple[int, int],
    workers: int,
) -> None:
    fold_dir = output_dir / f"fold{fold + 1}"
    train_dir = fold_dir / "train"
    val_dir = fold_dir / "val"

    train_df = df[df.fold != fold]
    val_df = df[df.fold == fold]

    def build_tasks(rows: Iterable[pd.Series], dst_root: Path) -> List[PathPair]:
        return [
            (
                Path(row["Path"]),
                dst_root / f"{Path(row['Path']).stem}.jpg",
            )
            for _, row in rows.iterrows()
        ]

    run_image_pool(
        build_tasks(train_df, train_dir),
        img_size,
        workers,
        f"Fold {fold + 1} Train",
    )
    run_image_pool(
        build_tasks(val_df, val_dir),
        img_size,
        workers,
        f"Fold {fold + 1} Val",
    )

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["Path"] = train_df["Image Index"].apply(
        lambda x: str(train_dir / f"{Path(x).stem}.jpg")
    )
    val_df["Path"] = val_df["Image Index"].apply(lambda x: str(val_dir / f"{Path(x).stem}.jpg"))

    fold_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(fold_dir / "train.csv", index=False)
    val_df.to_csv(fold_dir / "val.csv", index=False)


def prepare_flat_images(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    img_size: Tuple[int, int],
    workers: int,
) -> Tuple[Path, Path]:
    all_train = output_dir / "all_train_images"
    all_test = output_dir / "all_test_images"

    all_train.mkdir(parents=True, exist_ok=True)
    all_test.mkdir(parents=True, exist_ok=True)

    run_image_pool(
        collect_image_tasks(train_dir, all_train),
        img_size,
        workers,
        "Flattening train images",
    )
    run_image_pool(
        collect_image_tasks(test_dir, all_test),
        img_size,
        workers,
        "Flattening test images",
    )

    return all_train, all_test


def build_training_dataframe(
    csv_path: Path,
    image_root: Path,
    n_folds: int,
) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    labels = binarize_labels(df)

    df["Path"] = df["Image Index"].apply(lambda x: str(image_root / f"{Path(x).stem}.jpg"))
    df = df[df["Path"].apply(os.path.exists)].reset_index(drop=True)

    df = assign_folds(df, n_folds, "Patient ID")
    return df, labels


def materialize_folds(
    df: pd.DataFrame,
    output_dir: Path,
    n_folds: int,
    img_size: Tuple[int, int],
    workers: int,
) -> None:
    for fold in range(n_folds):
        process_fold(df, fold, output_dir, img_size, workers)


def build_test_set(
    csv_path: Path,
    all_test_images: Path,
    output_dir: Path,
    labels: List[str],
    img_size: Tuple[int, int],
    workers: int,
) -> None:
    test_out = output_dir / "test"
    test_img_dir = test_out / "images"
    test_img_dir.mkdir(parents=True, exist_ok=True)

    run_image_pool(
        [(p, test_img_dir / p.name) for p in all_test_images.iterdir()],
        img_size,
        workers,
        "Copying test images",
    )

    df_test = pd.read_csv(csv_path)
    for label in labels:
        df_test[label] = df_test["Finding Labels"].apply(lambda x: int(label in x.split("|")))

    df_test["Path"] = df_test["Image Index"].apply(
        lambda x: str(test_img_dir / f"{Path(x).stem}.jpg")
    )
    df_test = df_test[df_test["Path"].apply(os.path.exists)]

    test_out.mkdir(parents=True, exist_ok=True)
    df_test.to_csv(test_out / "test.csv", index=False)


def main() -> None:
    args = parse_args()

    all_train, all_test = prepare_flat_images(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        tuple(args.img_size),
        args.workers,
    )

    df, labels = build_training_dataframe(
        args.csv_path,
        all_train,
        args.n_folds,
    )

    materialize_folds(
        df,
        args.output_dir,
        args.n_folds,
        tuple(args.img_size),
        args.workers,
    )

    build_test_set(
        args.csv_path,
        all_test,
        args.output_dir,
        labels,
        tuple(args.img_size),
        args.workers,
    )

    shutil.rmtree(all_train, ignore_errors=True)
    shutil.rmtree(all_test, ignore_errors=True)


if __name__ == "__main__":
    main()
