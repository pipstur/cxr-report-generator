import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold
from tqdm import tqdm


class DatasetPreprocessor:
    """
    A class that handles preprocessing of the CheXpert dataset:
    - Path fixing
    - Filtering frontal images & study1
    - Extracting patient ID
    - Cleaning labels
    - GroupKFold splitting
    - Parallel image copying & resizing
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        train_input_dir: str,
        val_input_dir: str,
        output_dir: str,
        n_folds: int = 5,
        img_size: Tuple[int, int] = (224, 224),
    ):
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.train_input_dir = train_input_dir
        self.val_input_dir = val_input_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.img_size = img_size

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and concatenate train/val CSV files."""
        df_train = pd.read_csv(self.train_csv)
        df_val = pd.read_csv(self.val_csv)
        df = pd.concat([df_train, df_val], ignore_index=True)
        df = df[df["Path"].str.contains("frontal")].reset_index(drop=True)
        return df

    def fix_path(self, p: str) -> str:
        """Convert CSV paths to actual filesystem paths."""
        if "train/" in p:
            return os.path.join(self.train_input_dir, p.split("train/")[1])
        if "valid/" in p or "val/" in p:
            return os.path.join(self.val_input_dir, p.split("valid/")[1])
        return p

    def fix_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply path fixing to the whole dataframe."""
        df["Path"] = df["Path"].apply(self.fix_path)
        return df

    @staticmethod
    def filter_study1(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only samples containing 'study1' in the path."""
        return df[df["Path"].str.contains("study1")].reset_index(drop=True)

    @staticmethod
    def extract_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract patient IDs from paths and drop duplicates.
        Keeps only first image per patient.
        """
        df["patient_id"] = df["Path"].str.extract(r"(patient\d+)")
        df = df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
        return df

    @staticmethod
    def clean_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Replace -1 and NaN with 0, cast label columns to int8."""
        non_label_cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", "patient_id"]
        label_cols = [c for c in df.columns if c not in non_label_cols]

        df[label_cols] = df[label_cols].replace(-1, 0).fillna(0).astype("int8")
        return df, label_cols

    def apply_group_kfold(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply patient-based GroupKFold."""
        gkf = GroupKFold(n_splits=self.n_folds)
        df["fold"] = -1

        for fold_idx, (_, val_idx) in enumerate(gkf.split(df, groups=df["patient_id"])):
            df.loc[val_idx, "fold"] = fold_idx

        return df

    def create_folders(self) -> None:
        """Create train/val directories for each fold."""
        for fold in range(self.n_folds):
            base = os.path.join(self.output_dir, f"fold{fold + 1}")
            os.makedirs(os.path.join(base, "train"), exist_ok=True)
            os.makedirs(os.path.join(base, "val"), exist_ok=True)

    def copy_and_resize(self, row: pd.Series, target_dir: str) -> Tuple[int, Optional[str]]:
        """Copy & resize an image to target directory."""
        src = row["Path"]
        new_name = f"{row['patient_id']}_frontal.jpg"
        dst = os.path.join(target_dir, new_name)

        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                img.save(dst, format="JPEG", quality=95)
        except Exception as e:
            print(f"Error processing {src}: {e}")
            return row.name, None

        return row.name, dst

    def process_fold(
        self, df_train_fold: pd.DataFrame, df_val_fold: pd.DataFrame, fold: int
    ) -> None:
        """Process one fold: copy images + save CSVs."""
        fold_dir = os.path.join(self.output_dir, f"fold{fold + 1}")
        train_dir = os.path.join(fold_dir, "train")
        val_dir = os.path.join(fold_dir, "val")

        # --- train ---
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(self.copy_and_resize, row, train_dir)
                for _, row in df_train_fold.iterrows()
            ]
            for f in tqdm(
                as_completed(futures), total=len(futures), desc=f"Fold {fold + 1} Train"
            ):
                idx, new_path = f.result()
                if new_path:
                    df_train_fold.at[idx, "Path"] = new_path

        # --- val ---
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(self.copy_and_resize, row, val_dir)
                for _, row in df_val_fold.iterrows()
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Fold {fold + 1} Val"):
                idx, new_path = f.result()
                if new_path:
                    df_val_fold.at[idx, "Path"] = new_path

        df_train_fold.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        df_val_fold.to_csv(os.path.join(fold_dir, "val.csv"), index=False)

    def run(self) -> None:
        """Full execution pipeline."""
        df = self.load_data()
        df = self.fix_paths(df)
        df = DatasetPreprocessor.filter_study1(df)
        df = DatasetPreprocessor.extract_patient_ids(df)
        df, label_cols = DatasetPreprocessor.clean_labels(df)
        print("Detected label columns:", label_cols)
        df = self.apply_group_kfold(df)
        self.create_folders()

        print("Processing folds...")
        for fold in range(self.n_folds):
            df_train_fold = df[df["fold"] != fold].copy()
            df_val_fold = df[df["fold"] == fold].copy()
            self.process_fold(df_train_fold, df_val_fold, fold)

        print("\nDONE!")
        print(f"Output saved in: {self.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CheXpert 5-fold preprocessing")

    parser.add_argument(
        "--train-csv", type=str, required=True, help="Path to the training CSV file"
    )
    parser.add_argument(
        "--val-csv", type=str, required=True, help="Path to the validation CSV file"
    )
    parser.add_argument(
        "--train-dir", type=str, required=True, help="Path to the training images directory"
    )
    parser.add_argument(
        "--val-dir", type=str, required=True, help="Path to the validation images directory"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of folds for GroupKFold (default: 5)"
    )
    parser.add_argument(
        "--img-size",
        nargs=2,
        metavar=("H", "W"),
        type=int,
        default=[224, 224],
        help="Image size after resizing (default: 224 224)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processor = DatasetPreprocessor(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        train_input_dir=args.train_dir,
        val_input_dir=args.val_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        img_size=tuple(args.img_size),
    )
    processor.run()


if __name__ == "__main__":
    main()
