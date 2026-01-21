#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd
from loguru import logger
from ucimlrepo import fetch_ucirepo

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)

datasets = {
    "iris": 53,
    "heart_disease": 45,
    "wine_quality": 186,
    "breast_cancer_wisconsin_diagnostic": 17,
    "bank_marketing": 222,
    "adult": 2,
    "wine": 109,
    "car_evaluation": 19,
    "predict_students_dropout_and_academic_success": 697,
    "default_of_credit_card_clients": 350,
    "automobile": 10,
    "mushroom": 73,
    "statlog_german_credit_data": 144,
    "abalone": 1,
    "bike_sharing": 275,
    "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition": 544,
    "auto_mpg": 9,
    "spambase": 94,
    "online_shoppers_purchasing_intention_dataset": 468,
    "magic_gamma_telescope": 159,
    "breast_cancer_wisconsin_original": 15,
    "forest_fires": 162,
    "breast_cancer": 14,
    "dry_bean": 602,
    "heart_failure_clinical_records": 519,
    "credit_approval": 27,
    "real_estate_valuation": 477,
    "chronic_kidney_disease": 336,
    "concrete_compressive_strength": 165,
    "wholesale_customers": 292,
    "parkinsons": 174,
    "seoul_bike_sharing_demand": 560,
    "rice_cammeo_and_osmancik": 545,
    "optical_recognition_of_handwritten_digits": 80,
    "glass_identification": 42,
    "phishing_websites": 327,
    "appliances_energy_prediction": 374,
    "banknote_authentication": 267,
    "zoo": 111,
    "higher_education_students_performance_evaluation": 856,
    "productivity_prediction_of_garment_employees": 597,
    "maternal_health_risk": 863,
    "early_stage_diabetes_risk_prediction": 529,
    "letter_recognition": 59,
    "combined_cycle_power_plant": 294,
    "liver_disorders": 60,
    "national_poll_on_healthy_aging_npha": 936,
    "hepatitis": 46,
    "aids_clinical_trials_group_study_175": 890,
    "iranian_churn": 563,
}


def get_prefix(role, var_type):
    role = str(role).lower() if role else ""
    var_type = str(var_type).lower() if var_type else ""

    # Mapping logic
    mapping = {
        ("target", "categorical"): "cls_",
        ("target", "binary"): "cls_",
        ("target", "continuous"): "reg_",
        ("target", "integer"): "reg_",
        ("feature", "categorical"): "cat_",
        ("feature", "binary"): "cat_",
        ("feature", "continuous"): "num_",
        ("feature", "integer"): "num_",
        ("feature", "date"): "dat_",
    }

    if (role, var_type) in mapping:
        return mapping[(role, var_type)]

    if role == "other":
        if var_type in ["categorical", "binary"]:
            return "cat_"
        elif var_type in ["continuous", "integer"]:
            return "num_"
        elif var_type == "date":
            return "dat_"

    if role in ["id", "metadata"] or var_type == "id":
        return "id_"

    # Raise exception for truly unknown combinations
    raise ValueError(f"Unknown prefix for Role: {role}, Type: {var_type}")


def download_all(output_dir, datasets=datasets):
    os.makedirs(output_dir, exist_ok=True)

    for index, (name, ds_id) in enumerate(datasets.items(), start=1):
        out_path = os.path.join(output_dir, f"{index:03d}_{ds_id:03d}_{name}.csv")
        if os.path.exists(out_path):
            logger.info(f"Skipping {name}, already exists.")
            continue
        try:
            logger.info(f"Processing {index}/{len(datasets)}: {name}")
            ds = fetch_ucirepo(id=ds_id)

            # Combine components
            X = ds.data.features
            y = ds.data.targets
            df = pd.concat([X, y], axis=1)

            # Build rename map and clean types
            rename_map = {}
            for _, row in ds.variables.iterrows():
                col_name = row["name"]
                prefix = get_prefix(row["role"], row["type"])

                if prefix and col_name in df.columns:
                    rename_map[col_name] = f"{prefix}{col_name}"

            df = df.rename(columns=rename_map)

            # Final output
            df.to_csv(out_path, index=False)
            logger.success(f"Saved {name}")

        except Exception as e:
            logger.error(f"Error on {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all registered UCIML datasets.")
    parser.add_argument(
        "dir", type=str, help="Directory to save CSV files.", default="data", nargs="?"
    )
    args = parser.parse_args()

    download_all(args.dir)
