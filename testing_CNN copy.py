import numpy as np
import pandas as pd
from pathlib import Path
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,balanced_accuracy_score,)
from sklearn.model_selection import (train_test_split)
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (Dataset, DataLoader,WeightedRandomSampler,)

# CONFIG

CSV_PATH = ("/Users/arushimukherji/Desktop/AriyaLab/Processed_Data/May_all_particles.csv")

BATCH_SIZE = 32
RANDOM_STATE = 42

NUM_EPOCHS = 15
LR = 3e-4
PATIENCE = 6
MAX_PER_ACQ = 150

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# MATERIAL GROUPS
PURE_OILS = {"Canola", "Diesel", "Gasoline", "Bitumen",}
PURE_NONOILS = {"Water","CNT","Tailing Sands","Malonic Acid in Water","Succinic Acid", "Oxalic Acid"}
MIXTURES = {"Coke + Bitumen","Oil Sands",}

# BITUMINOUS = {
#     "Bitumen",
#     "Coke + Bitumen",
# }
NON_BITUMINOUS_OILS = {"Canola", "Diesel", "Gasoline",}

# FEATURES
PHYS_FEATURES = [

    "equiv_diameter_um",
    "area_um2",

    "feret_max_um",
    "feret_min_um",

    "aspect_ratio",
    "circularity",
    "solidity",
    "convex_deviation",

    "amp_mean",
    "amp_std",

    #"rim_ratio",
    "radial_gradient",

    "phase_mean",
    "integrated_opd_per_area",
    "mean_phase_gradient",
    "edge_phase_gradient_mean",
    "laplacian_variance",
    "center_edge_phase_difference",
    "Ra_optical",
    "Rt_optical",
]
# LOAD
df_master = pd.read_csv(CSV_PATH)
missing = (set(PHYS_FEATURES)- set(df_master.columns))
if missing:

    raise RuntimeError(
        f"Missing features: {missing}"
    )

# TASK DEFINITIONS
def build_model1_dataset(df):

    """
    MODEL 1:
    PURE OIL vs PURE NON-OIL
    """

    keep = PURE_OILS | PURE_NONOILS

    df = (
        df[df["material"].isin(keep)]
        .copy()
    )

    df["label"] = (
        df["material"]
        .isin(PURE_OILS)
        .astype(int)
    )

    return df.reset_index(drop=True)

# ============================================================
def build_model1b_dataset(df):

    """
    MODEL 1B:
    BITUMINOUS vs NON-BITUMINOUS OILS
    """

    BITUMINOUS_OILS = {"Bitumen","Oil Sands", "Coke + Bitumen"}

    NON_BITUMINOUS_OILS = {"Canola","Diesel","Gasoline",}

    keep = (BITUMINOUS_OILS| NON_BITUMINOUS_OILS)

    df = (df[df["material"].isin(keep)].copy())

    df["label"] = (
        df["material"]
        .isin(BITUMINOUS_OILS)
        .astype(int)
    )

    return df.reset_index(drop=True)

# ============================================================
def build_model2_dataset(df):

    """
    MODEL 2:
    OIL-ASSOCIATED PARTICLE DETECTION
    """

    MODEL2_POS = {
        "Bitumen",
        "Gasoline",
        "Diesel",
        "Canola",
        "Coke + Bitumen",
        "Oil Sands",
    }

    MODEL2_NEG = {"Water","CNT", "Malonic Acid in Water", "Succinic Acid", "Oxalic Acid","Tailing Sands",}

    keep = MODEL2_POS | MODEL2_NEG

    df = (df[df["material"].isin(keep)].copy())

    df["label"] = (
        df["material"]
        .isin(MODEL2_POS)
        .astype(int)
    )

    return df.reset_index(drop=True)

# SPLIT

def cap_particles_per_acquisition(
    df,
    max_per_acq=MAX_PER_ACQ,
    random_state=RANDOM_STATE,
):

    capped_parts = []

    grouped = df.groupby(
        "acquisition_id"
    )

    for _, g in grouped:

        if len(g) > max_per_acq:

            g = g.sample(
                n=max_per_acq,
                random_state=random_state,
            )

        capped_parts.append(g)
    df_out = pd.concat(capped_parts).reset_index(drop=True)
        
    return df_out

# HYBRID SPLIT
def hybrid_material_split(df):

    """
    Hybrid split:
    - particle-level split
    - stratified by material
    - acquisitions allowed across splits
    - no duplicate particles
    """

    df = df.copy()

    # TEST SPLIT
    train_df, test_df = train_test_split(

        df,
        test_size=0.20,
        stratify=df["material"],
        random_state=RANDOM_STATE,
    )

    # VAL SPLIT
    train_df, val_df = train_test_split(

        train_df,
        test_size=0.20,
        stratify=train_df["material"],
        random_state=RANDOM_STATE,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

# MATERIAL BALANCED SAMPLER
def build_material_sampler(df):

    material_counts = (
        df["material"]
        .value_counts()
        .to_dict()
    )

    weights = df["material"].map(
        lambda m: 1.0 / material_counts[m]
    ).values

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )

# FEATURE PREP

def prepare_features(
    train_df,
    val_df,
    test_df,
):

    def clean(df):

        df = df.copy()

        df[PHYS_FEATURES] = (

            df[PHYS_FEATURES]

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )

        return df

    train_df = clean(train_df)
    val_df = clean(val_df)
    test_df = clean(test_df)

    scaler = StandardScaler()

    train_df[PHYS_FEATURES] = (
        scaler.fit_transform(
            train_df[PHYS_FEATURES]
        )
    )

    val_df[PHYS_FEATURES] = (
        scaler.transform(
            val_df[PHYS_FEATURES]
        )
    )

    test_df[PHYS_FEATURES] = (
        scaler.transform(
            test_df[PHYS_FEATURES]
        )
    )

    return (
        train_df,
        val_df,
        test_df,
    )

# DATASET

class ParticleDataset(Dataset):

    def __init__(
        self,
        df,
        augment=False,
    ):

        self.df = (
            df.reset_index(drop=True)
        )

        self.augment = augment

    def __len__(self):

        return len(self.df)

    # ========================================================

    def augment_img(self, x):

        if random.random() < 0.5:

            noise = np.random.normal(
                0,
                0.005,
                size=x.shape,
            ).astype(np.float32)

            x = x + noise

        if random.random() < 0.5:

            scale = 0.95 + 0.1 * random.random()

            x = x * scale

        return x

    # ========================================================

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        data = np.load(
            row["npz_path"]
        )

        amp = data["amplitude"].astype(np.float32)
        phase = data["phase"].astype(np.float32)

        # ====================================================
        # AMP NORMALIZED
        # ====================================================

        amp = (
            amp - np.mean(amp)
        ) / (
            np.std(amp) + 1e-8
        )

        # RAW PHASE

        phase = phase.astype(np.float32)

        x_img = np.stack(
            [amp, phase],
            axis=0,
        )

        if self.augment:

            x_img = self.augment_img(x_img)

        x_img = x_img.astype(np.float32)

        x_phys = (

            row[PHYS_FEATURES]
            .values
            .astype(np.float32)
        )

        y = int(row["label"])

        return (

            torch.from_numpy(x_img),

            torch.from_numpy(x_phys),

            torch.tensor(
                y,
                dtype=torch.long,
            ),
        )

# MODEL

class CNNWithPhysics(nn.Module):

    def __init__(self, num_phys):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                2,
                16,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1,
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((8, 8)),
        )

        # IMAGE BRANCH

        self.fc_img = nn.Sequential(

            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
        )

       
        # PHYSICS BRANCH
        self.phys = nn.Sequential(

            nn.Linear(num_phys, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # CLASSIFIER

        self.classifier = nn.Sequential(

            nn.Linear(
                128 + 64,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 2),
        )

    # ========================================================

    def forward(
        self,
        x_img,
        x_phys,
    ):

        x = self.conv(x_img)

        x = x.view(x.size(0), -1)

        img_feat = self.fc_img(x)

        phys_feat = self.phys(x_phys)

        combined = torch.cat(
            [img_feat, phys_feat],
            dim=1,
        )

        return self.classifier(combined)
# EVAL

def evaluate(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    conf = []

    with torch.no_grad():

        for x_img, x_phys, y in loader:

            logits = model(
                x_img.to(DEVICE),
                x_phys.to(DEVICE),
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            confidence, preds = (
                probs.max(dim=1)
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

            y_true.extend(
                y.numpy()
            )

            conf.extend(
                confidence.cpu().numpy()
            )

    return (

        np.array(y_true),

        np.array(y_pred),

        np.array(conf),
    )
import seaborn as sns

# FEATURE IMPORTANCE VISUALIZATION

def visualize_phys_weights(
    model,
    phys_features,
    model_name,
    output_dir="phys_weights",
):
    """Extract and visualize physics MLP weights"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Get first linear layer weights from physics branch
    phys_weights = (
        model.phys[0].weight.data.cpu().numpy()
    )

    # Shape: (64, num_features)
    # Average absolute weight per feature
    feature_importance = np.abs(
        phys_weights
    ).mean(axis=0)

    # Sort by importance (descending)
    sorted_idx = np.argsort(
        feature_importance
    )[::-1]

    sorted_features = [
        phys_features[i]
        for i in sorted_idx
    ]

    sorted_importance = feature_importance[
        sorted_idx
    ]

    # PRINT TO TERMINAL
    print("\n" + "="*60)
    print(f"PHYSICS FEATURE IMPORTANCE - {model_name}")
    print("="*60)

    for feat, imp in zip(sorted_features, sorted_importance):
        print(f"{feat:40s} {imp:.6f}")

    print("="*60 + "\n")
    # SAVE TO CSV

    importance_df = pd.DataFrame({
        "feature": sorted_features,
        "mean_abs_weight": sorted_importance,
        "rank": range(1, len(sorted_features) + 1),
    })

    csv_path = os.path.join(
        output_dir,
        f"{model_name}_phys_weights.csv",
    )

    importance_df.to_csv(csv_path, index=False)

    print(f"✓ Saved to: {csv_path}\n")

    # PLOT BAR CHART

    plt.figure(figsize=(10, 8))

    colors = plt.cm.viridis(
        sorted_importance / sorted_importance.max()
    )

    plt.barh(
        sorted_features,
        sorted_importance,
        color=colors,
    )

    plt.xlabel("Mean Absolute Weight")

    plt.title(
        f"{model_name} - Physics MLP Feature Importance"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plot_path = os.path.join(
        output_dir,
        f"{model_name}_phys_weights.png",
    )

    plt.savefig(plot_path, dpi=300)

    plt.close()

    print(f"✓ Plot saved to: {plot_path}\n")

    return importance_df

# ============================================================
# MODIFY prepare_features FUNCTION
# ============================================================

def prepare_features(
    train_df,
    val_df,
    test_df,
):

    def clean(df):

        df = df.copy()

        df[PHYS_FEATURES] = (

            df[PHYS_FEATURES]

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )

        return df

    train_df = clean(train_df)
    val_df = clean(val_df)
    test_df = clean(test_df)

    # ========================================================
    # FIT SCALER ON TRAIN DATA ONLY
    # ========================================================

    scaler = StandardScaler()

    train_df[PHYS_FEATURES] = (
        scaler.fit_transform(
            train_df[PHYS_FEATURES]
        )
    )

    val_df[PHYS_FEATURES] = (
        scaler.transform(
            val_df[PHYS_FEATURES]
        )
    )

    test_df[PHYS_FEATURES] = (
        scaler.transform(
            test_df[PHYS_FEATURES]
        )
    )

    return (
        train_df,
        val_df,
        test_df,
    )

# ============================================================
# MODIFY train_model FUNCTION
# ============================================================

def train_model(
    df,
    model_name,
    use_weighted_loss=False,
):

    df = cap_particles_per_acquisition(
        df,
        max_per_acq=150,
    )

    print("\n================================================")
    print(model_name)
    print("================================================")

    print(
        df["label"]
        .value_counts()
    )

    # ========================================================
    # SPLIT
    # ========================================================

    train_df, val_df, test_df = (
        hybrid_material_split(df)
    )

    # ========================================================

    print(
        "Train:", len(train_df),
        "Val:", len(val_df),
        "Test:", len(test_df),
    )

    # ========================================================
    # STANDARDIZE FEATURES (NEW)
    # ========================================================

    train_df, val_df, test_df = (
        prepare_features(
            train_df,
            val_df,
            test_df,
        )
    )

    # ========================================================

    train_ds = ParticleDataset(
        train_df,
        augment=True,
    )

    val_ds = ParticleDataset(val_df)

    test_ds = ParticleDataset(test_df)

    # ========================================================

    sampler = build_material_sampler(train_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    val_loader = DataLoader(
        val_ds,
        BATCH_SIZE,
    )

    test_loader = DataLoader(
        test_ds,
        BATCH_SIZE,
    )

    # ========================================================

    model = CNNWithPhysics(
        len(PHYS_FEATURES)
    ).to(DEVICE)

    # ========================================================
    # OPTIONAL WEIGHTED CROSS ENTROPY
    # ========================================================

    if use_weighted_loss:

        class_counts = (
            train_df["label"]
            .value_counts()
            .sort_index()
            .values
        )

        # inverse-frequency weighting
        class_weights = 1.0 / class_counts

        # normalize
        class_weights = (
            class_weights
            / class_weights.sum()
        )

        class_weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
        ).to(DEVICE)

        print("\nUsing weighted CE:")
        print(class_weights)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )

    else:

        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_val = np.inf
    patience_counter = 0

    # ========================================================
    # TRAIN LOOP
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        model.train()

        train_loss = 0

        for x_img, x_phys, y in train_loader:

            x_img = x_img.to(DEVICE)

            x_phys = x_phys.to(DEVICE)

            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(
                x_img,
                x_phys,
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * x_img.size(0)
            )

        train_loss /= len(train_ds)

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for x_img, x_phys, y in val_loader:

                logits = model(
                    x_img.to(DEVICE),
                    x_phys.to(DEVICE),
                )

                loss = criterion(
                    logits,
                    y.to(DEVICE),
                )

                val_loss += (
                    loss.item()
                    * x_img.size(0)
                )

        val_loss /= len(val_ds)

        print(

            f"Epoch {epoch+1}: "

            f"train={train_loss:.4f}, "

            f"val={val_loss:.4f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            patience_counter = 0

            best_model = model.state_dict()

        else:

            patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):

                print("Early stopping")

                break

    # ========================================================
    # TEST
    # ========================================================

    model.load_state_dict(best_model)

    y_true, y_pred, conf = evaluate(
        model,
        test_loader,
    )

    print("\n===== TEST RESULTS =====")

    print(
        classification_report(
            y_true,
            y_pred,
        )
    )

    print(
        "Balanced Acc:",
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    # ========================================================
    # VISUALIZE PHYSICS WEIGHTS (NEW)
    # ========================================================

    visualize_phys_weights(
        model,
        PHYS_FEATURES,
        model_name,
        output_dir="phys_weights",
    )

    # ========================================================
    # EVAL DF
    # ========================================================

    df_eval = test_df.copy()

    df_eval["true"] = y_true
    df_eval["pred"] = y_pred
    df_eval["confidence"] = conf

    # ========================================================
    # MATERIAL ACCURACY
    # ========================================================

    print("\n===== MATERIAL ACCURACY =====")

    material_acc = (

        df_eval.groupby("material")

        .apply(

            lambda x:
            (
                x["true"]
                == x["pred"]
            ).mean(),

            include_groups=False,
        )

        .sort_values(
            ascending=False
        )
    )

    print(material_acc)

    # ========================================================
    # CONFUSION
    # ========================================================

    print("\n===== CONFUSION BY MATERIAL =====")

    cm = (

        df_eval.groupby(
            ["material", "true", "pred"]
        )

        .size()

        .unstack(fill_value=0)
    )

    print(cm)

    # ========================================================
    # TOP ERRORS
    # ========================================================

    print("\n===== TOP CONFIDENT ERRORS =====")

    errors = df_eval[
        df_eval["true"]
        !=
        df_eval["pred"]
    ]

    print(

        errors.sort_values(
            "confidence",
            ascending=False,
        )

        .head(10)[
            [
                "material",
                "confidence",
                "true",
                "pred",
            ]
        ]
    )

    bac = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    return bac
# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    SEEDS = [1,2,3,4,5]

    model1_bacs = []
    model1b_bacs = []
    model2_bacs = []

    for seed in SEEDS:

        print("\n================================================")
        print(f"RUNNING SEED {seed}")
        print("================================================")

        RANDOM_STATE = seed

        try:

            # ====================================================
            # MODEL 1
            # ====================================================

            df1 = build_model1_dataset(df_master)

            bac1 = train_model(
                df1,
                "MODEL 1 — PURE OIL vs PURE NON-OIL",
            )

            model1_bacs.append(bac1)

            # ====================================================
            # MODEL 1B
            # ====================================================

            df1b = build_model1b_dataset(df_master)

            bac1b = train_model(
                df1b,
                "MODEL 1B — BITUMINOUS vs NON-BITUMINOUS OIL",
                use_weighted_loss=True,
            )

            model1b_bacs.append(bac1b)

            # ====================================================
            # MODEL 2
            # ====================================================

            df2 = build_model2_dataset(df_master)

            bac2 = train_model(
                df2,
                "MODEL 2 — OIL PRESENCE DETECTION",
            )

            model2_bacs.append(bac2)

        except Exception as e:

            print(f"\n ERROR in seed {seed}:")
            print(f"{type(e).__name__}: {e}")
            print("Skipping to next seed...\n")
            continue

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n================================================")
    print("FINAL RESULTS")
    print("================================================")

    print(
        f"MODEL 1 BAC: "
        f"{np.mean(model1_bacs):.3f} "
        f"+/- "
        f"{np.std(model1_bacs):.3f}"
    )

    print(
        f"MODEL 1B BAC: "
        f"{np.mean(model1b_bacs):.3f} "
        f"+/- "
        f"{np.std(model1b_bacs):.3f}"
    )

    print(
        f"MODEL 2 BAC: "
        f"{np.mean(model2_bacs):.3f} "
        f"+/- "
        f"{np.std(model2_bacs):.3f}"
    )