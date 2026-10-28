
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="PITCHIQ | Football Analytics",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ PITCHIQ")
st.subheader("Football Player Position Prediction")
st.caption("Data cleaning • Machine learning • Player analytics")

DATA_FILE = Path(__file__).parent / "football_player_position_synthetic.csv"

FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "matches_played",
    "goals",
    "assists",
    "passes_completed",
    "pass_accuracy_percent",
    "shots_per_match",
    "tackles_per_match",
    "saves_per_match",
]

VALID_POSITIONS = [
    "Defender",
    "Forward",
    "Goalkeeper",
    "Midfielder",
]


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


@st.cache_data
def clean_data(raw_data):
    data = raw_data.copy()

    data = data.drop_duplicates()

    numeric_cols = data.select_dtypes(
        include="number"
    ).columns

    categorical_cols = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    for col in categorical_cols:
        if data[col].isnull().any():
            modes = data[col].mode()
            if not modes.empty:
                data[col] = data[col].fillna(modes.iloc[0])

    if "class" in data.columns:
        data["class"] = data["class"].astype("string").str.strip()
        data = data[data["class"].isin(VALID_POSITIONS)].copy()

    return data


# ---------------- TRAIN MODEL ----------------
@st.cache_resource
def train_model(data):
    X = data[FEATURES].copy()
    y = data["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = GaussianNB()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, X_test, y_test, predictions, accuracy


# ---------------- CHECK DATASET ----------------
if not DATA_FILE.exists():
    st.error("Dataset file not found.")
    st.info(
        "Save your dataset generator as generate_dataset.py "
        "and run: python generate_dataset.py"
    )
    st.stop()

try:
    raw_data = load_data(DATA_FILE)
except Exception as error:
    st.error(f"Unable to read dataset: {error}")
    st.stop()

missing_columns = [
    col for col in FEATURES + ["class"]
    if col not in raw_data.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

data = clean_data(raw_data)

if data.empty:
    st.error("No valid player records remain after data cleaning.")
    st.stop()

if data["class"].nunique() < 2:
    st.error("At least two player positions are required to train the model.")
    st.stop()

try:
    model, X_test, y_test, predictions, accuracy = train_model(data)
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("🏆 PITCHIQ MENU")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Dataset Explorer",
        "Data Cleaning",
        "Model Evaluation",
        "Predict Position",
    ]
)

st.sidebar.divider()
st.sidebar.caption("Model: Gaussian Naive Bayes")
st.sidebar.caption("Test split: 20%")


# ---------------- DASHBOARD ----------------
if page == "Dashboard":
    st.header("Football Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Players", f"{len(data):,}")
    col2.metric("Player Positions", data["class"].nunique())
    col3.metric("Model Accuracy", f"{accuracy:.2%}")
    col4.metric(
        "Missing Values After Cleaning",
        f"{int(data.isnull().sum().sum()):,}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Players by Position")
        position_counts = data["class"].value_counts()

        fig, ax = plt.subplots()
        position_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Player Position")
        ax.set_ylabel("Number of Players")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Average Goals by Position")
        goals_by_position = data.groupby("class")["goals"].mean()

        fig, ax = plt.subplots()
        goals_by_position.plot(kind="bar", ax=ax)
        ax.set_ylabel("Average Goals")
        ax.set_xlabel("Player Position")
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)


# ---------------- DATASET EXPLORER ----------------
elif page == "Dataset Explorer":
    st.header("📊 Dataset Explorer")

    st.write("Explore the football player dataset.")

    st.subheader("Dataset Preview")
    st.dataframe(data, use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Duplicate Rows", int(data.duplicated().sum()))

    st.subheader("Column Data Types")
    info_df = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing Values": data.isnull().sum().values,
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(data.describe(), use_container_width=True)

    st.download_button(
        "⬇️ Download Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="football_player_dataset.csv",
        mime="text/csv"
    )


# ---------------- DATA CLEANING ----------------
elif page == "Data Cleaning":
    st.header("🧹 Data Cleaning Process")

    st.write("Compare the original dataset with the cleaned dataset.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Before Cleaning")
        st.metric("Rows", len(raw_data))
        st.metric(
            "Missing Values",
            int(raw_data.isnull().sum().sum())
        )
        st.metric(
            "Duplicate Rows",
            int(raw_data.duplicated().sum())
        )

    with col2:
        st.subheader("After Cleaning")
        st.metric("Rows", len(data))
        st.metric(
            "Missing Values",
            int(data.isnull().sum().sum())
        )
        st.metric(
            "Duplicate Rows",
            int(data.duplicated().sum())
        )

    st.subheader("Missing Values by Column")

    missing_df = pd.DataFrame({
        "Column": raw_data.columns,
        "Before Cleaning": raw_data.isnull().sum().values,
        "After Cleaning": data.isnull().sum().values,
    })

    st.dataframe(missing_df, use_container_width=True)

    st.subheader("Valid Player Position Counts")
    st.dataframe(
        data["class"].value_counts().rename_axis("Position").reset_index(
            name="Players"
        ),
        use_container_width=True
    )

    st.subheader("Cleaned Dataset Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.download_button(
        "⬇️ Download Cleaned Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="football_player_position_cleaned.csv",
        mime="text/csv"
    )


# ---------------- MODEL EVALUATION ----------------
elif page == "Model Evaluation":
    st.header("🤖 Machine Learning Model Evaluation")

    st.metric("Gaussian Naive Bayes Accuracy", f"{accuracy:.2%}")

    st.caption(
        "Accuracy is measured on the held-out test set. "
        "It is not a guarantee of real-world prediction performance."
    )

    st.subheader("Confusion Matrix")

    labels = model.classes_
    cm = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots()
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Position")
    ax.set_ylabel("Actual Position")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

    st.subheader("Actual vs Predicted")
    comparison = pd.DataFrame({
        "Actual Position": y_test.values,
        "Predicted Position": predictions,
    })

    st.dataframe(comparison.head(50), use_container_width=True)


# ---------------- PLAYER PREDICTION ----------------
elif page == "Predict Position":
    st.header("⚽ Predict a Football Player's Position")

    st.write(
        "Enter player statistics below. The trained model "
        "will predict the most likely position."
    )

    with st.form("player_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input(
                "Age", min_value=17, max_value=40, value=25
            )
            height = st.number_input(
                "Height (cm)", min_value=158.0, max_value=205.0,
                value=180.0
            )
            weight = st.number_input(
                "Weight (kg)", min_value=52, max_value=110, value=78
            )
            matches = st.number_input(
                "Matches Played", min_value=1, max_value=50, value=20
            )

        with col2:
            goals = st.number_input(
                "Goals", min_value=0, max_value=100, value=5
            )
            assists = st.number_input(
                "Assists", min_value=0, max_value=100, value=3
            )
            passes = st.number_input(
                "Passes Completed", min_value=0, value=500
            )
            pass_accuracy = st.slider(
                "Pass Accuracy (%)", 55.0, 98.0, 80.0
            )

        with col3:
            shots = st.number_input(
                "Shots per Match", min_value=0.0,
                max_value=10.0, value=1.0, step=0.1
            )
            tackles = st.number_input(
                "Tackles per Match", min_value=0.0,
                max_value=10.0, value=1.0, step=0.1
            )
            saves = st.number_input(
                "Saves per Match", min_value=0.0,
                max_value=15.0, value=0.1, step=0.1
            )

        submitted = st.form_submit_button(
            "🔍 Predict Player Position",
            use_container_width=True
        )

    if submitted:
        player = pd.DataFrame([{
            "age": age,
            "height_cm": height,
            "weight_kg": weight,
            "matches_played": matches,
            "goals": goals,
            "assists": assists,
            "passes_completed": passes,
            "pass_accuracy_percent": pass_accuracy,
            "shots_per_match": shots,
            "tackles_per_match": tackles,
            "saves_per_match": saves,
        }], columns=FEATURES)

        prediction = model.predict(player)[0]
        probabilities = model.predict_proba(player)[0]

        st.success(f"Predicted Position: {prediction}")

        st.subheader("Prediction Probabilities")

        probability_df = pd.DataFrame({
            "Position": model.classes_,
            "Probability": probabilities,
        }).sort_values("Probability", ascending=False)

        st.bar_chart(
            probability_df.set_index("Position")["Probability"]
        )

        st.dataframe(
            probability_df.assign(
                Probability=probability_df["Probability"].map(
                    lambda value: f"{value:.2%}"
                )
            ),
            use_container_width=True
        )

st.divider()
st.caption("PITCHIQ | Football Player Position Prediction")
