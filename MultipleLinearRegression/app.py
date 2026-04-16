import pathlib

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = pathlib.Path(__file__).parent / "2-multiplegradesdataset.csv"
TARGET_COLUMN = "Exam Score"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model(df: pd.DataFrame):
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return model, feature_columns, metrics


def main():
    st.set_page_config(page_title="Multiple Linear Regression UI", layout="wide")
    st.title("Multiple Linear Regression Interface")

    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        return

    df = load_data()
    model, feature_columns, metrics = train_model(df)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Enter new student")
        user_input = {}
        for column in feature_columns:
            avg_value = float(df[column].mean())
            step = max(abs(avg_value) / 100, 0.1)
            user_input[column] = st.number_input(
                column,
                value=avg_value,
                step=step,
            )

        if st.button("Lets predict", use_container_width=True):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            st.success(f"Estimated Test Score: {prediction:.2f}")

        st.markdown("---")
        st.metric("R2 Skoru", f"{metrics['r2']:.3f}")
        st.metric("MAE", f"{metrics['mae']:.3f}")

    with right:
        st.subheader("Data vİsualization")
        st.dataframe(df, use_container_width=True)

        st.subheader("Target Variable Distributioni")
        fig = px.histogram(
            df,
            x=TARGET_COLUMN,
            nbins=20,
            title="Exam Score Histogram",
            color_discrete_sequence=["#4f46e5"],
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature-Correlation Map")
        corr = df.corr(numeric_only=True)
        heatmap = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Correlation Matrix",
        )
        st.plotly_chart(heatmap, use_container_width=True)


if __name__ == "__main__":
    main()
