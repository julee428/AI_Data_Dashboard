import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import plotly.express as px

st.set_page_config(
    page_title="AI Data Visualization Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- HEADER ----------------

st.title("📊 AI-Powered Data Visualization Dashboard")
st.write("Upload, clean, analyze and predict insights from your dataset.")

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("Dashboard Menu")

menu = st.sidebar.radio(
    "Select Section",
    [
        "Home",
        "Upload Dataset",
        "Data Cleaning",
        "Data Analysis",
        "ML Prediction",
        "Anomaly Detection",
        "AI Insights",
        "Power BI",
        "MySQL"
    ]
)

# ---------------- HOME ----------------

if menu == "Home":

    st.header("🏠 Welcome")

    st.write(
        "This dashboard helps users upload datasets, clean data, "
        "analyze patterns, detect anomalies and generate useful insights."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Processing", "Pandas + NumPy")

    with col2:
        st.metric("Machine Learning", "Scikit-learn")

    with col3:
        st.metric("Visualization", "Power BI")

    st.info(
        "Start by going to 'Upload Dataset' from the sidebar."
    )


# ---------------- UPLOAD DATASET ----------------

elif menu == "Upload Dataset":

    st.header("📂 Upload Dataset")

    file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if file:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.session_state["df"] = df

        st.success("Dataset uploaded successfully! ✅")

        st.subheader("📋 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader("📊 Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicates",
                int(df.duplicated().sum())
            )

    else:
        st.info("Please upload a CSV or Excel file.")


# ---------------- CHECK DATA ----------------

if "df" in st.session_state:

    df = st.session_state["df"]


# ---------------- DATA CLEANING ----------------

if menu == "Data Cleaning":

    st.header("🧹 Data Cleaning")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"].copy()

        st.subheader("Before Cleaning")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Missing Values:")
            st.write(df.isnull().sum())

        with col2:
            st.write(
                "Duplicate Rows:",
                df.duplicated().sum()
            )

        if st.button("Clean Dataset"):

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns

            for column in numeric_columns:
                df[column] = df[column].fillna(
                    df[column].median()
                )

            text_columns = df.select_dtypes(
                exclude=np.number
            ).columns

            for column in text_columns:
                if df[column].isnull().sum() > 0:
                    df[column] = df[column].fillna("Unknown")

            df = df.drop_duplicates()

            st.session_state["df"] = df

            st.success("Data cleaned successfully! ✅")

            st.subheader("Cleaned Dataset")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            csv = df.to_csv(index=False)

            st.download_button(
                "⬇️ Download Clean Dataset",
                csv,
                "cleaned_dataset.csv",
                "text/csv"
            )


# ---------------- DATA ANALYSIS ----------------

elif menu == "Data Analysis":

    st.header("📈 Data Analysis")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all").transpose(),
            use_container_width=True
        )

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_column = st.selectbox(
                "Select column for visualization",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=selected_column,
                title=f"Distribution of {selected_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No numerical columns found.")


# ---------------- ML PREDICTION ----------------

elif menu == "ML Prediction":

    st.header("🤖 Machine Learning Prediction")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) >= 2:

            target = st.selectbox(
                "Select Target Column",
                numeric_columns
            )

            features = [
                col for col in numeric_columns
                if col != target
            ]

            if features:

                X = df[features].fillna(0)
                y = df[target].fillna(0)

                if st.button("Train ML Model"):

                    model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )

                    model.fit(X, y)

                    predictions = model.predict(X)

                    result = df.copy()

                    result["Predicted"] = predictions

                    st.success(
                        "Machine Learning model trained successfully! ✅"
                    )

                    st.dataframe(
                        result.head(10),
                        use_container_width=True
                    )

        else:

            st.info(
                "At least 2 numerical columns are required."
            )


# ---------------- ANOMALY DETECTION ----------------

elif menu == "Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) >= 1:

            selected = st.multiselect(
                "Select columns",
                numeric_columns,
                default=numeric_columns[:1]
            )

            if selected and st.button(
                "Detect Anomalies"
            ):

                data = df[selected].fillna(0)

                model = IsolationForest(
                    contamination=0.05,
                    random_state=42
                )

                result = df.copy()

                result["Anomaly"] = model.fit_predict(data)

                result["Anomaly"] = result[
                    "Anomaly"
                ].map({
                    1: "Normal",
                    -1: "Anomaly"
                })

                st.success(
                    "Anomaly detection completed! ✅"
                )

                st.dataframe(
                    result,
                    use_container_width=True
                )

                st.write(
                    "Detected Anomalies:",
                    (result["Anomaly"] == "Anomaly").sum()
                )

        else:

            st.info("No numerical columns found.")


# ---------------- AI INSIGHTS ----------------

elif menu == "AI Insights":

    st.header("🧠 AI-Based Insights")

    if "df" not in st.session_state:

        st.warning("Please upload a dataset first.")

    else:

        df = st.session_state["df"]

        st.subheader("Dataset Insights")

        st.write(
            f"• Dataset contains **{df.shape[0]} rows** "
            f"and **{df.shape[1]} columns**."
        )

        missing = df.isnull().sum().sum()

        if missing > 0:

            st.write(
                f"• Dataset contains **{missing} missing values**."
            )

        else:

            st.write(
                "• Dataset has **no missing values**."
            )

        duplicates = df.duplicated().sum()

        if duplicates > 0:

            st.write(
                f"• Dataset contains **{duplicates} duplicate rows**."
            )

        else:

            st.write(
                "• No duplicate rows were found."
            )

        numeric = df.select_dtypes(
            include=np.number
        )

        if not numeric.empty:

            st.subheader("Numerical Insights")

            for column in numeric.columns:

                st.write(
                    f"• **{column}** average: "
                    f"{numeric[column].mean():.2f}"
                )


# ---------------- POWER BI ----------------

elif menu == "Power BI":

    st.header("📊 Power BI Dashboard")

    st.write(
        "Power BI will be used for advanced interactive "
        "visualization and dashboard creation."
    )

    st.info(
        "Power BI integration/embed can be added after "
        "creating the final Power BI dashboard."
    )

    st.write(
        "Workflow:"
    )

    st.code(
        "Clean Dataset → Power BI → Interactive Dashboard"
    )


# ---------------- MYSQL ----------------

elif menu == "MySQL":

    st.header("🗄️ MySQL Database")

    st.write(
        "MySQL will store processed datasets and project data."
    )

    host = st.text_input(
        "Host",
        "localhost"
    )

    database = st.text_input(
        "Database Name",
        "ai_dashboard"
    )

    username = st.text_input(
        "Username",
        "root"
    )

    password = st.text_input(
        "Password",
        type="password"
    )

    if st.button("Connect to MySQL"):

        try:

            import mysql.connector

            connection = mysql.connector.connect(
                host=host,
                user=username,
                password=password,
                database=database
            )

            if connection.is_connected():

                st.success(
                    "MySQL connected successfully! ✅"
                )

                connection.close()

        except Exception as e:

            st.error(
                f"Connection failed: {e}"
            )
            