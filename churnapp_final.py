
import re
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="ChurnGuard Pro", page_icon="📉", layout="wide")
st.title("📉 ChurnGuard Pro")
st.caption("Telco churn prediction, retention planning, custom chart builder, and chatbot")


# -----------------------------
# Data loading and cleaning
# -----------------------------
DEFAULT_DATA_FILES = [
    "telco_churn.xlsx",
    "Telco_customer_churn(1).xlsx",
    "Telco_customer_churn.xlsx",
]
TARGET_COL = "Churn Value"
ID_COL = "CustomerID"

MODEL_FEATURES = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]

DISPLAY_COLUMNS = [
    ID_COL,
    "City",
    "Gender",
    "Senior Citizen",
    "Tenure Months",
    "Contract",
    "Internet Service",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "churn_probability",
    "risk_level",
]


@st.cache_data
def load_telco_data(file_buffer=None) -> pd.DataFrame:
    if file_buffer is not None:
        df = pd.read_excel(file_buffer)
    else:
        data_path = next((f for f in DEFAULT_DATA_FILES if Path(f).exists()), None)
        if data_path is None:
            raise FileNotFoundError(
                "Dataset file not found. Upload the Excel file from the sidebar or place it in the same folder as this app."
            )
        df = pd.read_excel(data_path)

    df.columns = [str(c).strip() for c in df.columns]

    for col in ["Total Charges", "Monthly Charges", "Tenure Months", "CLTV", "Churn Score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Senior Citizen" in df.columns:
        df["Senior Citizen"] = (
            df["Senior Citizen"]
            .replace({"Yes": 1, "No": 0})
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    object_cols = df.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        df[col] = df[col].fillna("Unknown").replace({"": "Unknown"}).astype(str).str.strip()

    numeric_cols = [c for c in ["Total Charges", "Monthly Charges", "Tenure Months", "CLTV", "Churn Score"] if c in df.columns]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    if "Churn Label" in df.columns:
        df["Churn Label"] = df["Churn Label"].replace({"Yes": 1, "No": 0}).fillna(df[TARGET_COL]).astype(int)

    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET_COL].astype(int)

    categorical_cols = [c for c in MODEL_FEATURES if X[c].dtype == "object"]
    numeric_cols = [c for c in MODEL_FEATURES if c not in categorical_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "feature_cols": MODEL_FEATURES,
        "X_test": X_test,
        "y_test": y_test,
    }
    return pipeline, metrics


# -----------------------------
# Helper functions
# -----------------------------
def risk_band(prob: float) -> str:
    if prob >= 0.65:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def explain_customer(row: pd.Series) -> list[str]:
    reasons = []

    churn_reason = str(row.get("Churn Reason", "Unknown")).strip()
    churn_score = float(row.get("Churn Score", 0) or 0)
    monthly = float(row.get("Monthly Charges", 0) or 0)
    tenure = int(row.get("Tenure Months", 0) or 0)
    total = float(row.get("Total Charges", 0) or 0)

    if churn_reason not in ["Unknown", "No", ""]:
        reasons.append(f"Recorded churn reason in the dataset: {churn_reason}")

    if row.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract gives the customer an easy exit path")
    elif row.get("Contract") == "One year":
        reasons.append("One year contract reduces churn a bit, but not as much as a two year plan")

    if tenure <= 6:
        reasons.append(f"Very short tenure of {tenure} months suggests weak loyalty")
    elif tenure <= 12:
        reasons.append(f"Tenure is only {tenure} months, so the relationship is still early")

    if monthly >= 90:
        reasons.append(f"Monthly charges are high at ${monthly:.2f}, which can push cancellation risk up")
    elif monthly <= 30:
        reasons.append(f"Low monthly charges at ${monthly:.2f} may help retention")

    if row.get("Internet Service") == "Fiber optic":
        reasons.append("Fiber optic users in this dataset show higher churn more often than DSL users")
    elif row.get("Internet Service") == "No":
        reasons.append("No internet service limits product engagement and upsell potential")

    if row.get("Payment Method") == "Electronic check":
        reasons.append("Electronic check customers tend to show higher churn risk in this dataset")
    elif row.get("Payment Method") in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        reasons.append("Automatic payment usually supports more stable retention")

    if row.get("Tech Support") == "No":
        reasons.append("No tech support can increase frustration when service issues appear")
    if row.get("Online Security") == "No":
        reasons.append("No online security can reduce perceived value of the service")
    if row.get("Online Backup") == "No":
        reasons.append("No online backup lowers the stickiness of the service bundle")
    if row.get("Device Protection") == "No":
        reasons.append("No device protection means fewer service benefits are attached to the account")

    if row.get("Streaming TV") == "No" and row.get("Streaming Movies") == "No":
        reasons.append("The customer is not using entertainment add-ons, which can mean lower engagement")
    elif row.get("Streaming TV") == "Yes" or row.get("Streaming Movies") == "Yes":
        reasons.append("Streaming services increase usage, but they may also raise the bill")

    if row.get("Partner") == "No" and row.get("Dependents") == "No":
        reasons.append("No partner and no dependents may indicate fewer switching barriers")
    elif row.get("Dependents") == "Yes":
        reasons.append("Dependents can create a reason to stay if value and support remain strong")

    if total > 0 and tenure > 0 and total < monthly * max(tenure * 0.6, 1):
        reasons.append("Total spend is relatively low for the tenure, which may reflect a fragile relationship")

    if churn_score >= 80:
        reasons.append(f"Churn score is very high at {churn_score:.0f}")
    elif churn_score >= 60:
        reasons.append(f"Churn score is elevated at {churn_score:.0f}")

    unique_reasons = []
    for reason in reasons:
        if reason not in unique_reasons:
            unique_reasons.append(reason)

    if not unique_reasons:
        unique_reasons.append("This customer shows no strong individual churn trigger in the available fields")

    return unique_reasons[:5]


def get_retention_actions(row: pd.Series) -> list[str]:
    actions = []

    churn_reason = str(row.get("Churn Reason", "Unknown")).strip()
    monthly = float(row.get("Monthly Charges", 0) or 0)
    tenure = int(row.get("Tenure Months", 0) or 0)
    risk = str(row.get("risk_level", "Medium"))

    if "competitor" in churn_reason.lower():
        actions.append("Offer a targeted competitor match discount or extra value bundle")
    if "attitude" in churn_reason.lower() or "support" in churn_reason.lower():
        actions.append("Escalate to a service recovery call with priority support follow-up")
    if "price" in churn_reason.lower() or "offer" in churn_reason.lower():
        actions.append("Present a lower cost plan or a limited time bill credit")
    if "network" in churn_reason.lower() or "reliability" in churn_reason.lower():
        actions.append("Open a network quality review and provide a temporary goodwill credit")
    if "move" in churn_reason.lower():
        actions.append("Offer an address transfer or remote service continuity plan")

    if row.get("Contract") == "Month-to-month":
        actions.append("Offer a 12 month or 24 month contract with a discount")
    elif row.get("Contract") == "One year":
        actions.append("Offer an upgrade path to a two year contract with added benefits")

    if monthly >= 90:
        actions.append(f"Reduce the monthly bill from ${monthly:.2f} with a cheaper bundled plan")
    elif monthly >= 70:
        actions.append("Review the bill and recommend a better value package")

    if tenure <= 6:
        actions.append("Trigger an early life onboarding journey with a welcome incentive")
    elif tenure <= 12:
        actions.append("Send a retention check-in and first year loyalty offer")

    if row.get("Tech Support") == "No":
        actions.append("Provide free tech support for one month")
    if row.get("Online Security") == "No":
        actions.append("Offer a security add-on trial")
    if row.get("Online Backup") == "No":
        actions.append("Offer free online backup for a trial period")
    if row.get("Device Protection") == "No":
        actions.append("Bundle device protection into a retention package")

    if row.get("Payment Method") == "Electronic check":
        actions.append("Encourage autopay with a billing credit")
    elif row.get("Payment Method") == "Mailed check":
        actions.append("Promote digital billing and autopay for convenience")

    if row.get("Streaming TV") == "No" and row.get("Streaming Movies") == "No":
        actions.append("Offer an entertainment bundle trial to increase engagement")

    if row.get("Partner") == "No" and row.get("Dependents") == "No":
        actions.append("Use a personalized individual offer rather than a family bundle")

    if risk == "Low" and not actions:
        actions.append("Maintain engagement with routine loyalty messaging")
    elif not actions:
        actions.append("Schedule a proactive retention review and monitor the account")

    unique_actions = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)

    return unique_actions[:5]


def build_scored_dataframe(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    scored = df.copy()
    scored["churn_probability"] = pipeline.predict_proba(scored[MODEL_FEATURES])[:, 1]
    scored["risk_level"] = scored["churn_probability"].apply(risk_band)
    return scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def build_top_driver_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    churned = scored_df[scored_df[TARGET_COL] == 1]
    if churned.empty:
        return pd.DataFrame(columns=["reason", "count"])
    out = (
        churned["Churn Reason"]
        .fillna("Unknown")
        .replace({"": "Unknown"})
        .value_counts()
        .reset_index()
    )
    out.columns = ["reason", "count"]
    return out.head(10)


def chatbot_response(query: str, scored_df: pd.DataFrame) -> str:
    q = query.lower().strip()

    cust_match = re.search(r"[a-z0-9]{4}-[a-z0-9]{5}", q)
    if cust_match:
        cust_id = cust_match.group(0).upper()
        row = scored_df.loc[scored_df[ID_COL].str.upper() == cust_id]
        if row.empty:
            return f"I could not find customer {cust_id}. Try a valid CustomerID from the dashboard."
        row = row.iloc[0]
        reasons = explain_customer(row)
        actions = get_retention_actions(row)
        return (
            f"{cust_id} has a churn probability of {row['churn_probability']:.1%} and is in the {row['risk_level']} risk group. "
            f"Main reasons: {', '.join(reasons[:2])}. Suggested action: {actions[0]}."
        )

    if "highest risk" in q or "top risk" in q or "high risk customers" in q:
        top5 = scored_df[[ID_COL, "churn_probability"]].head(5)
        items = [f"{r[ID_COL]} ({r['churn_probability']:.1%})" for _, r in top5.iterrows()]
        return "Top high-risk customers: " + ", ".join(items)

    if "how many" in q and "high risk" in q:
        count = int((scored_df["risk_level"] == "High").sum())
        return f"There are {count} high-risk customers in the current dataset."

    if "churn rate" in q:
        rate = scored_df[TARGET_COL].mean()
        return f"Observed churn rate in this dataset is {rate:.1%}."

    if "top drivers" in q or "why do customers churn" in q or "drivers" in q:
        reasons = build_top_driver_summary(scored_df)
        if reasons.empty:
            return "I do not have churn reason records to summarize."
        top = ", ".join([f"{r.reason} ({r.count})" for r in reasons.head(3).itertuples()])
        return f"Top recorded churn reasons are {top}."

    if "month to month" in q:
        subset = scored_df[scored_df["Contract"] == "Month-to-month"]
        return f"Month-to-month customers have an average predicted churn probability of {subset['churn_probability'].mean():.1%}."

    return (
        "Try asking: 'Which customers are at highest risk?', 'How many high-risk customers do we have?', "
        "'What is the churn rate?', 'What are the top drivers?', or 'Why will 3668-QPYBK churn?'"
    )


def plot_custom_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str | None = None, agg: str = "mean"):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    if chart_type == "Bar":
        if y_col is None:
            plot_df = df[x_col].value_counts().head(15)
            ax.bar(plot_df.index.astype(str), plot_df.values)
            ax.set_ylabel("Count")
        else:
            grouped = df.groupby(x_col, dropna=False)[y_col].agg(agg).sort_values(ascending=False).head(15)
            ax.bar(grouped.index.astype(str), grouped.values)
            ax.set_ylabel(f"{agg.title()} of {y_col}")
        ax.set_xlabel(x_col)
        plt.xticks(rotation=30, ha="right")

    elif chart_type == "Line":
        if y_col is None:
            tmp = df[x_col].value_counts().sort_index()
            ax.plot(tmp.index.astype(str), tmp.values)
            ax.set_ylabel("Count")
        else:
            grouped = df.groupby(x_col, dropna=False)[y_col].agg(agg).sort_index()
            ax.plot(grouped.index, grouped.values)
            ax.set_ylabel(f"{agg.title()} of {y_col}")
        ax.set_xlabel(x_col)
        plt.xticks(rotation=30, ha="right")

    elif chart_type == "Histogram":
        ax.hist(df[x_col].dropna(), bins=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel("Frequency")

    elif chart_type == "Scatter":
        if y_col is None:
            raise ValueError("Scatter plots require both X and Y fields.")
        ax.scatter(df[x_col], df[y_col], alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)

    elif chart_type == "Box":
        if y_col is None:
            ax.boxplot(df[x_col].dropna())
            ax.set_ylabel(x_col)
        else:
            groups = []
            labels = []
            for category, subset in df.groupby(x_col):
                vals = subset[y_col].dropna().values
                if len(vals) > 0:
                    groups.append(vals)
                    labels.append(str(category))
            ax.boxplot(groups, tick_labels=labels)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=30, ha="right")

    ax.set_title(f"{chart_type} chart")
    st.pyplot(fig)


# -----------------------------
# Data source selector
# -----------------------------
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Telco churn Excel file", type=["xlsx"])
source_label = "Uploaded file" if uploaded_file is not None else "Local file search enabled"
st.sidebar.caption(source_label)

try:
    raw_df = load_telco_data(uploaded_file)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

pipeline, metrics = train_model(raw_df)
scored_df = build_scored_dataframe(raw_df, pipeline)


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
risk_filter = st.sidebar.multiselect(
    "Risk level",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
)
contract_filter = st.sidebar.multiselect(
    "Contract",
    options=sorted(scored_df["Contract"].dropna().unique().tolist()),
    default=sorted(scored_df["Contract"].dropna().unique().tolist()),
)
charge_min, charge_max = st.sidebar.slider(
    "Monthly charges",
    float(scored_df["Monthly Charges"].min()),
    float(scored_df["Monthly Charges"].max()),
    (
        float(scored_df["Monthly Charges"].min()),
        float(scored_df["Monthly Charges"].max()),
    ),
)

filtered_df = scored_df[
    (scored_df["risk_level"].isin(risk_filter))
    & (scored_df["Contract"].isin(contract_filter))
    & (scored_df["Monthly Charges"].between(charge_min, charge_max))
].copy()


# -----------------------------
# KPI section
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(scored_df):,}")
col2.metric("High risk", int((scored_df['risk_level'] == 'High').sum()))
col3.metric("Model ROC AUC", f"{metrics['roc_auc']:.2f}")
col4.metric("Observed churn rate", f"{scored_df[TARGET_COL].mean():.1%}")


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Risk Dashboard",
    "Customer Insights",
    "Custom Chart Builder",
    "What-if Simulator",
    "Churn Chatbot",
])


with tab1:
    st.subheader("Customer Risk Dashboard")
    display_df = filtered_df[DISPLAY_COLUMNS].copy()
    display_df["churn_probability"] = (display_df["churn_probability"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk distribution")
        risk_counts = scored_df["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(risk_counts.index, risk_counts.values)
        ax.set_xlabel("Risk level")
        ax.set_ylabel("Customers")
        st.pyplot(fig)

    with c2:
        st.subheader("Average churn probability by contract")
        contract_view = scored_df.groupby("Contract", as_index=False)["churn_probability"].mean().sort_values("churn_probability", ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(contract_view["Contract"], contract_view["churn_probability"])
        ax.set_xlabel("Contract")
        ax.set_ylabel("Average churn probability")
        plt.xticks(rotation=15)
        st.pyplot(fig)

    st.subheader("Top recorded churn reasons")
    reason_df = build_top_driver_summary(scored_df)
    if not reason_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(reason_df["reason"], reason_df["count"])
        ax.set_xlabel("Churn reason")
        ax.set_ylabel("Customers")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)


with tab2:
    st.subheader("Customer-Level Insights")
    customer_choice = st.selectbox("Select customer", scored_df[ID_COL].tolist())
    customer_row = scored_df.loc[scored_df[ID_COL] == customer_choice].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{customer_row['churn_probability']:.1%}")
    c2.metric("Risk level", customer_row["risk_level"])
    c3.metric("Tenure", f"{int(customer_row['Tenure Months'])} months")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Customer profile")
        st.write(
            {
                "City": customer_row["City"],
                "Gender": customer_row["Gender"],
                "Contract": customer_row["Contract"],
                "Internet service": customer_row["Internet Service"],
                "Payment method": customer_row["Payment Method"],
                "Monthly charges": float(customer_row["Monthly Charges"]),
                "Tech support": customer_row["Tech Support"],
                "Online security": customer_row["Online Security"],
                "Paperless billing": customer_row["Paperless Billing"],
                "Churn score": int(customer_row["Churn Score"]) if pd.notna(customer_row["Churn Score"]) else "Unknown",
                "Recorded churn reason": customer_row["Churn Reason"],
            }
        )

    with right:
        st.markdown("### Why this customer may churn")
        for reason in explain_customer(customer_row):
            st.write(f"• {reason}")

    st.markdown("### Recommended retention actions")
    for action in get_retention_actions(customer_row):
        st.success(action)

    st.markdown("### Model-wide feature importance")
    subset = scored_df.sample(min(500, len(scored_df)), random_state=42)
    X_subset = subset[metrics["feature_cols"]]
    y_subset = subset[TARGET_COL]
    perm = permutation_importance(
        pipeline,
        X_subset,
        y_subset,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
    )
    feat_imp = pd.DataFrame(
        {
            "feature": X_subset.columns,
            "importance": perm.importances_mean,
        }
    ).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feat_imp["feature"], feat_imp["importance"])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Permutation importance")
    plt.xticks(rotation=35, ha="right")
    st.pyplot(fig)


with tab3:
    st.subheader("Custom Chart Builder")
    st.caption("Build your own chart from the churn dataset")

    all_columns = scored_df.columns.tolist()
    numeric_columns = scored_df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [c for c in all_columns if c not in numeric_columns]

    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Histogram", "Scatter", "Box"])

    if chart_type in ["Histogram"]:
        x_col = st.selectbox("Numeric field", numeric_columns, index=numeric_columns.index("Monthly Charges") if "Monthly Charges" in numeric_columns else 0)
        plot_custom_chart(filtered_df, chart_type, x_col)
    elif chart_type in ["Scatter"]:
        x_col = st.selectbox("X axis", numeric_columns, index=numeric_columns.index("Tenure Months") if "Tenure Months" in numeric_columns else 0)
        y_col = st.selectbox("Y axis", numeric_columns, index=numeric_columns.index("Monthly Charges") if "Monthly Charges" in numeric_columns else min(1, len(numeric_columns)-1))
        plot_custom_chart(filtered_df, chart_type, x_col, y_col)
    elif chart_type == "Box":
        mode = st.radio("Box plot mode", ["One numeric field", "Category vs numeric"], horizontal=True)
        if mode == "One numeric field":
            x_col = st.selectbox("Numeric field", numeric_columns)
            plot_custom_chart(filtered_df, chart_type, x_col)
        else:
            x_col = st.selectbox("Category", categorical_columns, index=categorical_columns.index("Contract") if "Contract" in categorical_columns else 0)
            y_col = st.selectbox("Numeric field", numeric_columns, index=numeric_columns.index("Monthly Charges") if "Monthly Charges" in numeric_columns else 0)
            plot_custom_chart(filtered_df, chart_type, x_col, y_col)
    else:
        x_options = categorical_columns if chart_type == "Bar" else numeric_columns + categorical_columns
        x_default = "Contract" if "Contract" in x_options else x_options[0]
        x_col = st.selectbox("X axis", x_options, index=x_options.index(x_default))
        use_y = st.checkbox("Aggregate a second field", value=True)
        y_col = None
        agg = "mean"
        if use_y:
            y_col = st.selectbox("Y field", numeric_columns, index=numeric_columns.index("churn_probability") if "churn_probability" in numeric_columns else 0)
            agg = st.selectbox("Aggregation", ["mean", "median", "sum", "count"])
        plot_custom_chart(filtered_df, chart_type, x_col, y_col, agg)

    st.markdown("### Quick chart ideas")
    st.write("Compare churn probability by contract, analyze monthly charges, or inspect tenure versus total charges.")


with tab4:
    st.subheader("What-if Simulator")
    st.caption("Change customer attributes and see how churn risk changes")

    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_gender = st.selectbox("Gender", sorted(raw_df["Gender"].unique().tolist()))
        sim_senior = st.selectbox("Senior Citizen", [0, 1])
        sim_partner = st.selectbox("Partner", ["Yes", "No"])
        sim_dependents = st.selectbox("Dependents", ["Yes", "No"])
        sim_tenure = st.slider("Tenure Months", 0, 72, 12)
        sim_phone = st.selectbox("Phone Service", sorted(raw_df["Phone Service"].unique().tolist()))
        sim_multiple = st.selectbox("Multiple Lines", sorted(raw_df["Multiple Lines"].unique().tolist()))
        sim_internet = st.selectbox("Internet Service", sorted(raw_df["Internet Service"].unique().tolist()))
        sim_security = st.selectbox("Online Security", sorted(raw_df["Online Security"].unique().tolist()))

    with sim_col2:
        sim_backup = st.selectbox("Online Backup", sorted(raw_df["Online Backup"].unique().tolist()))
        sim_device = st.selectbox("Device Protection", sorted(raw_df["Device Protection"].unique().tolist()))
        sim_tech = st.selectbox("Tech Support", sorted(raw_df["Tech Support"].unique().tolist()))
        sim_tv = st.selectbox("Streaming TV", sorted(raw_df["Streaming TV"].unique().tolist()))
        sim_movies = st.selectbox("Streaming Movies", sorted(raw_df["Streaming Movies"].unique().tolist()))
        sim_contract = st.selectbox("Contract", sorted(raw_df["Contract"].unique().tolist()))
        sim_paperless = st.selectbox("Paperless Billing", sorted(raw_df["Paperless Billing"].unique().tolist()))
        sim_payment = st.selectbox("Payment Method", sorted(raw_df["Payment Method"].unique().tolist()))
        sim_monthly = st.slider("Monthly Charges", float(raw_df["Monthly Charges"].min()), float(raw_df["Monthly Charges"].max()), 80.0)

    sim_total = round(max(sim_tenure, 1) * sim_monthly, 2)
    sim_df = pd.DataFrame([
        {
            "Gender": sim_gender,
            "Senior Citizen": sim_senior,
            "Partner": sim_partner,
            "Dependents": sim_dependents,
            "Tenure Months": sim_tenure,
            "Phone Service": sim_phone,
            "Multiple Lines": sim_multiple,
            "Internet Service": sim_internet,
            "Online Security": sim_security,
            "Online Backup": sim_backup,
            "Device Protection": sim_device,
            "Tech Support": sim_tech,
            "Streaming TV": sim_tv,
            "Streaming Movies": sim_movies,
            "Contract": sim_contract,
            "Paperless Billing": sim_paperless,
            "Payment Method": sim_payment,
            "Monthly Charges": sim_monthly,
            "Total Charges": sim_total,
        }
    ])

    sim_prob = pipeline.predict_proba(sim_df)[0, 1]
    st.metric("Predicted churn probability", f"{sim_prob:.1%}")
    st.write(f"Risk level: **{risk_band(sim_prob)}**")

    sim_row = pd.Series({**sim_df.iloc[0].to_dict(), "churn_probability": sim_prob, "risk_level": risk_band(sim_prob)})
    st.markdown("### Suggested actions")
    for action in get_retention_actions(sim_row):
        st.info(action)


with tab5:
    st.subheader("Churn Chatbot")
    st.caption("Ask simple questions about churn risk, drivers, and individual customers")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            (
                "assistant",
                "Ask about high-risk customers, churn rate, churn drivers, or a customer like 3668-QPYBK.",
            )
        ]

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    user_query = st.chat_input("Ask a churn question")
    if user_query:
        st.session_state.chat_history.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)

        reply = chatbot_response(user_query, scored_df)
        st.session_state.chat_history.append(("assistant", reply))
        with st.chat_message("assistant"):
            st.write(reply)


with st.expander("Project notes"):
    st.write(
        "This version uses the IBM Telco-style churn dataset rather than synthetic data. "
        "It includes a real dataset workflow, customer scoring, customer specific retention recommendations, "
        "a custom chart builder, and a chatbot."
    )
