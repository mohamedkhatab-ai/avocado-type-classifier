import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. Page Configuration & Custom Dark Mode CSS
# -------------------------------------------------------------
st.set_page_config(
    page_title="Avocado Intelligence | Type Classifier",
    page_icon="🥑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme Styling
st.markdown("""
<style>
    /* Global background and font setup */
    .main {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    /* Sleek Container Cards */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #00E676 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Glassmorphism Box */
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(10px);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Success & Prediction Badges */
    .pred-box-organic {
        background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 100%);
        border: 1px solid #00E676;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #FFFFFF;
    }
    
    .pred-box-conventional {
        background: linear-gradient(135deg, #E65100 0%, #F57C00 100%);
        border: 1px solid #FF9800;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# 2. PyTorch Architecture Definition
# -------------------------------------------------------------
class TypeClassifierNN(nn.Module):
    """Multi-Layer Perceptron (MLP) for Binary Type Classification."""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# 3. Data Processing Pipeline
# -------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(file_path_or_buffer):
    """Loads CSV, performs feature extraction, and categorical encoding."""
    df = pd.read_csv(file_path_or_buffer)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Date feature engineering
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['Year_Val'] = df['year']

    # Encoders
    le_region = LabelEncoder()
    df['region_code'] = le_region.fit_transform(df['region'])

    le_type = LabelEncoder()
    df['type_code'] = le_type.fit_transform(df['type'])  # 0: conventional, 1: organic

    features = [
        'AveragePrice', 'Total Volume', '4046', '4225', '4770', 
        'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 
        'year', 'Month', 'Day', 'region_code'
    ]
    
    return df, features, le_region, le_type


# -------------------------------------------------------------
# 4. Model Training Logic
# -------------------------------------------------------------
@st.cache_resource
def train_models(df, features, n_estimators, epochs, lr):
    X = df[features]
    y = df['type_code']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # AdaBoost Training
    ada_clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    ada_clf.fit(X_train, y_train)
    ada_train_acc = accuracy_score(y_train, ada_clf.predict(X_train))
    ada_test_acc = accuracy_score(y_test, ada_clf.predict(X_test))

    # PyTorch MLP Training
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32)

    torch.manual_seed(42)
    dl_model = TypeClassifierNN(X_train_t.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(dl_model.parameters(), lr=lr)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(dl_model(bx), by)
            loss.backward()
            optimizer.step()

    dl_model.eval()
    with torch.no_grad():
        train_pred = (dl_model(X_train_t).numpy() > 0.5).astype(int)
        test_pred = (dl_model(X_test_t).numpy() > 0.5).astype(int)

    dl_train_acc = accuracy_score(y_train, train_pred)
    dl_test_acc = accuracy_score(y_test, test_pred)

    metrics = {
        'ada_train_acc': ada_train_acc,
        'ada_test_acc': ada_test_acc,
        'dl_train_acc': dl_train_acc,
        'dl_test_acc': dl_test_acc,
        'ada_pred': ada_clf.predict(X_test),
        'dl_pred': test_pred,
        'y_test': y_test,
        'X_test': X_test
    }

    return ada_clf, dl_model, scaler, metrics


# -------------------------------------------------------------
# 5. Sidebar Controls
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/3d-fluency/94/avocado.png", width=80)
st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader("Upload Avocado Dataset (.csv)", type=["csv"])
file_target = uploaded_file if uploaded_file is not None else "avocado.csv"

st.sidebar.subheader("Hyperparameter Tuning")
n_estimators = st.sidebar.slider("AdaBoost Estimators", 10, 200, 100, step=10)
epochs = st.sidebar.slider("PyTorch MLP Epochs", 5, 100, 30, step=5)
lr = st.sidebar.select_slider("Learning Rate", options=[0.0001, 0.001, 0.005, 0.01, 0.05], value=0.005)


# -------------------------------------------------------------
# 6. Main Application Execution
# -------------------------------------------------------------
st.title("🥑 Avocado Intelligence Platform")
st.caption("Deep Learning & Ensemble Analytics Dashboard for Organic vs. Conventional Avocado Classification")

try:
    df, features, le_region, le_type = load_and_preprocess_data(file_target)
    ada_model, dl_model, scaler, metrics = train_models(df, features, n_estimators, epochs, lr)

    tab1, tab2, tab3 = st.tabs(["📊 Executive Analytics", "🤖 Model Evaluation", "🔮 Live Inference Engine"])

    # ---------------------------------------------------------
    # TAB 1: EXECUTIVE ANALYTICS
    # ---------------------------------------------------------
    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records", f"{len(df):,}")
        m2.metric("Average Price", f"${df['AveragePrice'].mean():.2f}")
        m3.metric("Total Volume Traded", f"{df['Total Volume'].sum()/1e6:.1f}M")
        m4.metric("Unique Regions", f"{df['region'].nunique()}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Price Distribution by Avocado Type")
            fig_price = px.histogram(
                df, x="AveragePrice", color="type",
                barmode="overlay", template="plotly_dark",
                color_discrete_map={'conventional': '#3B82F6', 'organic': '#00E676'}
            )
            fig_price.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_price, use_container_width=True)

        with c2:
            st.subheader("Volume Trend Across Years")
            vol_year = df.groupby(['year', 'type'])['Total Volume'].sum().reset_index()
            fig_vol = px.bar(
                vol_year, x="year", y="Total Volume", color="type",
                barmode="group", template="plotly_dark",
                color_discrete_map={'conventional': '#3B82F6', 'organic': '#00E676'}
            )
            fig_vol.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_vol, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: MODEL EVALUATION
    # ---------------------------------------------------------
    with tab2:
        col_ada, col_dl = st.columns(2)

        with col_ada:
            st.subheader("🌲 AdaBoost Classifier")
            st.write(f"**Train Accuracy:** `{metrics['ada_train_acc']*100:.2f}%`")
            st.write(f"**Test Accuracy:** `{metrics['ada_test_acc']*100:.2f}%`")
            
            # Confusion Matrix
            cm_ada = confusion_matrix(metrics['y_test'], metrics['ada_pred'])
            fig_cm_ada = px.imshow(
                cm_ada, text_auto=True, color_continuous_scale='Greens',
                x=le_type.classes_, y=le_type.classes_, title="AdaBoost Confusion Matrix"
            )
            fig_cm_ada.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm_ada, use_container_width=True)

        with col_dl:
            st.subheader("🧠 PyTorch Neural Network (MLP)")
            st.write(f"**Train Accuracy:** `{metrics['dl_train_acc']*100:.2f}%`")
            st.write(f"**Test Accuracy:** `{metrics['dl_test_acc']*100:.2f}%`")

            # Confusion Matrix
            cm_dl = confusion_matrix(metrics['y_test'], metrics['dl_pred'])
            fig_cm_dl = px.imshow(
                cm_dl, text_auto=True, color_continuous_scale='Blues',
                x=le_type.classes_, y=le_type.classes_, title="PyTorch MLP Confusion Matrix"
            )
            fig_cm_dl.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm_dl, use_container_width=True)

        st.markdown("---")
        st.subheader("Feature Importance (AdaBoost)")
        importances = pd.DataFrame({
            'Feature': features,
            'Importance': ada_model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(
            importances, x='Importance', y='Feature', orientation='h',
            template='plotly_dark', color='Importance', color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: LIVE INFERENCE ENGINE (WITH DATASET SAMPLER)
    # ---------------------------------------------------------
    with tab3:
        st.subheader("Interactive Type Predictor & Sample Selector")
        st.write("Select a row directly from the dataset or click 'Random Sample' to auto-fill input fields.")

        # Session state for row selection
        if 'selected_row_idx' not in st.session_state:
            st.session_state.selected_row_idx = 0

        def set_random_row():
            st.session_state.selected_row_idx = int(np.random.choice(df.index))

        # Dataset Sample Picker Section
        col_load1, col_load2 = st.columns([3, 1])
        with col_load1:
            selected_idx = st.selectbox(
                "📂 Select a sample row directly from dataset:",
                options=df.index,
                index=int(st.session_state.selected_row_idx),
                format_func=lambda x: f"Row {x} | Actual: {df.loc[x, 'type'].upper()} | Price: ${df.loc[x, 'AveragePrice']:.2f} | Region: {df.loc[x, 'region']}",
                key="sb_row_select"
            )
            st.session_state.selected_row_idx = selected_idx

        with col_load2:
            st.write(" ")
            st.write(" ")
            st.button("🎲 Random Sample", on_click=set_random_row, use_container_width=True)

        selected_row = df.loc[st.session_state.selected_row_idx]

        # Display actual class info badge
        actual_type = str(selected_row['type']).upper()
        st.info(f"📍 **Currently Loaded Row ({st.session_state.selected_row_idx}):** Actual Class in Dataset: **{actual_type}** | Price: **${selected_row['AveragePrice']}**")

        st.markdown("---")

        # Input Form pre-filled from Dataset
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            avg_price = st.number_input("Average Price ($)", min_value=0.1, max_value=5.0, value=float(selected_row['AveragePrice']), step=0.05)
            tot_vol = st.number_input("Total Volume", min_value=0.0, value=float(selected_row['Total Volume']), step=1000.0)
            p4046 = st.number_input("PLU 4046 Volume", min_value=0.0, value=float(selected_row['4046']))
            p4225 = st.number_input("PLU 4225 Volume", min_value=0.0, value=float(selected_row['4225']))

        with col_in2:
            p4770 = st.number_input("PLU 4770 Volume", min_value=0.0, value=float(selected_row['4770']))
            tot_bags = st.number_input("Total Bags", min_value=0.0, value=float(selected_row['Total Bags']))
            small_bags = st.number_input("Small Bags", min_value=0.0, value=float(selected_row['Small Bags']))
            large_bags = st.number_input("Large Bags", min_value=0.0, value=float(selected_row['Large Bags']))

        with col_in3:
            xlarge_bags = st.number_input("XLarge Bags", min_value=0.0, value=float(selected_row['XLarge Bags']))
            
            region_list = sorted(le_region.classes_.tolist())
            reg_idx = region_list.index(selected_row['region']) if selected_row['region'] in region_list else 0
            selected_region = st.selectbox("Region", options=region_list, index=reg_idx)
            
            years_list = sorted(df['year'].unique().tolist())
            year_val = int(selected_row['year'])
            yr_idx = years_list.index(year_val) if year_val in years_list else 0
            selected_year = st.selectbox("Year", options=years_list, index=yr_idx)
            
            selected_month = st.slider("Month", 1, 12, int(selected_row['Month']))
            selected_day = st.slider("Day", 1, 31, int(selected_row['Day']))

        model_choice = st.radio("Select Model for Prediction", ["AdaBoost Classifier", "PyTorch Neural Network (MLP)"], horizontal=True)

        if st.button("Run Prediction Engine 🚀", use_container_width=True):
            reg_code = le_region.transform([selected_region])[0]
            input_data = np.array([[
                avg_price, tot_vol, p4046, p4225, p4770, 
                tot_bags, small_bags, large_bags, xlarge_bags, 
                selected_year, selected_month, selected_day, reg_code
            ]])

            if model_choice == "AdaBoost Classifier":
                pred_class = ada_model.predict(input_data)[0]
                proba = ada_model.predict_proba(input_data)[0][pred_class]
            else:
                input_sc = scaler.transform(input_data)
                input_t = torch.tensor(input_sc, dtype=torch.float32)
                dl_model.eval()
                with torch.no_grad():
                    prob_val = dl_model(input_t).item()
                    pred_class = int(prob_val > 0.5)
                    proba = prob_val if pred_class == 1 else 1 - prob_val

            result_label = le_type.inverse_transform([pred_class])[0].upper()

            st.markdown("---")
            
            # Match Status
            is_correct = (result_label == actual_type)
            match_status = "✅ Prediction matches actual dataset label!" if is_correct else "⚠️ Prediction differs from actual dataset label."

            if result_label == "ORGANIC":
                st.markdown(f"""
                <div class="pred-box-organic">
                    <h2>CLASSIFICATION RESULT: {result_label} 🥑</h2>
                    <h3>Model Confidence: {proba * 100:.2f}%</h3>
                    <p style="font-size: 16px; margin-top: 10px;"><b>{match_status}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-box-conventional">
                    <h2>CLASSIFICATION RESULT: {result_label} 📦</h2>
                    <h3>Model Confidence: {proba * 100:.2f}%</h3>
                    <p style="font-size: 16px; margin-top: 10px;"><b>{match_status}</b></p>
                </div>
                """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading dataset or executing model pipeline: {str(e)}")
    st.info("Please ensure 'avocado.csv' is placed in the project directory or uploaded via the sidebar.")