import streamlit as st
import joblib
import pandas as pd
import plotly.express as px


scaler = joblib.load("pkl/scaler.pkl")
model = joblib.load("pkl/best_model_ANN.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)
st.markdown("""
    <h1 style='text-align:center; color:#C2185B;'>Breast Cancer Prediction App - Powered by ANN Model</h1>
    <p style='text-align:center; font-size:17px;'> Upload your medical CSV file to predict benign or malignant cases.</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div.stButton > button {
    background-color: #6c63ff;
    color: white;
    height: 2em;
    width: 100%;
    border-radius: 10px;
    font-size: 16px;
}
    div[data-testid="stAppViewContainer"] {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
div.stDownloadButton> button {
    background-color: #6c63ff;
    color: white;
    height: 2em;
    width: 100%;
    border-radius: 10px;
    font-size: 16px;
}
    div[data-testid="stAppViewContainer"] {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)


st.write("""This application analyzes breast cancer diagnostic data using a trained Artificial Neural Network (ANN) model.
Simply upload your medical CSV file containing the 30 standard features extracted from cell nuclei measurements.
The system preprocesses your data using the same scaling method used during training, feeds it into the model, and predicts whether each case is Benign or Malignant.
You will also see statistical charts that help you better understand the distribution of the predictions.""")

uploaded_file = st.file_uploader("📁 Please Upload your CSV file (must contain 30 features)", type=["csv"])

if st.button("Predict"):
 if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview(first 5 rows)")
    st.write(df.head(5))

    try:
        scaled = scaler.transform(df)
        pred = model.predict(scaled)

        df["prediction"] = ["Maligant" if x==1 else "Benign" for x in pred]
        maligant_count = (df["prediction"] == 'Maligant').sum()
        benign_count = (df["prediction"] == 'Benign').sum()

        st.subheader("Prediction Results🧾")
        col1, col2, col3, col4 = st.columns([1,2,1,2]) 
        with col1:
          st.markdown("""
                    <div style="background-color:#ffcccc; padding:1px; border-radius:20px;">
                    <h3 style="color:#d10202;"> Maligant Cases:</h3>
                </div>
                """, unsafe_allow_html=True)
        with col2:
         st.subheader(maligant_count)
        with col3:
          st.markdown("""
                    <div style="background-color:#ccffcc; padding:1px; border-radius:20px;">
                    <h3 style="color:#2e7d32;"> Benign Cases:</h3>
                </div>
                """, unsafe_allow_html=True)
        with col4:
         st.subheader(benign_count)

        st.write("")
        st.write(df)

        tab1, tab2 = st.tabs(["Prediction Distribution (Pie Chart)", "Benign vs Malignant Counts (bar chart)"])

        with tab1:
           fig = px.pie(df,
                        names='prediction',
                        title="Benign vs Malignant — Overall Distribution",
                        width=800, height=600 )
           
           fig.update_traces(textinfo='percent+label')
           st.plotly_chart(fig)

        color_map = {
            "Malignant": "#ff4d4d",  
            "Benign": "#66cc66"     
        } 
        with tab2:
           fig = px.bar(df,
                        x= 'prediction',
                        height=600,
                        title="Count of Predicted Benign and Malignant Cases",
                        color='prediction',
                        color_discrete_map=color_map
                        )
           fig.update_traces(width=0.4)
           fig.update_xaxes('')
           st.plotly_chart(fig)
  
        st.subheader("Download results")
        csv = df.to_csv(index=False)
        st.download_button(
        label="Download",
        data= csv,
        file_name="results.csv",
        mime="text/csv"
        )
           
    except Exception as e:
        st.write("Error! Make sure your CSV contains correct 30 features.")
        st.error(e)





