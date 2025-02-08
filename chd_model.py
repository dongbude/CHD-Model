import os
import pickle
import pandas as pd
import streamlit as st
import shap
import numpy as np

# ======================== 加载模型 =========================
def load_model(model_path):
    """加载冠心病预测模型和归一化器"""
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    return data['model'], data['scaler']

current_dir = os.getcwd()
model_path = os.path.join(current_dir, 'KNN_model.pkl')

if not os.path.exists(model_path):
    st.error(f"模型文件 {model_path} 不存在！")
    st.stop()
else:
    model, scaler = load_model(model_path)
    st.success("模型加载成功！")

# ======================== 加载 SHAP 背景数据 =========================
csv_path = os.path.join(current_dir, 'background_train_data200.csv')
background_data = pd.read_csv(csv_path)

# ======================== Streamlit 页面布局 =========================
st.title("Calculation Tool for Predicting Coronary Heart Disease Risk")
st.markdown("**使用血浆炎症标志物和常规临床数据预测冠心病风险的计算工具**")

st.sidebar.markdown("<h3 style='text-align: center;'>河南大学第一附属医院心内科</h3>", unsafe_allow_html=True)

# 侧边栏输入特征
st.sidebar.header("Input feature data 输入特征数据")

age = st.sidebar.slider("Age (years)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
hypertension_history = st.sidebar.selectbox("Hypertension History (0: No, 1: Yes)", options=[0, 1])
t2dm = st.sidebar.selectbox("T2DM (0: No, 1: Yes)", options=[0, 1])
alcoholHistory = st.sidebar.selectbox("AlcoholHistory (0: No, 1: Yes)", options=[0, 1])
bmi = st.sidebar.slider("BMI (kg/m²)", min_value=15.0, max_value=40.0, value=25.0, step=0.1)
s100a8a9 = st.sidebar.slider("S100A8A9 (μg/L)", min_value=0.0, max_value=30.0, value=5.0, step=0.1)
SCr = st.sidebar.slider("SCr (μmmol/L)", min_value=30.0, max_value=707.0, value=50.0, step=0.1)
ast = st.sidebar.slider("AST (U/L)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)

input_data = pd.DataFrame({
    'HypertensionHistory': [hypertension_history],
    'T2DM': [t2dm],
    'Age': [age],
    'S100A8A9': [s100a8a9],
    'BMI': [bmi],
    'AlcoholHistory': [alcoholHistory],
    'AST': [ast],
    'SCr': [SCr]
})

input_data_scaled = scaler.transform(input_data)
feature_names = input_data.columns.tolist()

# ======================== 预测与可视化 =========================
if st.button("Predict 预测"):
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    
    predicted_class = prediction[0]
    class_labels = ["低风险 (0)", "高风险 (1)"]
    probability = prediction_proba[0][predicted_class]

    # 显示预测结果
    st.subheader("预测结果")
    st.markdown(f"**预测类别**: {class_labels[predicted_class]}")
    st.markdown(f"**当前类别概率**: {probability:.2%}")  # 格式化为百分比

    # 计算 SHAP 值
    explainer = shap.KernelExplainer(model.predict_proba, background_data)
    shap_values = explainer.shap_values(input_data_scaled)
    
    # 动态选择 SHAP 值
    shap_values_for_class = shap_values[0, :, predicted_class]
    baseline_value = explainer.expected_value[predicted_class]

    # 生成 SHAP 图
    st.subheader("当前类别特征影响分析 (SHAP)")
    force_plot = shap.force_plot(
        baseline_value,
        shap_values_for_class,
        input_data_scaled,
        feature_names=feature_names,
        figsize=(40,10),
        show=False,
        matplotlib=False
    )
    st.components.v1.html(shap.getjs() + force_plot.html(), height=400)
