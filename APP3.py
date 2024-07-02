#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 使用相对路径加载模型
model = joblib.load('./svm_model1.pkl')

# 使用相对路径加载数据
csv_path = './2910.csv'
df = pd.read_csv(csv_path)

# 特征和标签
X = df.drop('G', axis=1)
y = df['G']

# Streamlit界面设计
st.title("钢铁工人高血压风险评估网页工具")

st.sidebar.header("用户输入参数")

# 定义特征字典和用户输入函数
features = {
    "AGE": [1, 2, 3, 4],
    "Education": [1, 2, 3],
    "SMOKE": [1, 2, 3],
    "Salt intake": [1, 2, 3],
    "Cooling product": [0, 1],
    "CO exposure": [0, 1],
    "Family history of hypertension": [0, 1],
    "HGB": [0, 1],
    "FBG": [0, 1],
    "TG": [0, 1]
}

def user_input_features():
    user_input = {}
    user_input["AGE"] = st.sidebar.selectbox("AGE", options=features["AGE"], format_func=lambda x: f"{x}: 18-30岁" if x == 1 else f"{x}: 30-40岁" if x == 2 else f"{x}: 40-50岁" if x == 3 else f"{x}: 50岁及以上")
    user_input["Education"] = st.sidebar.selectbox("Education", options=features["Education"], format_func=lambda x: f"{x}: 初中及以下" if x == 1 else f"{x}: 高中或中专" if x == 2 else f"{x}: 大专及以上")
    user_input["SMOKE"] = st.sidebar.selectbox("SMOKE", options=features["SMOKE"], format_func=lambda x: f"{x}: 从不吸烟" if x == 1 else f"{x}: 曾经吸烟现在不吸" if x == 2 else f"{x}: 现在吸烟")
    user_input["Salt intake"] = st.sidebar.selectbox("Salt intake", options=features["Salt intake"], format_func=lambda x: f"{x}: 偏淡" if x == 1 else f"{x}: 正常" if x == 2 else f"{x}: 偏咸")
    user_input["Cooling product"] = st.sidebar.selectbox("Cooling product", options=features["Cooling product"], format_func=lambda x: f"{x}: 没有使用" if x == 0 else f"{x}: 有使用")
    user_input["CO exposure"] = st.sidebar.selectbox("CO exposure", options=features["CO exposure"], format_func=lambda x: f"{x}: 没有暴露" if x == 0 else f"{x}: 暴露")
    user_input["Family history of hypertension"] = st.sidebar.selectbox("Family history of hypertension", options=features["Family history of hypertension"], format_func=lambda x: f"{x}: 无" if x == 0 else f"{x}: 有")
    user_input["HGB"] = st.sidebar.selectbox("HGB", options=features["HGB"], format_func=lambda x: f"{x}: 正常" if x == 0 else f"{x}: 不正常")
    user_input["FBG"] = st.sidebar.selectbox("FBG", options=features["FBG"], format_func=lambda x: f"{x}: 正常" if x == 0 else f"{x}: 不正常")
    user_input["TG"] = st.sidebar.selectbox("TG", options=features["TG"], format_func=lambda x: f"{x}: 正常" if x == 0 else f"{x}: 不正常")

    return pd.DataFrame(user_input, index=[0])

input_df = user_input_features()


# 将用户输入与数据集列名对齐
input_df = input_df[X.columns]

# 模型预测
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# 显示预测结果和概率
st.subheader("预测结果")
hypertension_status = np.array(['无高血压', '有高血压'])
st.write(hypertension_status[prediction][0])

st.subheader("预测概率")
st.write(f"无高血压: {prediction_proba[0][0]:.2f}")
st.write(f"有高血压: {prediction_proba[0][1]:.2f}")

# 显示用户输入的特征
st.subheader("用户输入参数")
st.write(input_df)

