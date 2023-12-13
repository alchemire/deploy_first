import streamlit as st
import pandas as pd
import requests
import json

st.title("温故知新 feat.青空文庫")

input_text = st.text_input("青空文庫に掲載されている、1000文字以下の作品2995点から、コサイン類似度が高い文章TOP5を返します。先人たちに聞き役になってもらい、一緒に内省を深めましょう。")
st.caption(f'文字数 {len(input_text)} 文字')
if st.button("サンプル：徒然草(現代語訳)"):
    st.text("することもなく手持ちぶさたなのにまかせて、一日中、硯に向かって、心の中に浮かんでは消えていくたわいもないことを、とりとめもなく書きつけていると、思わず熱中して不思議と、気が変になる。")

data = {
    "input_text" : input_text
}
data_json = json.dumps(data).encode("utf-8")

if st.button("壁打ち"):
    # data_df = pd.DataFrame(data, index=["input_data"])
    # st.write(data_df)

    response = requests.post("http://localhost:8000/embedding", data=data_json)
    response_data = response.json()
    st.subheader("類似度の高い文章TOP5")
    with st.expander(f"1_{df.iloc[0, 2]}さんの連想"):
        st.write(df.iloc[0, 1])
    with st.expander(f"2_{df.iloc[1, 2]}さんの連想"):
        st.write(df.iloc[1, 1])
    with st.expander(f"3_{df.iloc[2, 2]}さんの連想"):
        st.write(df.iloc[2, 1])
    with st.expander(f"4_{df.iloc[3, 2]}さんの連想"):
        st.write(df.iloc[3, 1])
    with st.expander(f"5_{df.iloc[4, 2]}さんの連想"):
        st.write(df.iloc[4, 1])
