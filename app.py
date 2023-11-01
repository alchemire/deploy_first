import streamlit as st
import pandas as pd
import requests
import json

st.title("温故知新 feat.青空文庫")

input_text = st.text_input("つれづれなるままに、日暮らし硯に向かひて、心にうつりゆくよしなしごとを、そこはかとなく書きつくれば、あやしうこそものぐるほしけれ。")
if st.button("現代語訳"):
    st.text("手持ちぶさたなのにまかせて、一日中硯に向かって、心に浮かんだり消えたりしてうつっていくつまらないことを、とりとめもなく書きつけると、妙に正気を失った気分になる。")

data = {
    "input_text" : input_text
}
data_json = json.dumps(data).encode("utf-8")

if st.button("click"):
    # data_df = pd.DataFrame(data, index=["input_data"])
    # st.write(data_df)

    response = requests.post("http://localhost:8000/embedding", data=data_json)
    response_data = response.json()
    st.write(response_data)