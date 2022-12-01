import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Machine Learning",
    page_icon="👩‍💻",
)
st.title("Machine Learning 👩‍💻")

st.write("Giảng viên hướng dẫn: Trần Tiến Đức")

st.write("Sinh viên thực hiện:")
df = pd.DataFrame(
    np.array([["Phạm Thị Nguyệt Quế", 20110706],["Nguyễn Thanh Phương Thảo", 20110197]] ),
    columns=("Họ tên", "MSSV"))

st.table(df)

image = Image.open('pages/AnhML.jpg')
st.image(image)

