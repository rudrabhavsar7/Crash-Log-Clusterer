import streamlit as st
import pandas as pd

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})

@st.experimental_dialog("My Dialog")
def show_dialog(row_idx):
    st.write(f"You selected row {row_idx}")
    if st.button("Close"):
        st.rerun()

st.write("Click a row to open the dialog")
event = st.dataframe(st.session_state.df, on_select="rerun", selection_mode="single-row", key="test_df")

if event.selection.rows:
    row_idx = event.selection.rows[0]
    if st.session_state.get("last_row") != row_idx:
        st.session_state["last_row"] = row_idx
        show_dialog(row_idx)

if st.button("Other button"):
    st.write("Clicked other button")
