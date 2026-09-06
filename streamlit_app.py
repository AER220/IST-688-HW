import streamlit as st

# HW Manager - the main page that holds all my homework as separate pages.
st.set_page_config(page_title="HW Manager")

# register each homework page (the files live inside the HW folder)
hw1 = st.Page("HW/HW1.py", title="HW 1", icon="1️⃣")
hw2 = st.Page("HW/HW2.py", title="HW 2", icon="2️⃣", default=True)

# build the navigation and run whichever page is selected
pg = st.navigation([hw1, hw2])
pg.run()