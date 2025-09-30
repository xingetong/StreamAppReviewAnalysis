import streamlit as st

st.write("Testing token access...")

try:
    token = st.secrets["HUGGINGFACE_TOKEN"]
    st.success(f"✅ Token found! Starts with: {token[:10]}...")
except KeyError:
    st.error("❌ HUGGINGFACE_TOKEN not found in secrets")
    st.write("Available keys:", list(st.secrets.keys()))
except FileNotFoundError:
    st.error("❌ secrets.toml file not found")