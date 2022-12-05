import wikipedia
import streamlit as st


@st.cache
def get_wiki_page(query: str) -> str:
    content = wikipedia.page(query).content
    return content.strip()[:2000]
