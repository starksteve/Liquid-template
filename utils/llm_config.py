import os
from dotenv import load_dotenv

# Load from .env file if present (local dev).
# On Streamlit Cloud, secrets are injected automatically via st.secrets.
load_dotenv()


def get_llm_client():
    """
    Initialize and return a Groq LLM client using langchain-groq.

    Required environment variable:
        GROQ_API_KEY  — get a free key at https://console.groq.com

    Optional environment variable:
        GROQ_MODEL    — defaults to "llama-3.3-70b-versatile"
                        Other free options: "mixtral-8x7b-32768", "gemma2-9b-it"

    For Streamlit Cloud deployment, set these in:
        App Settings → Secrets (TOML format)
    """
    # Support Streamlit Cloud secrets as fallback
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
        model = st.secrets.get("GROQ_MODEL") or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "• Local: add GROQ_API_KEY=your_key to your .env file\n"
            "• Streamlit Cloud: add it under App Settings → Secrets\n"
            "• Get a free key at https://console.groq.com"
        )

    from langchain_groq import ChatGroq

    print(f"DEBUG: Initializing Groq client | model={model}")

    client = ChatGroq(
        api_key=api_key,
        model_name=model,
        temperature=0.7,
    )

    return client