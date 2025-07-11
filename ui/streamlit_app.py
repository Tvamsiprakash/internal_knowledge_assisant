import streamlit as st
import requests

# Page config
st.set_page_config(page_title="RAG Assistant", layout="centered")
st.markdown("## 💡 Internal Knowledge Assistant")
st.caption("Powered by **RAG + FAISS + Gemma 1B + FastAPI**")

# File upload
st.markdown("### 📤 Upload `.txt` or `.pdf`")
uploaded_file = st.file_uploader("Upload file", type=["txt", "pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    if st.button("Upload"):
        with st.spinner("Uploading and indexing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/upload",
                    files={"file": (uploaded_file.name, uploaded_file.read())}
                )
                if response.status_code == 200:
                    st.success(f"✅ `{uploaded_file.name}` uploaded and indexed!")
                else:
                    st.error(f"❌ Upload failed: {response.status_code}")
                    st.json(response.json())
            except Exception as e:
                st.error(f"🚨 Upload error: {e}")

st.divider()
st.markdown("### 💬 Ask a Question")
query = st.text_area("Question:", height=100)

if st.button("Ask") and query.strip():
    with st.spinner("💭 Thinking..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": query}
            )
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer")
                sources = result.get("sources", [])

                st.markdown("#### ✅ Answer")
                st.success(answer if answer else "No answer returned.")

                if sources:
                    st.markdown("#### 📎 Sources")
                    for src in sources:
                        st.markdown(f"- `{src['filename']}`, chunk `{src['chunk']}`")
                else:
                    st.info("No source metadata.")
            else:
                st.error(f"❌ Server error: {response.status_code}")
                st.json(response.json())
        except Exception as e:
            st.error(f"🚨 Request failed: {e}")
