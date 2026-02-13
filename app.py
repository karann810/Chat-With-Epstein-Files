import streamlit as st
from rag import ask_rag


# ---------------- Page Config ----------------
st.set_page_config(page_title="ChatWith-EpsteinFiles", layout="wide")

# ---------------- CSS (Header + Bottom Input + Spacing) ----------------
st.markdown("""
<style>
.header {
    position: fixed;
    top: 10px;
    left: 0;
    right: 0;
    height: 90px;
    background: red;
    z-index: 9999;  /* SUPER IMPORTANT */
    display: flex;
    justify-content: center;
    align-items: flex-end;
    font-size: 22px;
    font-weight: bold;
    border-bottom: 1px solid #ddd;
}

/* push content below */
.block-container {
    padding-top: 90px;
}
</style>

<div class="header">ChatWith-EpsteinFiles</div>
""", unsafe_allow_html=True)



# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Chat Area ----------------
st.markdown('<div class="main">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='text-align:right; margin:10px;'>
                <span style='background-color:#DCF8C6; padding:10px 15px;color:black; border-radius:15px; display:inline-block; max-width:60%;'>
                    {msg["content"]}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='text-align:left; margin:10px;'>
                <span style='background-color:#F1F0F0;color:black; padding:10px 15px; border-radius:15px; display:inline-block; max-width:60%;'>
                    {msg["content"]}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Input ----------------
prompt = st.chat_input("Ask something...")


# ---------------- On Send ----------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    # retrieve documents
    # docs = retriever.invoke(prompt)

    # combine text
    # context = "\n\n".join([d.page_content for d in docs])

    # # final prompt to LLM
    # final_prompt = f"""
    # Answer the question using the context below.

    # Context:
    # {context}

    # Question:
    # {prompt}
    # """

    response = ask_rag(prompt)


    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()
