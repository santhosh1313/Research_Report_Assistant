# app.py
"""
Streamlit UI for the Multi-Agent Research & Report Assistant.

This is a new, separate entry point — it does not touch or replace
main.py, which stays as-is for CLI use / the milestone-3 pipeline demo.
This file only imports and orchestrates the existing pieces:
  auth_db.py          -> user accounts + conversation metadata (SQLite)
  conversation_memory.py -> per-user chat history (ChromaDB)
  orchestrator.py      -> decides research-pipeline vs follow-up per message
"""
import os
import tempfile

import streamlit as st

import auth_db
import conversation_memory as convo_mem
import orchestrator

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎", layout="wide")

auth_db.init_db()

# =================================================
# SESSION STATE
# =================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.current_session_id = None

# A browser refresh wipes st.session_state (it's tied to the WebSocket
# connection, not the browser tab), but the URL survives. On every script
# run, if we're not "logged in" in session_state, check for a token in the
# URL and restore login from the sessions table if it's valid.
if st.session_state.user_id is None:
    _token = st.query_params.get("token")
    if _token:
        _session = auth_db.validate_session(_token)
        if _session:
            st.session_state.user_id = _session["user_id"]
            st.session_state.username = _session["username"]
            st.session_state.current_session_id = None


# =================================================
# LOGIN / REGISTER
# =================================================
def login_register_screen():
    st.title("Multi-Agent Research Assistant")
    st.caption("Log in or create an account to start a research conversation.")

    tab_login, tab_register = st.tabs(["Log In", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                try:
                    user_id = auth_db.authenticate_user(username, password)
                    token = auth_db.create_session(user_id)
                    st.query_params["token"] = token
                    st.session_state.user_id = user_id
                    st.session_state.username = username.strip()
                    st.session_state.current_session_id = None
                    st.rerun()
                except auth_db.AuthError as e:
                    st.error(str(e))

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Register", use_container_width=True)
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    try:
                        auth_db.register_user(new_username, new_password)
                        st.success("Account created — you can log in now from the Log In tab.")
                    except auth_db.AuthError as e:
                        st.error(str(e))


# =================================================
# SIDEBAR: conversation list
# =================================================
def render_sidebar():
    st.sidebar.title(f"{st.session_state.username}")

    if st.sidebar.button("➕ New Chat", use_container_width=True):
        session_id = auth_db.create_conversation(st.session_state.user_id, title="New Chat")
        st.session_state.current_session_id = session_id
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Your Conversations")

    conversations = auth_db.list_conversations(st.session_state.user_id)
    if not conversations:
        st.sidebar.caption("No conversations yet — start one above.")

    for conv in conversations:
        _render_conversation_row(conv)

    st.sidebar.markdown("---")
    if st.sidebar.button("Log Out", use_container_width=True):
        token = st.query_params.get("token")
        if token:
            auth_db.delete_session(token)
        st.query_params.clear()
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.current_session_id = None
        st.rerun()


def _render_conversation_row(conv):
    """One row in the sidebar: select-to-open + a '⋮' menu with
    Rename / Pin / Delete, matching the ChatGPT-style per-chat options."""
    session_id = conv["session_id"]
    user_id = st.session_state.user_id
    is_active = session_id == st.session_state.current_session_id
    pin_prefix = "📌 " if conv["pinned"] else ""
    label = f"{'● ' if is_active else ''}{pin_prefix}{conv['title'] or 'Untitled'}"

    with st.sidebar.container():
        col_select, col_menu = st.columns([5, 1])

        with col_select:
            if st.button(label, key=f"select_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()

        with col_menu:
            with st.popover("⋮", use_container_width=True):
                new_title = st.text_input(
                    "Rename", value=conv["title"], key=f"rename_input_{session_id}", label_visibility="collapsed"
                )
                if st.button("Save name", key=f"rename_save_{session_id}", use_container_width=True):
                    auth_db.rename_conversation(session_id, new_title.strip() or "Untitled")
                    st.rerun()

                pin_label = "📌 Unpin" if conv["pinned"] else "📌 Pin"
                if st.button(pin_label, key=f"pin_{session_id}", use_container_width=True):
                    auth_db.set_conversation_pinned(session_id, not conv["pinned"])
                    st.rerun()

                st.markdown("---")

                confirm_key = f"confirm_delete_{session_id}"
                if st.session_state.get(confirm_key):
                    st.warning("Delete this chat permanently?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes, delete", key=f"confirm_yes_{session_id}", use_container_width=True):
                            auth_db.delete_conversation(session_id)
                            convo_mem.delete_conversation_memory(user_id, session_id)
                            if st.session_state.current_session_id == session_id:
                                st.session_state.current_session_id = None
                            st.session_state[confirm_key] = False
                            st.rerun()
                    with c2:
                        if st.button("Cancel", key=f"confirm_no_{session_id}", use_container_width=True):
                            st.session_state[confirm_key] = False
                            st.rerun()
                else:
                    if st.button("🗑️ Delete", key=f"delete_{session_id}", use_container_width=True):
                        st.session_state[confirm_key] = True
                        st.rerun()


# =================================================
# MAIN: chat window
# =================================================
def _save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    paths = []
    for f in uploaded_files:
        path = os.path.join(temp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return paths


def render_chat():
    if not st.session_state.current_session_id:
        st.info("Start a new chat from the sidebar, or select an existing conversation.")
        return

    user_id = st.session_state.user_id
    session_id = st.session_state.current_session_id

    history = convo_mem.get_conversation_history(user_id, session_id)
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    prompt = st.chat_input(
        "Ask anything, or attach PDF(s) for document analysis...",
        accept_file="multiple",
        file_type=["pdf"],
    )

    file_paths = None
    display_text = None

    if prompt:
        if prompt.text:
            display_text = prompt.text
        if prompt.files:
            file_paths = _save_uploaded_files(prompt.files)

    if file_paths or display_text:
        with st.spinner("Working on it — this can take a little while for a fresh research run..."):
            try:
                reply = orchestrator.handle_message(
                    user_id, session_id, display_text or "", file_paths=file_paths
                )
            except orchestrator.OrchestratorError as e:
                reply = f"Something went wrong: {e}"

        auth_db.touch_conversation(session_id)

        # Auto-title a brand-new conversation from its first message/upload
        conv_list = auth_db.list_conversations(user_id)
        current = next((c for c in conv_list if c["session_id"] == session_id), None)
        if current and current["title"] == "New Chat":
            if display_text:
                new_title = display_text.strip()[:40]
            elif file_paths:
                new_title = os.path.basename(file_paths[0])[:40]
            else:
                new_title = "New Chat"
            auth_db.rename_conversation(session_id, new_title)

        st.rerun()


# =================================================
# APP ENTRY
# =================================================
if st.session_state.user_id is None:
    login_register_screen()
else:
    render_sidebar()
    st.title("Research Report Assistant")
    render_chat()