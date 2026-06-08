"""
auth.py — thin wrapper around st.session_state for login state.

Call init_auth() at the top of every page.
Use is_logged_in(), current_user() and logout() everywhere.
"""

import streamlit as st
from utils.user_store import login_user, register_user


def init_auth():
    """Ensure session state keys exist."""
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None   # None or {"userId": int, "username": str}


def is_logged_in() -> bool:
    init_auth()
    return st.session_state["auth_user"] is not None


def current_user() -> dict | None:
    init_auth()
    return st.session_state["auth_user"]


def current_user_id() -> int | None:
    u = current_user()
    return u["userId"] if u else None


def current_username() -> str | None:
    u = current_user()
    return u["username"] if u else None


def logout():
    st.session_state["auth_user"] = None


def show_login_banner():
    """
    Render a compact login / register form inside an expander.
    Call this at the top of any page that benefits from login but
    should still be usable without it.
    """
    init_auth()
    if is_logged_in():
        u = current_user()
        col1, col2 = st.columns([6, 1])
        col1.caption(f"👤 Logged in as **{u['username']}**  (ID {u['userId']})")
        if col2.button("Log out", key="_global_logout"):
            logout()
            st.rerun()
        return

    with st.expander("🔐 Log in for personalised predictions", expanded=False):
        _login_form(compact=True)


def _login_form(compact=False):
    """
    Full login + register form. compact=True uses a smaller layout.
    """
    tab_login, tab_reg = st.tabs(["Log in", "Create account"])

    with tab_login:
        uname = st.text_input("Username", key="_login_uname")
        pwd   = st.text_input("Password", type="password", key="_login_pwd")
        if st.button("Log in", key="_login_btn", type="primary"):
            ok, user = login_user(uname, pwd)
            if ok:
                st.session_state["auth_user"] = user
                st.success(f"Welcome back, {user['username']}!")
                st.rerun()
            else:
                st.error("Incorrect username or password.")

    with tab_reg:
        new_uname = st.text_input("Choose a username", key="_reg_uname")
        new_pwd   = st.text_input("Choose a password", type="password", key="_reg_pwd")
        new_pwd2  = st.text_input("Confirm password",  type="password", key="_reg_pwd2")
        if st.button("Create account", key="_reg_btn", type="primary"):
            if new_pwd != new_pwd2:
                st.error("Passwords do not match.")
            else:
                ok, msg = register_user(new_uname, new_pwd)
                if ok:
                    st.success(msg + " You can now log in.")
                else:
                    st.error(msg)
