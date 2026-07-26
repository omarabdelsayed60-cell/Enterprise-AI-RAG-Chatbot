"""
app.py
------
Streamlit Conversational Web Interface for Corporate Policy Assistant.

Features:
- Streamlit components.v1.html JS component rendering the entire AI message box with embedded copy button.
- Copy button (🗐) placed INSIDE the text box at the top right.
- Instant '✓ Copied!' green feedback notification directly on the button.
- WhatsApp-style user layout (👤 You on right in green, 🤖 AI Assistant on left in slate).
- Non-clipped Donut/Pie Chart status dashboard.
- Zero technical developer jargon.
"""

import streamlit as st
import pandas as pd
import altair as alt
import json

# Import project modules
from config import GEMINI_API_KEY, DB_SERVER, DB_NAME
from logger import logger
import database
import vector_store
import rag_pipeline

# ---------------------------------------------------------
# Page Configuration & WhatsApp-Style Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Corporate Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Styling
st.markdown("""
<style>
    /* Prevent header cutoff - ample top padding */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* White Main Title Header */
    .app-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-top: 0.2rem;
        margin-bottom: 0.1rem;
    }
    
    .app-subtitle {
        font-size: 0.98rem;
        color: #94A3B8 !important;
        margin-bottom: 1.2rem;
    }
    
    /* Executive Button Styling */
    .stButton > button {
        background-color: #1E3A8A !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.4rem 0.8rem !important;
        font-size: 0.88rem !important;
    }
    
    .stButton > button:hover {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
    }

    /* WhatsApp-Style User Chat Bubble Formatting (Right Aligned) */
    .chat-bubble-user {
        background-color: #065F46;
        color: #FFFFFF;
        padding: 0.75rem 1.1rem;
        border-radius: 16px 16px 2px 16px;
        margin-left: auto;
        margin-right: 0;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        max-width: 75%;
        width: fit-content;
        text-align: right;
        font-size: 0.98rem;
        line-height: 1.5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .chat-label-user {
        text-align: right;
        font-size: 0.88rem;
        color: #34D399;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #334155;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        background-color: #065F46;
        color: #A7F3D0;
        font-weight: 600;
        border-radius: 18px;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session-Isolated Vector Store Initialization
# ---------------------------------------------------------
if "vector_store" not in st.session_state:
    try:
        st.session_state["vector_store"] = vector_store.load_faiss_vector_store(api_key=GEMINI_API_KEY)
    except Exception as err:
        logger.error(f"Error initializing session FAISS vector store: {err}")
        st.session_state["vector_store"] = None

def render_assistant_bubble(text: str, idx: str):
    """
    Renders the AI Assistant chat bubble with the 🗐 copy button INSIDE the text box.
    Uses streamlit.components.v1.html to execute working JavaScript copy.
    """
    escaped_text = json.dumps(text)
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .assistant-container {{
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            padding-bottom: 4px;
        }}
        .label {{
            font-size: 0.88rem;
            color: #60A5FA;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .bubble {{
            background-color: #1E293B;
            color: #F8FAFC;
            padding: 0.8rem 1.1rem;
            border-radius: 14px 14px 14px 2px;
            max-width: 85%;
            font-size: 0.98rem;
            line-height: 1.5;
            border: 1px solid #334155;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            box-sizing: border-box;
        }}
        .copy-btn {{
            background: transparent;
            border: none;
            color: #94A3B8;
            cursor: pointer;
            font-size: 13px;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            transition: all 0.2s ease;
            user-select: none;
            flex-shrink: 0;
        }}
        .copy-btn:hover {{
            color: #FFFFFF;
            background-color: #334155;
        }}
    </style>
    </head>
    <body>
        <div class="assistant-container">
            <div class="label">🤖 AI Assistant</div>
            <div class="bubble">
                <span id="text_{idx}">{text}</span>
                <button id="btn_{idx}" class="copy-btn" onclick="copyText_{idx}()" title="Copy text">🗐</button>
            </div>
        </div>
        <script>
            function copyText_{idx}() {{
                var txt = {escaped_text};
                function onSuccess() {{
                    var btn = document.getElementById('btn_{idx}');
                    if (btn) {{
                        btn.innerHTML = '✓ Copied!';
                        btn.style.color = '#34D399';
                        setTimeout(function() {{
                            btn.innerHTML = '🗐';
                            btn.style.color = '#94A3B8';
                        }}, 2000);
                    }}
                }}
                
                if (navigator.clipboard && window.isSecureContext) {{
                    navigator.clipboard.writeText(txt).then(onSuccess).catch(function() {{
                        fallbackCopy();
                    }});
                }} else {{
                    fallbackCopy();
                }}
                
                function fallbackCopy() {{
                    var ta = document.createElement("textarea");
                    ta.value = txt;
                    ta.style.position = "fixed";
                    ta.style.left = "-999999px";
                    ta.style.top = "-999999px";
                    document.body.appendChild(ta);
                    ta.focus();
                    ta.select();
                    try {{
                        document.execCommand('copy');
                        onSuccess();
                    }} catch(e) {{
                        console.error('Copy failed', e);
                    }}
                    document.body.removeChild(ta);
                }}
            }}
        </script>
    </body>
    </html>
    """
    lines = len(text) // 55 + 1
    calc_height = max(80, 50 + lines * 24)
    st.iframe(html_code, height=calc_height)

# ---------------------------------------------------------
# Sidebar Controls & Live Database Status
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/briefcase.png", width=52)
st.sidebar.title("Company Assistant")

# Fetch database policies for realtime metrics
try:
    policies_list = database.get_all_policies()
    st.sidebar.markdown("<div class='status-badge'>🟢 Knowledge Base Active</div>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error("Database connection check failed.")
    policies_list = []

st.sidebar.markdown("---")

# Refresh Knowledge Base button
if st.sidebar.button("🔄 Refresh Knowledge Base", width="stretch"):
    if not GEMINI_API_KEY:
        st.sidebar.error("Gemini API key is missing in .env!")
    else:
        with st.spinner("Updating policy knowledge base..."):
            try:
                num_chunks, synced_topics = vector_store.sync_vector_store(api_key=GEMINI_API_KEY)
                # Clear session state vector store to reload fresh disk index
                st.session_state["vector_store"] = vector_store.load_faiss_vector_store(api_key=GEMINI_API_KEY, force_rebuild=True)
                
                if num_chunks > 0:
                    st.sidebar.success(f"Knowledge base updated! Synced {len(synced_topics)} topics.")
                    st.rerun()
                else:
                    st.sidebar.info("Knowledge base is already up to date.")
            except Exception as sync_err:
                st.sidebar.error("Sync error occurred.")
                logger.error(f"Sync error: {sync_err}")

# Clear Chat Conversation
if st.sidebar.button("🗑️ Clear Conversation", width="stretch"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Corporate Policy Assistant. How can I help you today with company policies, benefits, or workplace guidelines?"}
    ]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("© Corporate Human Resources & Operations")

# ---------------------------------------------------------
# Main Header Section
# ---------------------------------------------------------
st.markdown("<div class='app-header'>🏢 Corporate Policy Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>Your intelligent assistant for company policies, benefits, and workplace guidelines.</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Donut / Pie Chart Dashboard Section
# ---------------------------------------------------------
total_count = len(policies_list)
synced_count = sum(1 for p in policies_list if p.get("Processed") == 1)
pending_count = total_count - synced_count

col_title, col_chart = st.columns([1, 2])

with col_title:
    st.markdown("##### 📊 Knowledge Base Status")
    st.write(f"• **Total Policies**: {total_count}")
    st.write(f"• **Processed & Synced**: {synced_count}")
    st.write(f"• **Pending Sync**: {pending_count}")

with col_chart:
    chart_df = pd.DataFrame({
        "Status": ["Processed & Synced", "Pending Sync"],
        "Count": [synced_count, pending_count]
    })
    
    pie_chart = alt.Chart(chart_df).mark_arc(innerRadius=25, outerRadius=50, cornerRadius=3).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(
            field="Status",
            type="nominal",
            scale=alt.Scale(domain=["Processed & Synced", "Pending Sync"], range=["#10B981", "#F59E0B"]),
            legend=alt.Legend(title="", orient="right")
        ),
        tooltip=["Status", "Count"]
    ).properties(width=280, height=140)
    
    st.altair_chart(pie_chart, width="stretch")

st.markdown("---")

# ---------------------------------------------------------
# WhatsApp-Style Chat Interface
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Corporate Policy Assistant. How can I help you today with company policies, benefits, or workplace guidelines?"}
    ]

# Render chat messages
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; flex-direction: column; align-items: flex-end;'>
                <div class='chat-label-user'>👤 You</div>
                <div class='chat-bubble-user'>{msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Render AI Assistant message box with embedded copy icon INSIDE the text box
        render_assistant_bubble(msg["content"], str(idx))

# User Chat Input Box
if user_prompt := st.chat_input("Ask any question about company policies..."):
    # 1. Append User Message to session history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 2. Immediately render User Chat Bubble on screen
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: flex-end;'>
            <div class='chat-label-user'>👤 You</div>
            <div class='chat-bubble-user'>{user_prompt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Read current API key dynamically from environment
    import os
    from dotenv import load_dotenv
    from config import env_path
    load_dotenv(dotenv_path=env_path, override=True)
    current_api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)

    # 4. Generate Assistant Response
    with st.spinner("Searching policies..."):
        try:
            result = rag_pipeline.ask_rag_assistant(
                question=user_prompt,
                vector_store_instance=st.session_state.get("vector_store"),
                api_key=current_api_key
            )
            assistant_response = result["answer"]
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.rerun()
            
        except Exception as err:
            error_msg = "I encountered an error retrieving policy information. Please try again or contact HR."
            logger.error(f"Chat error: {err}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
