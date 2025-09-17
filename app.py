import streamlit as st
from chatbot import search_context, generate_response, check_gpt_index_contents

st.set_page_config(page_title="🎥 Video Chatbot", layout="wide")

st.title("🎥 Video Analysis Chatbot")
st.markdown("Ask questions about processed videos. The bot uses **frames, audio, and GPT descriptions** as context.")

# Sidebar for video ID
video_id = st.sidebar.text_input("Video ID (optional)", value="")

# User query input
query = st.text_input("Enter your query:", placeholder="e.g., What is this video about?")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching across indexes..."):
            # Show some debug info
            st.subheader("🔎 GPT Index Contents (sample)")
            docs = check_gpt_index_contents()
            if docs:
                st.json(docs)

            # Search context
            context = search_context(query, video_id)

            st.subheader("📑 Retrieved Context")
            st.json(context)

            # Generate final response
            response = generate_response(query, context, video_id)
            st.subheader("💬 Chatbot Response")
            st.write(response)
# import streamlit as st
# import uuid
# import asyncio
# from pathlib import Path

# # Import pipeline + chatbot functions from your code
# from main import run_session
# from chatbot import search_context, generate_response

# st.set_page_config(page_title="Video Chatbot", layout="wide")

# st.sidebar.header("🎥 Video Options")

# # Choose between video upload or existing ID
# upload_option = st.sidebar.radio("Choose input method:", ["Upload Video", "Use Existing Video ID"])

# video_id = None

# if upload_option == "Upload Video":
#     uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
#     if uploaded_file is not None:
#         video_path = Path(f"./uploads/{uploaded_file.name}")
#         video_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(video_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.sidebar.success(f"Uploaded: {uploaded_file.name}")

#         # Generate fresh video_id
#         video_id = uuid.uuid4().hex
#         user_text = f"Process this video: {str(video_path)} with video_id: {video_id}"

#         # Run pipeline
#         with st.spinner("🚀 Running full pipeline... please wait"):
#             try:
#                 final_text, tool_payloads = asyncio.run(run_session(user_text))
#                 st.sidebar.success("✅ Video processed successfully!")
#                 st.sidebar.json(tool_payloads)
#             except Exception as e:
#                 st.sidebar.error(f"Pipeline error: {e}")

# elif upload_option == "Use Existing Video ID":
#     video_id = st.sidebar.text_input("Enter Video ID")
#     if video_id:
#         st.sidebar.success(f"Using video ID: {video_id}")

# # Main chatbot interface
# st.title("💬 Video Chatbot")

# if video_id:
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     # Display history
#     for role, text in st.session_state.chat_history:
#         with st.chat_message(role):
#             st.write(text)

#     # Input box
#     if query := st.chat_input("Ask about your video..."):
#         st.session_state.chat_history.append(("user", query))
#         with st.chat_message("user"):
#             st.write(query)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 context = search_context(query, video_id)
#                 response = generate_response(query, context, video_id)
#                 st.write(response)

#         st.session_state.chat_history.append(("assistant", response))
# else:
#     st.info("⬅️ Upload a video or enter an existing Video ID to start chatting.")