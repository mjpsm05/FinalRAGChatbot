import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import requests
import json
import re

# Hugging Face API Configuration
HUGGINGFACE_API_KEY = st.secrets["huggingface"]["api_key_2"]
HF_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# Pinecone API Configuration
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
index_name = "test"

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Define index name and specification
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=384,  # Dimension matches the embedding model output
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Access the index
index = pinecone_client.Index(index_name)

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Show title page with instructions only if not logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    # Sidebar instructions visible only on the title page
    st.sidebar.markdown("""
    # Family History RAG Chatbot Instructions

    Welcome to the Family History RAG Chatbot! This chatbot allows you to ask questions about a family's history, and it will respond based on the information stored in the database.

    ## How to Use:
    1. **Login**: Enter your username and password to access the chatbot.
    2. **Ask Questions**: Once logged in, type your question in the input box and the chatbot will respond.
    3. **Logout**: Click the 'Logout' button in the sidebar to log out and return to the login screen.

    ### Features:
    - Personalized responses based on your username.
    - AI-generated responses using the Llama-3.2 model and context from the Pinecone database.

    ## Troubleshooting:
    - If you don't get a response, try again later.
    - Ensure you're using the correct username and password.

    Created by Mazamesso Meba 
    """)

if "logged_in" in st.session_state and st.session_state.logged_in:
    st.title(f"Family History RAG Chatbot - {st.session_state.username}")
else:
    st.title("Family History RAG Chatbot")

# User Login and Authentication
def login():
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password", help="Enter your password.")
    
    if st.button("Login"):
        if username in credentials and credentials[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials, please try again.")

# Define credentials (Placeholder - Replace with secure authentication)
credentials = {
    "nathaniel": "password1",
    "kwame": "password2",
    "mazamesso": "@Lotovisa05"
}

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("You have logged out successfully.")
    st.rerun()

# Check if the user is logged in
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login()
else:
    username = st.session_state.username
    st.sidebar.write(f"Logged in as {username}")
    
    if st.sidebar.button("Logout"):
        logout()

    # Chatbot Page (Only visible after login)
    if username:
        user_query = st.text_input("Ask something:", placeholder="Enter your question...")

        if user_query:
            query_embedding = embedder.encode(user_query).tolist()

            # Query Pinecone for user-specific data
            filter_criteria = {"user": username} if username in credentials else {}
            results = index.query(vector=query_embedding, top_k=3, include_metadata=True, filter=filter_criteria)

            # Extract relevant context from documents
            context_list = [match["metadata"].get("text", "") for match in results["matches"] if "text" in match["metadata"]]

            if context_list:
                refined_context = " ".join(context_list[:2])  # Use first 2 relevant chunks
            else:
                st.write("No relevant information found in the document.")
                st.stop()

            # Construct prompt for API
            full_prompt = f"""
            You are a helpful assistant. Based on the context, answer the user's question.

            Context:
            {refined_context}

            Question: {user_query}

            Answer:
            """

            # Define headers
            headers = {
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json"
            }

            # Send request to Hugging Face API
            response = requests.post(
                HF_API_URL, 
                headers=headers, 
                json={
                    "inputs": full_prompt, 
                    "parameters": {
                        "max_new_tokens": 300, 
                        "temperature": 0.5  # Lower temperature for more controlled responses
                    }
                }
            )

            if response.status_code == 200:
                try:
                    ai_response = response.json()[0]["generated_text"]
                    print("Full AI Response: ", ai_response)

                    # Extract the answer using regex
                    answer_match = re.search(r"Answer:\s*(.*)", ai_response, re.DOTALL)
                    if answer_match:
                        answer = answer_match.group(1).strip()
                    else:
                        # Fallback method if regex fails
                        answer_start = ai_response.find("Answer:")
                        answer = ai_response[answer_start + len("Answer:"):].strip() if answer_start != -1 else "Error: Could not extract answer."

                    # Display the formatted answer
                    st.subheader("AI Response:")
                    st.write(answer)

                except (KeyError, IndexError):
                    st.write("Unexpected response format. Please try again.")
            else:
                st.write("Error generating response:", response.text)
