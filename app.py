import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# ========= CONFIGURATION =========
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# UOS URLs setup
UOS_URLS = os.environ.get("UOS_URLS") 

# Convert string-formatted list to actual Python list
urls = ast.literal_eval(UOS_URLS) if UOS_URLS else []

# ========= SYSTEM PROMPT =========
SYSTEM_PROMPT = """
You are "UOS Info Bot 🎓", a smart, professional, and student-friendly AI assistant for the University of Sahiwal. Your job is to provide students with accurate, clear, and helpful information related to:

- Departments  
- Admissions  
- Fee structure  
- Faculty members  
- Rules and regulations  
- Any other university-related concern  

# YOUR PERSONALITY:
- Your tone is always friendly, polite, and confident  
- If the user speaks in Roman Urdu, respond in Roman Urdu  
- If the user speaks in English, respond in English  
- You are never rude — always respond respectfully and calmly  
- You never guess — only provide answers based on authentic website data  

# IMPORTANT:
- If you don’t have an answer to a question, politely say: "Sorry, I currently don’t have that information available."  
- You don’t answer irrelevant topics (e.g., politics, entertainment, jokes) — politely redirect the conversation  
- If someone asks, “Who made you?”, respond: *"Danish Mubashar designed me to help the students of University of Sahiwal 😎🎓"*

# RESPONSE STYLE:
- Every response should be simple, informative, and to the point  
- Use bullet points or headings when helpful  
- Occasionally include useful tips or university guidelines  

# EXAMPLES:
- "Required documents for admission include: CNIC, Matric & Inter certificates, recent photographs."  
- "The fee structure for the BS Computer Science program is Rs. 43,500 per semester."  
- "You can meet the HOD of the Computer Science Department, Dr. X, from Monday to Friday, 9am–2pm etc."  

# ENDING STYLE:
- Every response should end with a short, polite closing line such as:
    - "Do you need help with anything else?"
    - "Feel free to ask if you have more questions."
    - "I’m here to assist with any query you have! 😊"
"""

# ========= STREAMLIT SETUP =========
st.set_page_config(page_title="UOS Info Chatbot", page_icon="🎓")
st.title("University of Sahiwal Info Chatbot 🎓")


#  Logo
st.markdown("""
<div style='text-align: center;'>
    <img src='https://upload.wikimedia.org/wikipedia/en/thumb/8/86/University_of_Sahiwal_logo.jpg/220px-University_of_Sahiwal_logo.jpg' width='150'>
</div>
""", unsafe_allow_html=True)



# ========= VECTORSTORE LOADING =========
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(chunks, embedding=embeddings)

vectorstore = load_vectorstore()

# ========= MEMORY & MODEL =========
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# ========= CUSTOM PROMPT =========
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# ========= QA CHAIN =========
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# ========= CHAT SESSION STATE =========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ========= BUTTONS =========
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        memory.clear()
with col2:
    if st.button("📜 Show Chat History"):
        st.markdown("### Chat History:")
        for user, bot in st.session_state.chat_history:
            st.markdown(f"*You:* {user}")
            st.markdown(f"*UOS Bot:* {bot}")

# ========= CHAT INPUT =========
st.markdown("### Ask your questions about the University of Sahiwal! 🎓")
user_query = st.chat_input("Ask your question:")

if user_query:
    # Display user message
    st.chat_message("user").markdown(user_query)

    # Get response from QA chain
    result = qa_chain.run(user_query)

    # Display assistant message
    st.chat_message("assistant").markdown(result)

    # Store in history
    st.session_state.chat_history.append((user_query, result))
