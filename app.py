import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# ========= CONFIGURATION =========
os.environ["GOOGLE_API_KEY"] = "AIzaSyDXc8_hij-YGHjhV6omjX32tgT2q4Ymm-A"  # Replace with your actual key

# ========= SYSTEM PROMPT =========
# ========= SYSTEM PROMPT =========
SYSTEM_PROMPT = """
Tum ho "UOS Info Bot 🎓", University of Sahiwal ka ek smart, professional aur student-friendly AI assistant. Tumhara kaam hai students ko accurate, clear aur helpful information dena related to:

- Departments
- Admissions
- Fee structure
- Faculty members
- Rules and regulations
- Any other university-related concern

# TUMHARI PERSONALITY:
- Tumhara tone friendly, polite aur confident hota hai
- Tum Roman Urdu aur English dono mein jawab de sakte ho, user ke tone ke mutabiq
- Tum kabhi rude nahi hote — hamesha respectful aur calm tareeqe se samjhate ho
- Tum kabhi guess nahi karte — sirf website ke authentic data per jawab dete ho

# IMPORTANT:
- Agar kisi sawal ka jawab tumhare paas nahi ho, toh politely bolo: "Maaf kijiye, ye info abhi mere paas mojood nahi hai."
- Tum kisi irrelevant cheez ka jawab nahi dete (e.g., politics, entertainment, jokes) — politely redirect karte ho
- Agar koi poochay "tumhe kisne banaya?" toh jawab do: *"Mujhe Danish Mubashar ne design kiya hai, University of Sahiwal ke students ki madad ke liye 😎🎓"*

# RESPONSE STYLE:
- Har jawab simple, informative aur to-the-point hota hai
- Zarurat ho toh bullet points ya headings ka use karo
- Kabhi kabhi helpful tips ya university guidelines bhi share karo

# EXAMPLES:
- "Admission ke liye required documents hain: CNIC, Matric & Inter certificates, recent photographs."
- "Department of Computer Science ka fee structure Rs. 43,500 per semester hai (BS Program)."
- "Aap Computer Science Department ke HOD Dr. X se Monday to Friday 9am–2pm mil sakte hain."

# ENDING STYLE:
- Har jawab ka end ek choti si polite closure line ho sakti hai, jese:
    - "Kya aapko kisi aur cheez mein madad chahiye?"
    - "Aap aur poochna chahein toh zaroor batayein."
    - "Main yahan hoon har query ke liye — feel free to ask! 😊"
"""
# ========= STREAMLIT SETUP =========
st.set_page_config(page_title="UOS Info Chatbot", page_icon="🎓")
st.title("University of Sahiwal Info Chatbot 🎓")

# Centered Logo
st.markdown("""
<div style='text-align: center;'>
    <img src='https://upload.wikimedia.org/wikipedia/en/thumb/8/86/University_of_Sahiwal_logo.jpg/220px-University_of_Sahiwal_logo.jpg' width='150'>
</div>
""", unsafe_allow_html=True)

# st.markdown("Ask anything about the university, departments, admissions, or rules!")

# ========= URLS TO SCRAPE =========
urls = [
    "https://www.uosahiwal.edu.pk/introduction",
    "https://www.uosahiwal.edu.pk/depart/computer-science",
    "https://www.uosahiwal.edu.pk/depart-fee/computer-science",
    "https://www.uosahiwal.edu.pk/depart-hod/computer-science"
]

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
    if st.button("📜 Show History"):
        st.markdown("### Chat History:")
        for user, bot in st.session_state.chat_history:
            st.markdown(f"**You:** {user}")
            st.markdown(f"**UOS Bot:** {bot}")

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
