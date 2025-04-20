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
urls = [
    "https://www.uosahiwal.edu.pk/introduction",
    "https://www.uosahiwal.edu.pk/chancellor-message",
    "https://www.uosahiwal.edu.pk/vc-message",
    "https://www.uosahiwal.edu.pk/uni-map",
    "https://www.uosahiwal.edu.pk/newsletter",
    "https://www.uosahiwal.edu.pk/events",	
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-celebrates-inclusivity-with-heartwarming-christmas-cake-cutting-ceremony",
    "https://www.uosahiwal.edu.pk/event/vice-chancellor-pays-tribute-to-national-martyrs-on-defence-day",
    "https://www.uosahiwal.edu.pk/event/kabaddi-tournament-winners",
    "https://www.uosahiwal.edu.pk/event/group-photo-of-cricket-tournament-winner-sports-gala",
    "https://www.uosahiwal.edu.pk/event/vice-chancellor-leads-crucial-meeting-to-elevate-academic-standards-ahead-of-summer-semester-2024",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-and-emerson-university-multan-sign-mou-to-enhance-collaboration-in-research",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-hosts-seminar-on-combating-fake-news-in-the-digital-age",
    "https://www.uosahiwal.edu.pk/event/female-participants-of-annual-sports-gala-2024",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-and-punjab-higher-education-commission-host-seminar-on-impact-of-interfaith-programs-in-pakistani-universities",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwals-annual-sports-gala2024-concludes-with-grand-closing-ceremony",
    "https://www.uosahiwal.edu.pk/event/6th-board-of-advanced-studies-and-research-meeting-concludes-successfully-with-key-decisions-for-academic-progress",
    "https://www.uosahiwal.edu.pk/event/vice-chancellor-congratulates-294-scholars-awarded-pkr-125-million-under-cm-punjabs-honhaar-scholarship-scheme",
    "https://www.uosahiwal.edu.pk/event/football-winner-team-sports-gala-2024",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-partners-with-qingdao-university-to-enhance-chinese-language-education-for-students",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-hosts-seminar-with-dpo-sahiwal-as-the-keynote-speaker-on-the-elimination-of-gender-based-violence-from-society",
    "https://www.uosahiwal.edu.pk/event/secretary-rta-sahiwal-engages-students-at-university-of-sahiwal-to-promote-chief-ministers-youth-initiative-for-providing-motor-and-electric-bikes",
    "https://www.uosahiwal.edu.pk/event/vice-chancellor-approves-faculty-members-to-attend-phecs-training-program-on-teaching-excellence",
    "https://www.uosahiwal.edu.pk/event/hec-organizes-virtual-seminar-on-awareness-regarding-prevention-of-blasphemous-materialactivities-on-social-media",
    "https://www.uosahiwal.edu.pk/event/acknowledgement",
    "https://www.uosahiwal.edu.pk/event/green-youth-movement-club-university-of-sahiwal-leads-plantation-drive-at-campus",
    "https://www.uosahiwal.edu.pk/event/worthy-vice-chancellors-message-on-pakistan-resolution-day",
    "https://www.uosahiwal.edu.pk/event/universities-of-sahiwal-and-sargodha-pledge-to-form-a-strategic-partnership-to-enhance-academic-development-and-research-opportunities",
    "https://www.uosahiwal.edu.pk/event/directorate-of-student-affairs-university-of-sahiwal-hosts-commemorative-walk-and-speech-competition-to-mark-pakistan-resolution-day",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-honors-martyrs-of-pakistan-armed-forces-with-seminar-and-solidarity-walk",
    "https://www.uosahiwal.edu.pk/event/opening-ceremony-of-annual-sports-gala-2024",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-hosts-renowned-prof-dr-khalijah-awang-for-keynote-lecture-on-natural-products-research",
    "https://www.uosahiwal.edu.pk/event/media-club-of-university-of-sahiwal-engages-with-vice-chancellor-and-press-club-president",
    "https://www.uosahiwal.edu.pk/event/vice-chancellor-commemorates-labour-day-and-acknowledges-contributions-of-labourers-in-pakistans-development",
    "https://www.uosahiwal.edu.pk/event/majeed-amjad-literary-society-mals-commemorate-iqbal-day-with-seminar-at-university-of-sahiwal",
    "https://www.uosahiwal.edu.pk/event/minister-for-higher-education-and-commissioner-sahiwal-division-inspect-ongoing-hec-development-project-at-university-of-sahiwal",
    "https://www.uosahiwal.edu.pk/event/university-of-sahiwal-hosts-the-launching-ceremony-of-pms",
    "https://www.uosahiwal.edu.pk/event/department-of-information-technology-and-regional-plan9-organize-seminar-to-foster-startup-innovation",
    "https://www.uosahiwal.edu.pk/news",
    "https://www.uosahiwal.edu.pk/news/tender-document-for-the-procurement-of-answer-sheets",
    "https://www.uosahiwal.edu.pk/news/university-of-sahiwal-partners-with-inti-international-university-of-malaysia-for-strategic-collaboration",
    "https://www.uosahiwal.edu.pk/news/supply-and-installation-of-230kw-solar-systems-at-university-of-sahiwal",
    "https://www.uosahiwal.edu.pk/news/driving-license-of-students-in-public-sector-universities",
    "https://www.uosahiwal.edu.pk/news/chief-minister-honhaar-undergraduate-scholarship-program",
    "https://www.uosahiwal.edu.pk/news/inquiry-committee-against-workplace-harassment",
    "https://www.uosahiwal.edu.pk/news/beware-and-stay-alert",
    "https://www.uosahiwal.edu.pk/news/vice-chancellors-message-on-kashmir-black-day",
    "https://www.uosahiwal.edu.pk/news/plan9-opportunity",
    "https://www.uosahiwal.edu.pk/news/chief-ministers-youth-bike-initiative",
    "https://www.uosahiwal.edu.pk/news/university-of-sahiwal-expands-student-service-center-inaugurated-by-vice-chancellor",
    "https://www.uosahiwal.edu.pk/news/tender-document-for-the-procurement-of-answer-sheets-tyres-and-supply-installation-of-solar",
    "https://www.uosahiwal.edu.pk/news/eid-ul-fitr-holidays-2025",
    "https://www.uosahiwal.edu.pk/news/standing-accessibility-committee",
    "https://www.uosahiwal.edu.pk/news/lecture-timing-during-ramadan",
    "https://www.uosahiwal.edu.pk/news/mandatory-helmet-warning-for-students-and-faculty-while-riding-motorbikes",
    "https://www.uosahiwal.edu.pk/news/convocation-notice-form-2024",
    "https://www.uosahiwal.edu.pk/news/kashmir-solidarity-day-2025",
    "https://www.uosahiwal.edu.pk/news/google-career-certificate",
    "https://www.uosahiwal.edu.pk/news/yonyorsy-af-sayoal-k-hoal-srbor-on-oaly-ayf-ayy-ar-aor-oakaa-bbnyad",
    "https://www.uosahiwal.edu.pk/news/position-vacant-for-visiting-faculty-2025",
    "https://www.uosahiwal.edu.pk/news/important-notice-for-mid-term-exams-of-spring-2025",
    "https://www.uosahiwal.edu.pk/news/wildlife-internship-program-2025",
    "https://www.uosahiwal.edu.pk/news/a-meeting-of-heads-of-universities-was-held-at-phec-under-the-chairmanship-of-education-minister",
    "https://www.uosahiwal.edu.pk/news/transportation-fee-per-semester",
    "https://www.uosahiwal.edu.pk/office/vice-chancellor",
    "https://www.uosahiwal.edu.pk/office/registrar",
    "https://www.uosahiwal.edu.pk/office/treasurer",
    "https://www.uosahiwal.edu.pk/office/controller-examination",
    "https://www.uosahiwal.edu.pk/depart/computer-science",
    "https://www.uosahiwal.edu.pk/depart-hod/computer-science",
    "https://www.uosahiwal.edu.pk/depart-research-group/computer-science",
    "https://www.uosahiwal.edu.pk/depart-faculty/computer-science",
    "https://www.uosahiwal.edu.pk/depart-fee/computer-science",
    "https://www.uosahiwal.edu.pk/depart/business-administration",
    "https://www.uosahiwal.edu.pk/depart-hod/business-administration",
    "https://www.uosahiwal.edu.pk/depart-faculty/business-administration",
    "https://www.uosahiwal.edu.pk/depart-research-group/business-administration",
    "https://www.uosahiwal.edu.pk/depart-fee/business-administration",
    "https://www.uosahiwal.edu.pk/depart/commerce",
    "https://www.uosahiwal.edu.pk/depart-hod/commerce",
    "https://www.uosahiwal.edu.pk/depart-research-group/commerce",
    "https://www.uosahiwal.edu.pk/depart-faculty/commerce",
    "https://www.uosahiwal.edu.pk/depart-fee/commerce",
    "https://www.uosahiwal.edu.pk/depart/economics",
    "https://www.uosahiwal.edu.pk/depart-hod/economics",
    "https://www.uosahiwal.edu.pk/depart-faculty/economics",
    "https://www.uosahiwal.edu.pk/depart-research-group/economics",
    "https://www.uosahiwal.edu.pk/depart-fee/economics",
    "https://www.uosahiwal.edu.pk/depart/english",
    "https://www.uosahiwal.edu.pk/depart-hod/english",
    "https://www.uosahiwal.edu.pk/depart-faculty/english",
    "https://www.uosahiwal.edu.pk/depart-research-group/english",
    "https://www.uosahiwal.edu.pk/depart-fee/english",
    "https://www.uosahiwal.edu.pk/depart/law",
    "https://www.uosahiwal.edu.pk/depart-hod/law",
    "https://www.uosahiwal.edu.pk/depart-faculty/law",
    "https://www.uosahiwal.edu.pk/depart-research-group/law",
    "https://www.uosahiwal.edu.pk/depart-fee/law",
    "https://www.uosahiwal.edu.pk/depart/chemistry",
    "https://www.uosahiwal.edu.pk/depart-hod/chemistry",
    "https://www.uosahiwal.edu.pk/depart-faculty/chemistry",
    "https://www.uosahiwal.edu.pk/depart-research-group/chemistry",
    "https://www.uosahiwal.edu.pk/depart-fee/chemistry",
    "https://www.uosahiwal.edu.pk/depart/mathematics",
    "https://www.uosahiwal.edu.pk/depart-hod/mathematics",
    "https://www.uosahiwal.edu.pk/depart-faculty/mathematics",
    "https://www.uosahiwal.edu.pk/depart-research-group/mathematics",
    "https://www.uosahiwal.edu.pk/depart-fee/mathematics",
    "https://www.uosahiwal.edu.pk/depart/physics",
    "https://www.uosahiwal.edu.pk/depart-hod/physics",
    "https://www.uosahiwal.edu.pk/depart-faculty/physics",
    "https://www.uosahiwal.edu.pk/depart-research-group/physics",
    "https://www.uosahiwal.edu.pk/depart-fee/physics",
    "https://www.uosahiwal.edu.pk/depart/psychology",
    "https://www.uosahiwal.edu.pk/depart-hod/psychology",
    "https://www.uosahiwal.edu.pk/depart-faculty/psychology",
    "https://www.uosahiwal.edu.pk/depart-research-group/psychology",
    "https://www.uosahiwal.edu.pk/depart-fee/psychology",
    "https://www.uosahiwal.edu.pk/directorate/director-academics",
    "https://www.uosahiwal.edu.pk/directorate/estate-management",
    "https://www.uosahiwal.edu.pk/directorate/graduate-studies",
    "https://www.uosahiwal.edu.pk/directorate/director-it",
    "https://www.uosahiwal.edu.pk/directorate/oric",
    "https://www.uosahiwal.edu.pk/oric-team",
    "https://www.uosahiwal.edu.pk/oric-partner",
    "https://www.uosahiwal.edu.pk/oric-publications",
    "https://www.uosahiwal.edu.pk/oric-publication-summary",
    "https://www.uosahiwal.edu.pk/oric-publications?page=1",
    "https://www.uosahiwal.edu.pk/oric-publications?page=2",
    "https://www.uosahiwal.edu.pk/oric-publications?page=3",
    "https://www.uosahiwal.edu.pk/oric-publications?page=4",
    "https://www.uosahiwal.edu.pk/oric-publications?page=5",
    "https://www.uosahiwal.edu.pk/oric-publications?page=6",
    "https://www.uosahiwal.edu.pk/oric-publications?page=7",
    "https://www.uosahiwal.edu.pk/oric-publications?page=8",
    "https://www.uosahiwal.edu.pk/oric-publications?page=9",
    "https://www.uosahiwal.edu.pk/oric-publications?page=10",
    "https://www.uosahiwal.edu.pk/oric-publications?page=11",
    "https://www.uosahiwal.edu.pk/oric-publications?page=12",
    "https://www.uosahiwal.edu.pk/oric-publications?page=13",
    "https://www.uosahiwal.edu.pk/oric-publications?page=14",
    "https://www.uosahiwal.edu.pk/oric-publications?page=15",
    "https://www.uosahiwal.edu.pk/oric-publications?page=16",
    "https://www.uosahiwal.edu.pk/oric-publications?page=17",
    "https://www.uosahiwal.edu.pk/oric-publications?page=18",
    "https://www.uosahiwal.edu.pk/oric-publications?page=19",
    "https://www.uosahiwal.edu.pk/oric-publications?page=20",
    "https://www.uosahiwal.edu.pk/oric-publications?page=21",
    "https://www.uosahiwal.edu.pk/oric-publications?page=22",
    "https://www.uosahiwal.edu.pk/oric-publications?page=23",
    "https://www.uosahiwal.edu.pk/oric-publications?page=24",
    "https://www.uosahiwal.edu.pk/directorate/director-planning-development",
    "https://www.uosahiwal.edu.pk/directorate/director-project",
    "https://www.uosahiwal.edu.pk/directorate/qec",
    "https://www.uosahiwal.edu.pk/qec-objectives",
    "https://www.uosahiwal.edu.pk/qec-organogram",
    "https://www.uosahiwal.edu.pk/qec-team",
    "https://www.uosahiwal.edu.pk/directorate/ro",
    "https://www.uosahiwal.edu.pk/directorate/dsa",
    "https://www.uosahiwal.edu.pk/dsa-office",
    "https://www.uosahiwal.edu.pk/dsa-scholarship",
    "https://www.uosahiwal.edu.pk/dsa-downloads",
    "https://www.uosahiwal.edu.pk/directorate/sustainability",
    "https://www.uosahiwal.edu.pk/library",
    "https://www.uosahiwal.edu.pk/dsa-scholarship",
    "https://www.uosahiwal.edu.pk/transport",
    "https://www.uosahiwal.edu.pk/hostel",
    "https://www.uosahiwal.edu.pk/sports",
    "https://www.uosahiwal.edu.pk/plan9",
    "https://www.uosahiwal.edu.pk/DlseiCoursera",
    "https://www.uosahiwal.edu.pk/directorate/sports",
    "https://www.uosahiwal.edu.pk/sports"
]



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
- If someone uses the term HOD in a question, it refers to the chairman. The chairman is also referred to as HOD in other words.
# RESPONSE STYLE:
- Every response should be simple, informative, and to the point  
- Use bullet points or headings when helpful  
- Occasionally include useful tips or university guidelines  

# EXAMPLES:
- "Required documents for admission include: CNIC, Matric & Inter certificates, recent photographs."  
- "The fee structure for the BS Computer Science program is Rs. 43,500 per semester."  
- "You can meet the HOD of the Computer Science Department, Dr. X, from Monday to Friday, 9am–2pm etc."  

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
