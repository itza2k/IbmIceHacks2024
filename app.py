import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

def analyze_resume(pdf_file, api_key):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_ai_response(prompt, api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
    response = llm.predict(prompt)
    return response

def main():
    st.set_page_config(page_title="RecruSync - AI Recruitment Assistant", layout="wide")
    
    st.title("🎯 RecruSync")
    st.subheader("Your AI-Powered Recruitment Assistant")
    
    # API Key Input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Google API Key:", type="password")
        os.environ["GOOGLE_API_KEY"] = api_key
        
        st.header("Navigation")
        page = st.radio("Select Feature", 
                       ["Resume Analysis", "Interview Prep", "Career Paths", "FAQ"])
    
    if not api_key:
        st.warning("Please enter your Google API key to continue.")
        return
    
    if page == "Resume Analysis":
        st.header("📄 Resume Analysis")
        uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=['pdf'])
        
        if uploaded_file and st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                resume_text = analyze_resume(uploaded_file, api_key)
                prompt = f"""Analyze this resume and provide feedback on:
                1. Key strengths
                2. Areas for improvement
                3. ATS optimization suggestions
                4. Format and presentation
                
                Resume text: {resume_text}"""
                
                analysis = get_ai_response(prompt, api_key)
                st.success("Analysis Complete!")
                st.write(analysis)
    
    elif page == "Interview Prep":
        st.header("🎯 Interview Preparation")
        job_role = st.text_input("Enter the job role you're preparing for:")
        if job_role:
            prompt = f"Generate 5 common technical interview questions for {job_role} position with detailed answers."
            if st.button("Generate Questions"):
                with st.spinner("Generating interview questions..."):
                    response = get_ai_response(prompt, api_key)
                    st.write(response)
    
    elif page == "Career Paths":
        st.header("🛣️ Career Pathways")
        current_role = st.text_input("Enter your current/target role:")
        if current_role:
            prompt = f"Suggest a 5-year career progression path for someone in {current_role}, including skills to develop and certifications to pursue."
            if st.button("Explore Path"):
                with st.spinner("Generating career path..."):
                    response = get_ai_response(prompt, api_key)
                    st.write(response)
    
    else:  # FAQ
        st.header("❓ Recruitment FAQ")
        question = st.text_input("Ask any recruitment-related question:")
        if question:
            with st.spinner("Finding answer..."):
                response = get_ai_response(question, api_key)
                st.write(response)

if __name__ == "__main__":
    main()