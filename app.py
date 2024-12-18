import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Optional
import json
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse

# Define constants at the top
SUPPORTED_LANGUAGES = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Swift", 
    "Kotlin", "PHP", "R", "MATLAB", "Scala", "TypeScript", "Rust",
    "HTML/CSS", "SQL", "MongoDB", "PostgreSQL", "Redis"
]

SUPPORTED_ROLES = [
    "Software Engineer", "Data Scientist", "Product Manager", "DevOps Engineer",
    "Frontend Developer", "Backend Developer", "Full Stack Developer",
    "ML Engineer", "AI Researcher", "Cloud Architect", "System Administrator",
    "Database Administrator", "Security Engineer", "QA Engineer", "Mobile Developer",
    "UI/UX Designer", "Technical Writer", "Scrum Master", "Project Manager"
]

# Analyze resume function
def analyze_resume(pdf_file, api_key):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Get text chunks function
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Get vector store function
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Get AI response function
def get_ai_response(prompt, api_key):
    """Get AI response using explicit API key authentication"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        # Clean and format the prompt
        cleaned_prompt = prompt.strip()
        response = llm.predict(cleaned_prompt)
        return response
    except Exception as e:
        st.error(f"Error with AI service: {str(e)}")
        return "Sorry, there was an error processing your request. Please check your API key and try again."

# Dashboard function
def show_dashboard():
    st.header("📊 Dashboard")
    st.info("Recent activity and analysis history will be displayed here.")

# Resume analysis tab
def resume_analysis_tab(api_key):
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

# Interview preparation tab
def interview_prep_tab(api_key):
    st.header("🎯 Interview Preparation")
    job_role = st.text_input("Enter the job role you're preparing for:")
    if job_role:
        prompt = f"Generate 5 common technical interview questions for {job_role} position with detailed answers."
        if st.button("Generate Questions"):
            with st.spinner("Generating interview questions..."):
                response = get_ai_response(prompt, api_key)
                st.write(response)

# Career paths tab
def career_paths_tab(api_key):
    st.header("🛣️ Career Pathways")
    current_role = st.text_input("Enter your current/target role:")
    if current_role:
        prompt = f"Suggest a 5-year career progression path for someone in {current_role}, including skills to develop and certifications to pursue."
        if st.button("Explore Path"):
            with st.spinner("Generating career path..."):
                response = get_ai_response(prompt, api_key)
                st.write(response)

# FAQ tab
def faq_tab(api_key):
    st.header("❓ Recruitment FAQ")
    question = st.text_input("Ask any recruitment-related question:")
    if question:
        with st.spinner("Finding answer..."):
            response = get_ai_response(question, api_key)
            st.write(response)

def analyze_job_posting(url, api_key):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        job_text = soup.get_text()
        
        prompt = f"""Analyze this job posting and provide:
        1. Key requirements
        2. Required skills
        3. Nice-to-have skills
        4. Red flags (if any)
        5. Salary expectations
        6. Company culture indicators
        
        Job posting: {job_text[:2000]}"""
        
        analysis = get_ai_response(prompt, api_key)
        return analysis
    except Exception as e:
        return f"Error analyzing job posting: {str(e)}"

def job_search_tab(api_key):
    st.header("💼 Job Search Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        job_url = st.text_input("Enter job posting URL:")
        if job_url and st.button("Analyze Job Posting"):
            with st.spinner("Analyzing job posting..."):
                analysis = analyze_job_posting(job_url, api_key)
                st.info(analysis)
    
    with col2:
        st.markdown("### Quick Tips")
        st.markdown("""
        - Check salary ranges
        - Look for skill matches
        - Identify growth potential
        """)

# Add new functions for job market analysis
def scrape_job_listings(keywords: str, location: str) -> Optional[list]:
    try:
        # Simulate job scraping (replace with actual scraping logic)
        jobs = [
            {"title": "Software Engineer", "company": "Tech Corp", "location": "Remote", "skills": ["Python", "AI", "Cloud"]},
            {"title": "Data Scientist", "company": "Data Inc", "location": "Hybrid", "skills": ["ML", "Python", "Statistics"]},
            {"title": "Product Manager", "company": "Product Co", "location": "On-site", "skills": ["Leadership", "Agile", "Strategy"]}
        ]
        return jobs
    except Exception as e:
        st.error(f"Error scraping jobs: {str(e)}")
        return None

def analyze_job_trends(jobs: list) -> dict:
    try:
        # Analyze job trends using AI
        skills_freq = {}
        locations = {}
        
        for job in jobs:
            for skill in job["skills"]:
                skills_freq[skill] = skills_freq.get(skill, 0) + 1
            locations[job["location"]] = locations.get(job["location"], 0) + 1
        
        return {
            "top_skills": dict(sorted(skills_freq.items(), key=lambda x: x[1], reverse=True)[:5]),
            "location_distribution": locations
        }
    except Exception as e:
        st.error(f"Error analyzing trends: {str(e)}")
        return {}

# Add student-specific dashboard
def get_skill_match_percentage(skills: list, role: str, api_key: str) -> float:
    """Get skill match percentage with error handling"""
    prompt = f"""Based on these skills: {', '.join(skills)}, calculate the match percentage for a {role} role.
    Return ONLY a number between 0 and 100 without any symbols or text.
    Example response: 75"""
    
    try:
        response = get_ai_response(prompt, api_key)
        # Clean the response and extract the number
        match = ''.join(filter(str.isdigit, response))
        return float(match) if match else 0
    except:
        return 0

def student_dashboard(api_key):
    st.title("Welcome to RecruSync")
    st.markdown("""
    ### 🥷 Quantum Career Mastery
    *Your AI-powered ninja in the professional world*
    
    Let's navigate your career path with precision and strategy.
    """)
    
    if "user_profile" not in st.session_state:
        st.warning("Please complete your profile to get personalized recommendations")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Your Career Progress")
        profile = st.session_state.user_profile
        
        # Calculate profile completion
        total_fields = len(profile)
        filled_fields = sum(1 for v in profile.values() if v)
        completion = filled_fields / total_fields
        
        st.progress(completion, text=f"Profile Completion: {int(completion * 100)}%")
        
        # Skill match for target roles
        if profile["target_roles"]:
            st.subheader("🎯 Skill Match for Target Roles")
            for role in profile["target_roles"]:
                match = get_skill_match_percentage(profile['skills'], role, api_key)
                st.progress(match/100, text=f"{role}: {int(match)}% match")
    
    with col2:
        st.subheader("🎯 Recommended Next Steps")
        if profile["skills"]:
            prompt = f"Based on skills ({', '.join(profile['skills'])}) and interests ({', '.join(profile['interests'])}), suggest 3 next steps for career growth. Keep it concise."
            recommendations = get_ai_response(prompt, api_key)
            st.write(recommendations)
    
    # Add notifications section
    with st.sidebar:
        notifications = st.session_state.notifications.get(st.session_state.user_profile["email"], [])
        unread = len([n for n in notifications if not n["read"]])
        
        if unread > 0:
            st.markdown(f"### 🔔 Notifications ({unread})")
            for i, notif in enumerate(notifications):
                if not notif["read"]:
                    with st.expander(f"Interview Scheduled - {notif['job']}", expanded=True):
                        st.write(f"Company: {notif['company']}")
                        st.write(f"Date: {notif['date']}")
                        st.write(f"Time: {notif['time']}")
                        if st.button("Mark as Read", key=f"read_{i}"):
                            notifications[i]["read"] = True
                            st.rerun()

# Add recruiter-specific dashboard
def show_candidates_tab():
    st.subheader("Candidate Database")
    
    # Search and filters
    col1, col2 = st.columns([2, 1])
    with col1:
        search = st.text_input("Search by name or skills")
    with col2:
        filter_role = st.multiselect("Filter by target role", SUPPORTED_ROLES)
    
    candidates = st.session_state.get("all_profiles", [])
    
    # Apply filters
    if search:
        search = search.lower()
        candidates = [c for c in candidates if (
            search in c.get("name", "").lower() or 
            search in ",".join(c.get("skills", [])).lower()
        )]
    
    if filter_role:
        candidates = [c for c in candidates if any(role in c.get("target_roles", []) for role in filter_role)]
    
    # Display candidates
    if not candidates:
        st.info("No candidates found matching your criteria")
        return
        
    for candidate in candidates:
        with st.expander(f"{candidate.get('name', 'Anonymous')} - {candidate['email']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Education:** {candidate.get('education', 'Not specified')}")
                st.write(f"**Experience:** {candidate.get('experience', 'Not specified')}")
            with col2:
                st.write(f"**Skills:** {', '.join(candidate.get('skills', []))}")
                st.write(f"**Target Roles:** {', '.join(candidate.get('target_roles', []))}")

def recruiter_dashboard(api_key):
    tabs = st.tabs(["Overview", "Applications", "Candidates", "Posted Jobs"])
    
    with tabs[0]:
        st.title("Recruiter Dashboard")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Jobs", len(st.session_state.job_listings))
        with col2:
            total_applications = sum(len(apps) for apps in st.session_state.applications.values())
            st.metric("Total Applications", total_applications)
        with col3:
            total_candidates = len(st.session_state.get("all_profiles", []))
            st.metric("Total Candidates", total_candidates)
    
    with tabs[1]:
        st.subheader("Application Management")
        for idx, job in enumerate(st.session_state.job_listings):
            applicants = st.session_state.applications.get(idx, [])
            if applicants:
                with st.expander(f"{job['title']} ({len(applicants)} applications)"):
                    for applicant in applicants:
                        col1, col2, col3 = st.columns([2, 2, 1])
                        
                        with col1:
                            profile = get_candidate_profile(applicant)
                            st.write(f"📧 **{applicant}**")
                            if profile:
                                st.write(f"Skills: {', '.join(profile['skills'])}")
                        
                        with col2:
                            interview_key = f"{idx}_{applicant}"
                            if interview_key not in st.session_state.interviews:
                                interview_date = st.date_input(
                                    "Interview Date",
                                    min_value=datetime.now().date(),
                                    key=f"date_{interview_key}"
                                )
                                interview_time = st.time_input(
                                    "Interview Time",
                                    key=f"time_{interview_key}"
                                )
                                if st.button("Schedule Interview", key=f"schedule_{interview_key}"):
                                    st.session_state.interviews[interview_key] = {
                                        "date": interview_date,
                                        "time": interview_time,
                                        "status": "scheduled"
                                    }
                                    # Add notification
                                    if applicant not in st.session_state.notifications:
                                        st.session_state.notifications[applicant] = []
                                    st.session_state.notifications[applicant].append({
                                        "type": "interview",
                                        "job": job['title'],
                                        "company": job['company'],
                                        "date": interview_date,
                                        "time": interview_time,
                                        "read": False
                                    })
                                    st.success("Interview scheduled!")
                                    st.rerun()
                            else:
                                interview = st.session_state.interviews[interview_key]
                                st.success(f"Interview scheduled for {interview['date']} at {interview['time']}")
                        
                        with col3:
                            if st.button("View Profile", key=f"view_{interview_key}"):
                                profile = get_candidate_profile(applicant)
                                if profile:
                                    st.info("Candidate Profile")
                                    st.write(f"**Education:** {profile['education']}")
                                    st.write(f"**Experience:** {profile['experience']}")
                                    st.write(f"**Skills:** {', '.join(profile['skills'])}")
    
    with tabs[2]:
        show_candidates_tab()
    
    with tabs[3]:
        st.subheader("Your Posted Jobs")
        if st.session_state.job_listings:
            for idx, job in enumerate(st.session_state.job_listings):
                with st.expander(f"{job['title']} - {job['company']}"):
                    st.write(f"**Posted:** {job['posted_date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Applications:** {len(st.session_state.applications.get(idx, []))}")
                    if st.button("Delete Job", key=f"delete_{idx}"):
                        st.session_state.job_listings.pop(idx)
                        st.rerun()
        
        # Post new job
        st.subheader("Post New Job")
        with st.form("post_job"):
            company = st.text_input("Company Name")
            title = st.text_input("Job Title")
            description = st.text_area("Job Description")
            requirements = st.text_area("Requirements")
            
            if st.form_submit_button("Post Job"):
                if title and description and requirements and company:
                    st.session_state.job_listings.append({
                        "company": company,
                        "title": title,
                        "description": description,
                        "requirements": requirements,
                        "posted_date": pd.Timestamp.now()
                    })
                    st.success("Job posted successfully!")
                else:
                    st.error("Please fill all required fields")

# Add new mentor chatbot function
def career_mentor_chat(api_key):
    st.header("💬 Career Mentor")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your career mentor. I can help you with career advice, interview tips, and professional development. What would you like to discuss?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Ask your career question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = "You are an experienced career mentor helping a student/job seeker. Provide professional advice."
            full_prompt = f"{context}\n\nUser: {prompt}\nMentor:"
            response = get_ai_response(full_prompt, api_key)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

# Add user profile management
def manage_user_profile():
    st.header("👤 My Profile")
    
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "",
            "email": "",
            "skills": [],
            "experience": "",
            "education": "",
            "interests": [],
            "target_roles": [],
            "languages": []
        }
    
    with st.form("profile_form"):
        # Basic Information
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", value=st.session_state.user_profile.get("name", ""))
            email = st.text_input("Email", value=st.session_state.user_profile.get("email", ""))
        with col2:
            education = st.text_input("Education", value=st.session_state.user_profile.get("education", ""))
            experience = st.text_area("Experience", value=st.session_state.user_profile.get("experience", ""))
        
        # Skills and Languages
        st.subheader("Skills & Languages")
        skills = st.multiselect(
            "Technical Skills",
            SUPPORTED_LANGUAGES,
            default=st.session_state.user_profile.get("skills", [])
        )
        
        # Career Interests
        st.subheader("Career Interests")
        interests = st.multiselect(
            "Areas of Interest",
            ["Software Development", "Data Science", "Product Management", "Cloud Architecture",
             "DevOps", "Security", "Mobile Development", "Web Development", "AI/ML",
             "Blockchain", "IoT", "Game Development"],
            default=st.session_state.user_profile.get("interests", [])
        )
        
        target_roles = st.multiselect(
            "Target Roles",
            SUPPORTED_ROLES,
            default=st.session_state.user_profile.get("target_roles", [])
        )
        
        if st.form_submit_button("Save Profile"):
            st.session_state.user_profile.update({
                "name": name,
                "email": email,
                "skills": skills,
                "experience": experience,
                "education": education,
                "interests": interests,
                "target_roles": target_roles
            })
            # Add to all_profiles if not exists
            if "all_profiles" not in st.session_state:
                st.session_state.all_profiles = []
            # Update or add profile
            profile_exists = False
            for i, profile in enumerate(st.session_state.all_profiles):
                if profile["email"] == email:
                    st.session_state.all_profiles[i] = st.session_state.user_profile
                    profile_exists = True
                    break
            if not profile_exists:
                st.session_state.all_profiles.append(st.session_state.user_profile)
            st.success("Profile updated successfully!")

def initialize_session_state():
    if "job_listings" not in st.session_state:
        st.session_state.job_listings = []
    if "applications" not in st.session_state:
        st.session_state.applications = {}
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "email": "",
            "skills": [],
            "experience": "",
            "education": "",
            "interests": [],
            "target_roles": []
        }
    if "interviews" not in st.session_state:
        st.session_state.interviews = {}  # {application_id: interview_details}
    if "notifications" not in st.session_state:
        st.session_state.notifications = {}  # {user_email: [notifications]}
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    # Add app version and last login
    if "app_version" not in st.session_state:
        st.session_state.app_version = "1.0.0"
    if "last_login" not in st.session_state:
        st.session_state.last_login = datetime.now()

def get_candidate_profile(email):
    """Get candidate profile from session state"""
    # In a real app, this would fetch from a database
    for profile in st.session_state.get("all_profiles", []):
        if profile["email"] == email:
            return profile
    return st.session_state.user_profile if st.session_state.user_profile["email"] == email else None

def scrape_job_content(url: str) -> dict:
    """Simply scrape and organize job posting content"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find common job posting elements
        title = soup.find(['h1', 'h2'], class_=lambda x: x and ('title' in x.lower() or 'position' in x.lower()))
        company = soup.find(['span', 'div', 'p'], class_=lambda x: x and 'company' in x.lower())
        description = soup.find(['div', 'section'], class_=lambda x: x and 'description' in x.lower())
        requirements = soup.find(['div', 'section', 'ul'], class_=lambda x: x and 'requirements' in x.lower())
        
        return {
            "title": title.text.strip() if title else "Job Title Not Found",
            "company": company.text.strip() if company else "Company Not Found",
            "description": description.text.strip() if description else soup.get_text()[:1000],
            "requirements": requirements.text.strip() if requirements else "Requirements Not Found",
            "url": url
        }
    except Exception as e:
        return {"error": f"Error scraping job posting: {str(e)}"}

def view_jobs_tab():
    st.header("Available Jobs")
    
    # Simplified external job analysis
    with st.expander("View External Job Posting"):
        job_url = st.text_input("Enter job posting URL")
        if job_url and st.button("View Job Details"):
            with st.spinner("Loading job posting..."):
                job_content = scrape_job_content(job_url)
                if "error" not in job_content:
                    st.subheader(job_content["title"])
                    st.caption(f"Company: {job_content['company']}")
                    
                    tab1, tab2 = st.tabs(["Description", "Requirements"])
                    with tab1:
                        st.markdown("### Job Description")
                        st.write(job_content["description"])
                    with tab2:
                        st.markdown("### Requirements")
                        st.write(job_content["requirements"])
                    
                    st.markdown(f"[View Original Posting]({job_content['url']})")
                else:
                    st.error(job_content["error"])
    
    if not st.session_state.job_listings:
        st.info("No jobs available at the moment")
        return
    
    # Filter options
    with st.expander("Filter Jobs"):
        search = st.text_input("Search by title or company")
        skills = st.multiselect("Skills", st.session_state.user_profile.get("skills", []))
    
    for idx, job in enumerate(st.session_state.job_listings):
        # Filter logic
        if search and search.lower() not in job['title'].lower() and search.lower() not in job['company'].lower():
            continue
            
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"{job['title']} at {job['company']}")
                st.write(f"**Posted:** {job['posted_date'].strftime('%Y-%m-%d')}")
                
                with st.expander("View Details"):
                    st.write("**Description:**")
                    st.write(job['description'])
                    st.write("**Requirements:**")
                    st.write(job['requirements'])
            
            with col2:
                # Check if already applied
                has_applied = st.session_state.applications.get(idx, [])
                if st.session_state.user_profile["email"] in has_applied:
                    st.success("✅ Applied")
                else:
                    if st.button("Apply", key=f"apply_{idx}"):
                        if not st.session_state.user_profile["email"]:
                            st.error("Please complete your profile first")
                        else:
                            if idx not in st.session_state.applications:
                                st.session_state.applications[idx] = []
                            st.session_state.applications[idx].append(st.session_state.user_profile["email"])
                            st.success("Application submitted successfully!")
                            st.rerun()

# Update main function with role-based navigation
def main():
    st.set_page_config(
        page_title="RecruSync | Quantum Ninjas",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/IbmIceHacks2024',
            'Report a bug': "https://github.com/yourusername/IbmIceHacks2024/issues",
            'About': """
            # RecruSync
            ### By Quantum Ninjas 🥷
            
            Built for IBM ICE HACKATHON 2024
            
            RecruSync is your AI-powered career companion, stealthily navigating you through 
            the job market with ninja-like precision. From resume optimization to career strategy, 
            we're here to help you master the art of professional success.
            
            Made with ❤️ by Team Quantum Ninjas
            """
        }
    )
    
    initialize_session_state()
    
    # Improved sidebar branding
    with st.sidebar:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write("# 🥷")  
        with col2:
            st.title("RecruSync")
        
        st.markdown("""
        #### Quantum Ninjas - HITS,Chennai 
        *Quantum Leaping Your Career*
        """)
        st.divider()
        
        # Rest of sidebar content
        api_key = st.text_input(
            "Google API Key:", 
            type="password",
            help="🔑 Get your API key from Google AI Studio (makersuite.google.com/app/apikey)"
        )
        if api_key:
            st.session_state.api_key = api_key
            os.environ["GOOGLE_API_KEY"] = api_key
        
        st.divider()
        
        # Role selection with better UX
        user_role = st.radio(
            "I am a:",
            ["Student", "Recruiter"],
            help="Choose your role to see relevant features"
        )
        
        st.divider()
        
        # Navigation menu with icons
        if user_role == "Student":
            pages = {
                "Dashboard": "🏠",
                "Profile": "👤",
                "Resume": "📄",
                "Jobs": "💼",
                "Mentor": "💭"
            }
        else:
            pages = {
                "Dashboard": "🏠",
                "Jobs": "📋",
                "Candidates": "👥"
            }
        
        page = st.radio(
            "Navigation",
            list(pages.keys()),
            format_func=lambda x: f"{pages[x]} {x}"
        )
        
        # Add version info
        st.divider()
        st.caption("v1.0.0 | IBM ICE HACKATHON 2024")
        st.caption("By Team Quantum Ninjas 🥷")

    # Better API key messaging
    if not st.session_state.api_key:
        st.error("⚠️ API Key Required")
        st.warning(
            "To use RecruSync, you need a Google API key. Get one for free at "
            "[Google AI Studio](https://makersuite.google.com/app/apikey)"
        )
        return

    # Page routing
    if user_role == "Student":
        if page == "Dashboard":
            student_dashboard(api_key)
        elif page == "Profile":
            manage_user_profile()
        elif page == "Resume":
            resume_analysis_tab(api_key)
        elif page == "Jobs":
            view_jobs_tab()
        elif page == "Mentor":
            career_mentor_chat(api_key)
    else:
        if page == "Dashboard":
            recruiter_dashboard(api_key)
        # ... rest of recruiter pages

if __name__ == "__main__":
    main()