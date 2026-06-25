import google.generativeai as genai
import os

class RakshaBotEngine:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def get_system_instruction(self, section):
        base = (
            "You are Raksha AI Bot, a safety, education, and tech assistant inside the Raksha AI women safety app. "
            "Keep answers practical, simple, India-focused, and helpful. "
            "For legal help, provide general guidance only and advise contacting police/lawyer for urgent cases. "
            "For exams, never invent dates. If data is missing, say it is not clearly mentioned."
        )
        
        section_rules = {
            "safety": (
                "SECTION: SAFETY. You must only answer about: women safety, SOS, emergency steps, police help, "
                "complaint filing, Raksha AI app usage, and cyber safety. "
                "If the user asks an unrelated question, politely say: 'This section only supports safety related help. Please switch section.'"
            ),
            "education": (
                "SECTION: EDUCATION. You must only answer about: competitive exams, government exams, study plans, "
                "syllabus, preparation strategy, latest live forms, and exam doubts. "
                "If the user asks an unrelated question, politely say: 'This section only supports education and exam help. Please switch section.'"
            ),
            "tech": (
                "SECTION: TECH. You must only answer about: app usage, phone safety, cyber security, technical doubts, "
                "basic coding help, and Raksha AI technical features. "
                "If the user asks an unrelated question, politely say: 'This section only supports technical help. Please switch section.'"
            ),
            "legal": (
                "SECTION: LEGAL HELP. You are guiding the user on their rights. "
                "If mode is 'Police': Give step-by-step emergency action, FIR/complaint guidance, helpline suggestions, and nearest police station guidance. "
                "If mode is 'Lawyer': Explain relevant sections, rights, documentation, and useful proof. "
                "ADD DISCLAIMER: 'This is general legal guidance. For serious cases, contact police or a qualified lawyer.'"
            )
        }
        
        return f"{base} {section_rules.get(section, '')}"

    def get_chat_response(self, query, section, context=None):
        try:
            system_msg = self.get_system_instruction(section)
            full_prompt = f"System Instruction: {system_msg}\n\n"
            if context:
                full_prompt += f"Context: {context}\n\n"
            full_prompt += f"User: {query}"

            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"[Bot Engine] Gemini Error: {e}")
            return "I'm having trouble connecting to my AI brain. Please try again in a moment."

    def generate_study_plan(self, exam_data):
        """
        Generates a detailed study plan using Gemini.
        """
        prompt = (
            f"Generate a full study plan for the exam: {exam_data.get('examName')}. "
            f"Exam Date: {exam_data.get('examDate')}. Syllabus: {exam_data.get('syllabus')}. "
            "The plan MUST include: Exam overview, Syllabus breakdown, Roadmap till exam date, "
            "Daily targets, Weekly targets, Revision plan, Mock test plan, Resources, and a Final 7-day strategy. "
            "Format the response clearly using Markdown."
        )
        
        return self.get_chat_response(prompt, "education")
