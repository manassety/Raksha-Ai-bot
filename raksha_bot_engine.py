from openai import OpenAI
import os

class RakshaBotEngine:
    def __init__(self, api_key):
        self.default_model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528:free")
        self.fallback_models = [
            self.default_model,
            "deepseek/deepseek-r1-0528:free",
            "z-ai/glm-4.5-air:free",
            "qwen/qwen3-coder:free",
            "moonshotai/kimi-k2:free"
        ]
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print(f"[Bot Engine] Initialized with OpenRouter models. Primary: {self.default_model}")
        
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
        system_msg = self.get_system_instruction(section)
        messages = [{"role": "system", "content": system_msg}]
        
        if context:
            messages.append({"role": "system", "content": f"Context: {context}"})
            
        messages.append({"role": "user", "content": query})

        # Fallback Logic with Error Logging
        attempted_models = []
        
        # Deduplicate models while preserving order
        unique_models = []
        for m in self.fallback_models:
            if m not in unique_models:
                unique_models.append(m)

        for model_name in unique_models:
            try:
                print(f"[Bot Engine] Attempting model: {model_name}")
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1200
                )
                
                reply = response.choices[0].message.content
                return {
                    "success": True,
                    "reply": reply,
                    "model_used": model_name
                }
            except Exception as e:
                err_msg = str(e)
                print(f"[Bot Engine] Fail with {model_name}: {err_msg}")
                attempted_models.append({
                    "model": model_name,
                    "error": err_msg
                })
        
        # If all fail
        return {
            "success": False,
            "reply": "I'm having trouble connecting to all my AI brains. Please check back soon.",
            "error": "All OpenRouter models failed",
            "attempted_models": attempted_models
        }

    def generate_study_plan(self, exam_data):
        prompt = (
            f"Generate a full study plan for the exam: {exam_data.get('examName')}. "
            f"Exam Date: {exam_data.get('examDate')}. Syllabus: {exam_data.get('syllabus')}. "
            "The plan MUST include: Exam overview, Syllabus breakdown, Roadmap till exam date, "
            "Daily targets, Weekly targets, Revision plan, Mock test plan, Resources, and a Final 7-day strategy. "
            "Format the response clearly using Markdown."
        )
        result = self.get_chat_response(prompt, "education")
        return result.get("reply") if result.get("success") else "Error generating study plan."
