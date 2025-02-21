import google.generativeai as genai
from config import GOOGLE_API_KEY

# Cấu hình Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_response(prompt):
    """Gọi API Gemini để lấy phản hồi"""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text
