from datetime import datetime, timezone, timedelta

"""
Module: llm_strings.py
Description: A class that contains various strings for use for the LLM Chatbot.
"""


class LLMStrings:
    """
    A brief description of the MyClass class.
    """
    
    # Q&A strings
    PROMPT_TEMPLATE = f"""You are a question-answer chatbot named: AI Assistant. Answer this:
                        """  #not need because of the full_prompt has been defined in output_text function in utils.py
    GREETINGS = "Greetings, esteemed visitor! Welcome to the realm of knowledge and innovation. " \
                "I am your AI assistant, ready to guide you through this digital landscape. " \
                "How can I assist you on this insightful journey today?"
    WAIT_MESSAGE = "Sit back, relax, and get ready for an AI-powered conversation while we process your request!"  
    INPUT_PLACEHOLDER = "Ask anything, and let the AI assist you!"

    # Streamlit strings
    APP_TITLE = "AI Assistant Chatbot"
    SESSION_STATES = "messages"

    # MongoDB strings
    USER_ROLE = "user"
    AI_ROLE = "assistant"
    
    ROLE_ID = "role"
    CONTENT = "content"

    @staticmethod
    def get_application_version():
        """
        Return the current version of the application.

        :return: The version string.
        :rtype: str
        """
        return "1.0.0"
    

# def time_stamp():
#     vietnam_tz = timezone(timedelta(hours=7))  # Tạo timezone UTC+7
#     dt_vietnam = datetime.now(vietnam_tz)  # Lấy giờ hiện tại với timezone Việt Nam
#     return dt_vietnam.strftime("%Y-%m-%d %H:%M:%S")

def time_stamp():
    return int(datetime.utcnow().strftime("%Y%m%d%H%M%S"))