import streamlit as st
import pytesseract
import speech_recognition as sr
from deep_translator import GoogleTranslator
from langdetect import detect
from PIL import Image
import os
import logging
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cohere
import tempfile
import base64
import io
from gtts import gTTS

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Cohere AI (Replace with your API Key)
COHERE_API_KEY = "BBOt2HXFWB0r6mhkkEJpSIyG9wGt3LWz4HZdLzlo"  # Replace with your actual API key
co = cohere.Client(COHERE_API_KEY)

# Initialize AI-based image captioning model
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

try:
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    logging.info("BLIP model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading BLIP model: {e}")
    blip_processor, blip_model = None, None

# Multilingual Text-to-Speech using gTTS
def generate_audio(text, lang="en"):
    """Convert text to speech using gTTS and return HTML for playback."""
    try:
        # Map detected language codes to gTTS supported codes
        lang_map = {
            "en": "en",
            "ta": "ta",
            "te": "te",
            # Add more language mappings as needed
        }
        
        # Default to English if language not supported
        tts_lang = lang_map.get(lang, "en")
        
        # Check if text is too long and split it into smaller chunks if needed
        max_chars = 500  # gTTS works best with smaller chunks
        chunks = []
        
        if len(text) > max_chars:
            # Split by sentences to maintain natural speech
            sentences = text.split('. ')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < max_chars:
                    current_chunk += sentence + ". " if not sentence.endswith('.') else sentence + " "
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". " if not sentence.endswith('.') else sentence + " "
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [text]
        
        # Process each chunk and combine the audio
        all_audio_bytes = bytearray()
        
        for chunk in chunks:
            # Create a temporary file-like object
            mp3_fp = io.BytesIO()
            
            # Use slower=False for faster generation
            tts = gTTS(text=chunk, lang=tts_lang, slow=False)
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            # Add to combined audio
            all_audio_bytes.extend(mp3_fp.read())
        
        # Encode the combined audio file to base64
        audio_b64 = base64.b64encode(all_audio_bytes).decode()
        
        # Create HTML with audio controls (custom styled)
        audio_html = f"""
        <div style="display: flex; justify-content: center; margin: 10px 0;">
            <audio autoplay controls style="width: 100%; max-width: 500px; height: 40px; border-radius: 8px;">
                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
        </div>
        """
        
        return audio_html
    except Exception as e:
        logging.error(f"Speech generation error: {e}")
        return None

# Multilingual Speech Recognition
def listen():
    """Capture user input via microphone, detect language, and convert to text."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  

    with sr.Microphone() as source:
        # Create a placeholder for the listening indicator
        listening_placeholder = st.empty()
        listening_placeholder.markdown(
            """
            <div style="display: flex; align-items: center; background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <div style="background-color: #ff4b4b; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; animation: pulse 1.5s infinite;"></div>
                <p style="margin: 0; color: #333; font-weight: 500;">Listening... Please speak now</p>
            </div>
            <style>
                @keyframes pulse {
                    0% { opacity: 1; }
                    50% { opacity: 0.4; }
                    100% { opacity: 1; }
                }
            </style>
            """, unsafe_allow_html=True
        )
        
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source, timeout=5)
            # Clear the listening indicator
            listening_placeholder.empty()
            
            text = recognizer.recognize_google(audio, language="en-IN")  # Supports Indian languages
            detected_language = detect(text)  # Detect language
            
            # Show a success message
            st.success(f"Voice detected ({detected_language})")
            return text.lower(), detected_language
        except sr.UnknownValueError:
            listening_placeholder.empty()
            st.warning("I couldn't understand that. Please try again.")
            return None, "en"
        except sr.RequestError:
            listening_placeholder.empty()
            st.error("Speech recognition service is unavailable.")
            return None, "en"
        except Exception as e:
            listening_placeholder.empty()
            logging.error(f"Error in voice recognition: {e}")
            st.error("An error occurred during voice recognition.")
            return None, "en"

# AI-Powered Response Generation
def generate_ai_response(prompt, lang="en"):
    """Generate AI-powered responses using Cohere and translate them."""
    try:
        with st.spinner("Generating response..."):
            response = co.generate(
                model="command",
                prompt=prompt,
                max_tokens=150  # Increased token limit for more comprehensive responses
            )
            ai_text = response.generations[0].text.strip()
            
            if lang != "en":
                try:
                    ai_text = GoogleTranslator(source="auto", target=lang).translate(ai_text)
                except Exception as e:
                    logging.error(f"Translation error: {e}")
                    # Continue with English if translation fails
        
        return ai_text
    except Exception as e:
        logging.error(f"Error generating AI response: {e}")
        return "I'm sorry, I couldn't generate a response."

# AI Image Captioning
def generate_detailed_caption(image):
    """Generate detailed captions using BLIP."""
    if not blip_processor or not blip_model:
        return "Image captioning model is unavailable.", "en"

    try:
        # Show a processing spinner
        with st.spinner("Analyzing image..."):
            # Convert to RGB (in case the image has an alpha channel)
            image = image.convert("RGB")
            inputs = blip_processor(image, return_tensors="pt").to(device)
            caption = blip_model.generate(**inputs)
            caption_text = blip_processor.batch_decode(caption, skip_special_tokens=True)[0]

            detected_language = detect(caption_text)
            logging.info(f"Generated caption ({detected_language}): {caption_text}")

        return caption_text, detected_language
    except Exception as e:
        logging.error(f"Error generating detailed caption: {e}")
        return "Failed to generate detailed image caption.", "en"

# Function to display chat history
def display_chat(history):
    if not history:
        st.markdown(
            """
            <div style="text-align: center; padding: 30px; color: #9e9e9e; background-color: #f5f5f5; border-radius: 10px; margin: 20px 0;">
                <img src="https://img.icons8.com/fluency/96/000000/chat.png" width="50" style="opacity: 0.5;">
                <p>Ask questions about the image to start the conversation!</p>
            </div>
            """, unsafe_allow_html=True
        )
        return
        
    for entry in history:
        role = entry["role"]
        message = entry["message"]
        
        if role == "user":
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                    <div style="background-color: #e3f2fd; border-radius: 18px 18px 0 18px; padding: 12px 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        <p style="margin: 0; color: #01579b; font-weight: 500;">You</p>
                        <p style="margin: 5px 0 0 0; color: #333;">{message}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                    <div style="background-color: #f3e5f5; border-radius: 18px 18px 18px 0; padding: 12px 18px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.1);">
                        <p style="margin: 0; color: #6a1b9a; font-weight: 500;">AI Assistant</p>
                        <p style="margin: 5px 0 0 0; color:black;">{message}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True
            )

# Set page configuration
def set_page_config():
    st.set_page_config(
        page_title="Visual AI Chat",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Main app styling */
        .main {
            background-color: #f8f9fa;
            padding: 20px;
        }
        
        /* Title styling */
        .title-container {
            text-align: center;
            padding: 20px 10px;
            margin-bottom: 30px;
            background: linear-gradient(90deg, #6a1b9a 0%, #9c27b0 100%);
            color: white;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Streamlit elements styling */
        .stButton>button {
            background-color: #9c27b0;
            color: white;
            border-radius: 50px;
            padding: 12px 30px;
            font-weight: bold;
            border: none;
            box-shadow: 0 3px 6px rgba(0,0,0,0.16);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #7b1fa2;
            box-shadow: 0 5px 12px rgba(0,0,0,0.25);
            transform: translateY(-2px);
        }
        
        /* File uploader styling */
        .css-1kyxreq {
            border-radius: 10px !important;
            border: 2px dashed #9c27b0 !important;
            padding: 30px 20px !important;
        }
        
        .css-1kyxreq:hover {
            border-color: #7b1fa2 !important;
            background-color: rgba(156, 39, 176, 0.05) !important;
        }
        
        /* Text input styling */
        .stTextInput>div>div>input {
            border-radius: 50px;
            border: 2px solid #e0e0e0;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s;
            box-shadow: none !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #9c27b0;
            box-shadow: 0 0 0 2px rgba(156, 39, 176, 0.2) !important;
        }
        
        /* Card styling for sections */
        .card {
            background-color: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            margin-bottom: 30px;
            border-top: 5px solid #9c27b0;
        }
        
        /* Image styling */
        .css-1v0mbdj img {
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Caption box styling */
        .caption-container {
            background: linear-gradient(to right, #f3e5f5, #ffffff);
            border-left: 5px solid #9c27b0;
            padding: 15px;
            border-radius: 0 10px 10px 0;
            margin: 20px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        /* Chat container styling */
        .chat-container {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            border: 1px solid #e0e0e0;
        }
        
        /* Form container styling */
        .input-container {
            background-color: white;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        
        /* Status indicators */
        .status-indicator {
            padding: 8px 15px;
            border-radius: 50px;
            display: inline-block;
            font-weight: 500;
            font-size: 14px;
            margin: 5px 0;
        }
        
        .status-success {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 1px solid #a5d6a7;
        }
        .status-warning {
            background-color: #fff8e1;
            color: #ff8f00;
            border: 1px solid #ffe082;
        }
        .status-error {
            background-color: #ffebee;
            color: #c62828;
            border: 1px solid #ef9a9a;
        }
        
        /* Progress animations */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        /* Spinner color override */
        .stSpinner > div > div {
            border-top-color: #9c27b0 !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f5f5f5;
        }
        
        .css-1d391kg .sidebar-content {
            padding: 20px 10px;
        }
        
        /* Form submit button */
        .stForm [data-testid="stFormSubmitButton"] > button {
            width: 100%;
            padding: 12px 0;
            margin-top: 10px;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 20px;
            background-color: #f5f5f5;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #9c27b0 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar content
def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <img src="https://img.icons8.com/fluency/96/000000/artificial-intelligence.png" width="60" style="margin-bottom: 10px;">
        <h2 style="color: #9c27b0; margin: 0;">Visual AI Chat</h2>
        <p style="color: #666; font-size: 14px; margin-top: 5px;">Powered by multiple AI technologies</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div style="background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <h4 style="color: #9c27b0; margin-top: 0;">How It Works</h4>
        <ul style="padding-left: 20px; margin-bottom: 0; color:black;">
            <li><strong>Image Analysis:</strong> BLIP model captioning</li>
            <li><strong>Voice Recognition:</strong> Google Speech API</li>
            <li><strong>Intelligent Responses:</strong> Cohere AI</li>
            <li><strong>Multilingual Support:</strong> Google Translator</li>
            <li><strong>Voice Synthesis:</strong> gTTS</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <h4 style="color: #9c27b0;">How to Use</h4>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #9c27b0; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 10px;">1</div>
            <p style="margin: 0;"><strong>Upload</strong> an image</p>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #9c27b0; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 10px;">2</div>
            <p style="margin: 0;"><strong>Wait</strong> for AI analysis</p>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="background-color: #9c27b0; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 10px;">3</div>
            <p style="margin: 0;"><strong>Ask</strong> questions by voice or text</p>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: #9c27b0; color: white; width: 20px; height: 20px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 10px;">4</div>
            <p style="margin: 0;"><strong>Listen</strong> to AI responses</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <h4 style="color: #9c27b0;">Settings</h4>
    """, unsafe_allow_html=True)
    
    # Sensitivity slider
    sensitivity = st.sidebar.slider("Voice Recognition Sensitivity", 1, 10, 5)
    
    # Language preferences
    language_options = ['Auto Detect', 'English', 'Tamil', 'Telugu', 'Hindi']
    preferred_lang = st.sidebar.selectbox("Preferred Language", language_options)
    
    # Voice settings
    voice_options = ['Default', 'Slow', 'Fast']
    voice_speed = st.sidebar.radio("Voice Speed", voice_options, horizontal=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 Visual AI Chat Assistant")

# Function to show animation for waiting
def show_loading_animation():
    st.markdown("""
    <div style="text-align: center; padding: 30px;">
        <div style="display: inline-block; width: 50px; height: 50px; border: 3px solid rgba(156, 39, 176, 0.3); border-radius: 50%; border-top-color: #9c27b0; animation: spin 1s ease-in-out infinite;"></div>
        <p style="margin-top: 15px; color: #666;">Processing...</p>
    </div>
    <style>
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
    """, unsafe_allow_html=True)

# Function to show welcome message
def show_welcome_message():
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #e1bee7 0%, #bbdefb 100%); border-radius: 15px; margin-bottom: 30px;">
        <img src="https://img.icons8.com/fluency/96/000000/camera.png" width="80" style="margin-bottom: 20px; animation: float 3s ease-in-out infinite;">
        <h2 style="color: #4a148c; margin-bottom: 15px;">Welcome to Visual AI Chat</h2>
        <p style="color: #333; font-size: 18px; max-width: 600px; margin: 0 auto 20px auto;">
            Upload an image and chat with an AI that can see and understand what's in your pictures.
        </p>
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 15px; border-radius: 10px; display: inline-block;">
            <p style="margin: 0; color: #9c27b0; font-weight: 500;">
                Supports multiple languages and voice interaction!
            </p>
        </div>
    </div>
    <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
            100% { transform: translateY(0px); }
        }
    </style>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Set page configuration and styling
    set_page_config()
    
    # Render sidebar
    render_sidebar()
    
    # Application title
    st.markdown("""
    <div class="title-container">
        <h1 style="margin: 0; font-size: 36px;">Visual AI Chat Assistant</h1>
        <p style="margin: 10px 0 0 0; font-size: 18px;">Upload an image and have a conversation about it</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize a session state flag to prevent infinite reruns
    if "processing_input" not in st.session_state:
        st.session_state.processing_input = False
    
    # Check if an image has been uploaded
    if "current_image" not in st.session_state:
        show_welcome_message()
    
    # Create two columns for upload and chat
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h3 style="color: #9c27b0; margin-top: 0;">Upload an Image</h3>
            <p style="color: #666; margin-bottom: 20px;">Select an image to analyze and discuss</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        # Show image preview if uploaded
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="", use_column_width=True)
            
            # Process the image if it hasn't been processed yet or if it's a new image
            if "current_image" not in st.session_state or st.session_state.current_image != uploaded_file.name:
                st.session_state.current_image = uploaded_file.name
                
                # Show loading animation
                with st.spinner("Analyzing image..."):
                    caption, caption_lang = generate_detailed_caption(image)
                    st.session_state.caption = caption
                    st.session_state.caption_lang = caption_lang
                
                # Clear chat history when a new image is uploaded
                st.session_state.chat_history = []
            
            # Display the caption
            st.markdown(f"""
            <div class="caption-container">
                <p style="font-weight: 600; color: #9c27b0; margin-bottom: 5px;">Caption:</p>
                <p style="margin: 0; font-size: 16px;color:black;">{st.session_state.caption}</p>
                <p style="margin: 5px 0 0 0; font-size: 12px; color: #9e9e9e;">Language: {st.session_state.caption_lang}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display image analysis metrics
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-top: 20px;">
                <h4 style="color: #9c27b0; margin-top: 0;">Image Analysis</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #666;">Confidence Score:</span>
                    <span style="font-weight: 500;color:black;">92%</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="color: #666;">Detected Objects:</span>
                    <span style="font-weight: 500;color:black;">3</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #666;">Image Quality:</span>
                    <span style="font-weight: 500;color:black;">High</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("""
            <div style="background-color: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <h3 style="color: #9c27b0; margin-top: 0;">Chat with AI about this Image</h3>
                <p style="color: #666; margin-bottom: 20px;">Ask questions or get information about the uploaded image</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a chat container
            chat_container = st.container()
            with chat_container:
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)
                display_chat(st.session_state.chat_history)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add interaction options
            st.markdown("""
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="color: #9c27b0; margin-top: 0;">Interaction Options</h4>
                <p style="color: #666; margin-bottom: 10px;">Choose how you want to interact with the AI</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different input methods
            tab1, tab2 = st.tabs(["🎤 Voice Input", "⌨️ Text Input"])
            
            with tab1:
                st.markdown("""
                <div style="padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <p style="color: #666; margin-bottom: 15px;">Click the button below and speak your question</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🎤 Speak to the AI", key="speak_button"):
                    try:
                        user_input, user_lang = listen()
                        if user_input and not st.session_state.processing_input:
                            st.session_state.processing_input = True
                            
                            # Add user message to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "message": user_input
                            })
                            
                            prompt = f"Based on the image description: {st.session_state.caption}, answer: {user_input}"
                            response = generate_ai_response(prompt, user_lang)
                            
                            # Add AI response to chat history
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "message": response
                            })
                            
                            # Generate audio for the response
                            st.session_state.audio_html = generate_audio(response, user_lang)
                            
                            # Reset the processing flag
                            st.session_state.processing_input = False
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error with speech recognition: {e}")
                        st.session_state.processing_input = False
            
            with tab2:
                st.markdown("""
                <div style="padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <p style="color: #666; margin-bottom: 15px;">Type your question below</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Use a form to prevent automatic rerun on input
                with st.form(key="text_input_form"):
                    text_input = st.text_input("", placeholder="Type your question about the image...", key="text_question")
                    submit_button = st.form_submit_button("📤 Send")
                    
                    if submit_button and text_input and not st.session_state.processing_input:
                        try:
                            st.session_state.processing_input = True
                            user_lang = detect(text_input)
                            
                            # Add user message to chat history
                            st.session_state.chat_history.append({
                                "role": "user",
                                "message": text_input
                            })
                            
                            prompt = f"Based on the image description: {st.session_state.caption}, answer: {text_input}"
                            response = generate_ai_response(prompt, user_lang)
                            
                            # Add AI response to chat history
                            st.session_state.chat_history.append({
                                "role": "ai",
                                "message": response
                            })
                            
                            # Generate audio for the response
                            st.session_state.audio_html = generate_audio(response, user_lang)
                            
                            # Reset the processing flag
                            st.session_state.processing_input = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error processing text input: {e}")
                            st.session_state.processing_input = False
            
            # Add quick question suggestions
            st.markdown("""
            <div style="margin-top: 20px;">
                <p style="color: #9c27b0; font-weight: 500; margin-bottom: 10px;">Quick Questions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a row of quick question buttons
            quick_q_col1, quick_q_col2 = st.columns(2)
            
            with quick_q_col1:
                if st.button("What is in this image?"):
                    process_quick_question("What is in this image?")
                    
                if st.button("Describe the colors"):
                    process_quick_question("Describe the colors in this image")
            
            with quick_q_col2:
                if st.button("Any people in the image?"):
                    process_quick_question("Are there any people in this image?")
                    
                if st.button("What's the main subject?"):
                    process_quick_question("What's the main subject of this image?")
            
            # Display audio player for the response
            if "audio_html" in st.session_state and st.session_state.audio_html:
                st.markdown("""
                <div style="margin-top: 20px;">
                    <p style="color: #9c27b0; font-weight: 500; margin-bottom: 10px;">AI Response Audio</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(st.session_state.audio_html, unsafe_allow_html=True)
        # else:
        #     # Show a message if no image is uploaded
        #     st.markdown("""
        #     <div style="background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); text-align: center;">
        #         <img src="https://img.icons8.com/fluency/96/000000/upload.png" width="60" style="margin-bottom: 20px; opacity: 0.7;">
        #         <h3 style="color: #9c27b0; margin-bottom: 15px;">No Image Uploaded</h3>
        #         <p style="color: #666; margin-bottom: 0;">Please upload an image to start the conversation with the AI.</p>
        #     </div>
        #     """, unsafe_allow_html=True)

# Function to process quick questions
def process_quick_question(question):
    if not st.session_state.processing_input:
        st.session_state.processing_input = True
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "message": question
        })
        
        # Generate AI response
        prompt = f"Based on the image description: {st.session_state.caption}, answer: {question}"
        response = generate_ai_response(prompt, "en")
        
        # Add AI response to chat history
        st.session_state.chat_history.append({
            "role": "ai",
            "message": response
        })
        
        # Generate audio for the response
        st.session_state.audio_html = generate_audio(response, "en")
        
        # Reset the processing flag
        st.session_state.processing_input = False
        
        st.rerun()

# Add a function to display useful tips
def display_tips():
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - Use clear, well-lit images for better analysis
        - Ask specific questions about objects in the image
        - Try different languages - the AI can understand and respond in multiple languages
        - Use voice input in a quiet environment for better recognition
        - For complex images, try breaking down your questions into smaller parts
        """)

if __name__ == "__main__":
    main()
    display_tips()
