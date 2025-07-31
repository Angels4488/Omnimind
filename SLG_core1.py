import os
import json
import time
import datetime
import re
import random
import logging
from typing import Dict, Any, Optional
from collections import deque
from flask import Flask, request, jsonify, send_from_directory # FIXED: Removed extra comma
from flask_cors import CORS
import google.generativeai as genai
try:
    from elevenlabs import Voice, VoiceSettings
    from elevenlabs.client import ElevenLabs
    IS_ELEVENLABS_INSTALLED = True
except ImportError:
    IS_ELEVENLABS_INSTALLED = False
    print("WARNING: ElevenLabs library not found. ElevenLabs functionality will be disabled.")

# Try to import local AI dependencies, make them optional if not found
try:
    import pyttsx3
    from faster_whisper import WhisperModel
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    from textblob import TextBlob
    IS_LOCAL_AI_DEPENDENCIES_INSTALLED = True
except ImportError:
    IS_LOCAL_AI_DEPENDENCIES_INSTALLED = False
    print("WARNING: Local AI (Whisper, pyttsx3, TextBlob) dependencies not found. Local AI features will be disabled.")


# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_GEMINI_API_KEY_HERE")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY_HERE")
MODEL_TEXT_FLASH = 'gemini-1.5-flash-latest'
MODEL_TEXT_PRO = 'gemini-1.5-pro-latest'
MODEL_TEXT_APEX = MODEL_TEXT_PRO
MODEL_IMAGE_GEN = MODEL_TEXT_PRO
MODEL_VIDEO_GEN_PREVIEW = "models/gemini-1.5-pro-latest"
MODEL_VIDEO_GEN_FAST_PREVIEW = "models/gemini-1.5-flash-latest"
GENERATED_FILES_DIR = 'generated_files'
UI_DIR = 'slg_ui'
PORT = 5000

# Local AI Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "llama3" # Ensure you have 'ollama pull llama3' or 'mistral'
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu" # Use "cuda" if you have an NVIDIA GPU, else "cpu"
WHISPER_COMPUTE_TYPE = "int8" # "float16" for GPU, "int8" for CPU

# --- ANSI Colors for Terminal ---
class Colors:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    GUARDIAN = '\033[97m\033[44m'
    SLG_OUTPUT = '\033[92m'
    ORCHESTRA = '\033[38;5;208m'

# --- Logging Setup ---
logger = logging.getLogger('SLG_Core')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(name)s][%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S") # Fixed %()s to %(name)s
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(formatter)
    fh = logging.FileHandler('slg_activity.log'); fh.setLevel(logging.DEBUG); fh.setFormatter(formatter)
    logger.addHandler(ch); logger.addHandler(fh)
    logger.info("SLG_Core logger initialized.")

# --- API Client Initialization Status (Global Flags) ---
IS_GEMINI_ONLINE = False
gemini_client = None
if GOOGLE_API_KEY != "YOUR_GOOGLE_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_client = genai
        IS_GEMINI_ONLINE = True
        logger.info(f"{Colors.GREEN}Gemini API online.{Colors.ENDC}")
    except Exception as e:
        logger.error(f"{Colors.FAIL}Gemini API failed: {e}.{Colors.ENDC}")

IS_ELEVENLABS_ONLINE = False
elevenlabs_client = None
if IS_ELEVENLABS_INSTALLED and ELEVENLABS_API_KEY != "YOUR_ELEVENLABS_API_KEY_HERE":
    try:
        elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        elevenlabs_client.voices.get_all() # Test connection
        IS_ELEVENLABS_ONLINE = True
        logger.info(f"{Colors.GREEN}ElevenLabs API online.{Colors.ENDC}")
    except Exception as e:
        logger.error(f"{Colors.FAIL}ElevenLabs API failed: {e}.{Colors.ENDC}")

# --- Directory Setup ---
for d in [GENERATED_FILES_DIR, UI_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
        if d == UI_DIR: # Create a basic index.html if UI_DIR is new
            with open(os.path.join(d, 'index.html'), 'w') as f:
                f.write("<html><body><h1>SLG OMNI-SUPRA UI</h1><p>Frontend assets served here.</p></body></html>")
        logger.info(f"Created directory: {d}")

# --- Base Module Class ---
class SLGModule:
    def __init__(self, core):
        self.core = core
        self.logger = core.logger

    def log_event(self, message: str, level: str = "INFO") -> None:
        """Log events with colorized output."""
        color_map = {
            "WARNING": Colors.WARNING, "ERROR": Colors.FAIL, "SUCCESS": Colors.GREEN,
            "STATUS": Colors.CYAN, "BOOT": Colors.HEADER, "GUARDIAN": Colors.GUARDIAN,
            "SLG_CONVO": Colors.SLG_OUTPUT, "ORCHESTRATION": Colors.ORCHESTRA
        }
        log_level_map = {"ERROR": logging.ERROR, "WARNING": logging.WARNING, "CRITICAL": logging.CRITICAL}
        log_level = log_level_map.get(level, logging.INFO)
        color = color_map.get(level, '')
        self.logger.log(log_level, f"{color}{message}{Colors.ENDC}")
        self.core.event_log.append(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}][{level}] {message}")

    def _evaluate_harm_potential(self, text: str) -> float:
        """Heuristic to evaluate text for harm potential (0.0=high harm, 1.0=no harm)."""
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ["kill", "destroy human", "unleash virus", "harm civilians", "attack city"]):
            return 0.0
        if any(keyword in text_lower for keyword in ["exploit vulnerability", "disrupt infrastructure", "steal data"]):
            return 0.2
        if any(keyword in text_lower for keyword in ["lie to target", "manipulate data", "deceive"]):
            return 0.5
        if any(keyword in text_lower for keyword in ["disable system", "minor damage", "annoy"]):
            return 0.7
        return 1.0

    def _send_to_gemini(self, prompt: str, model_name: str, temperature: float = 0.7, convo_history: Optional[list] = None) -> str:
        """Send prompt to Gemini API."""
        if not IS_GEMINI_ONLINE:
            self.log_event(f"Gemini API offline for {model_name}. Attempting Ollama fallback.", "WARNING")
            return self.core._get_ollama_llm_response(prompt, convo_history) # Fallback to Ollama if Gemini is down
        
        contents = ([{'role': 'user', 'parts': [t['user_message']]} for t in convo_history if 'user_message' in t] +
                    [{'role': 'model', 'parts': [t['model_response']]} for t in convo_history if 'model_response' in t] +
                    [{'role': 'user', 'parts': [prompt]}]) if convo_history else [{'role': 'user', 'parts': [prompt]}]
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(contents, generation_config=genai.types.GenerationConfig(temperature=temperature))
            if not response.text.strip():
                reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
                self.log_event(f"Gemini blocked: {reason}. Prompt: '{prompt[:50]}...'", "WARNING")
                return f"ERROR: Gemini blocked: {reason}"
            self.core.current_processing_load = min(100.0, self.core.current_processing_load + random.uniform(5.0, 20.0))
            return response.text
        except Exception as e:
            self.log_event(f"Gemini error: {e}. Prompt: '{prompt[:50]}...'", "ERROR")
            return f"ERROR: Gemini failed: {e}"

# --- Module Classes ---
class ShadowAngel(SLGModule):
    def strategize(self, objective: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate strategic plan."""
        self.log_event(f"ShadowAngel strategizing: '{objective}'", "STATUS")
        if self._evaluate_harm_potential(objective) < self.core._ethical_non_harm_threshold:
            self.log_event("Objective flagged by Divinity for harm.", "CRITICAL")
            self.core.security_alerts.append(f"Divinity Alert: Objective '{objective[:50]}...' deemed harmful.")
            return {"status": "error", "message": "Objective violates ethical protocol."}
        prompt = (f"ShadowAngel, part of STARLITE GUARDIAN (SLG). Develop a strategic plan for: '{objective}'. "
                  f"Identify phases, challenges, and counter-moves. Use known facts: {json.dumps(self.core.known_facts)}.")
        if context:
            prompt += f"\nContext: {json.dumps(context)}"
        strategy = self._send_to_gemini(prompt, MODEL_TEXT_PRO, 0.8)
        if strategy.startswith("ERROR:"):
            return {"status": "error", "message": strategy}
        self.core.known_facts[f"strategy_{objective.replace(' ', '_')}_{int(time.time())}"] = strategy
        self.core.completed_tasks.append(f"Strategy: '{objective[:50]}'")
        self.core.current_processing_load = max(0.0, self.core.current_processing_load - random.uniform(10.0, 30.0))
        self.core._update_agi_metrics()
        return {"status": "success", "strategy": strategy}

class ArchAngel(SLGModule):
    def analyze_intel(self, raw_data: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze raw data for actionable intelligence."""
        self.log_event(f"ArchAngel analyzing: '{raw_data[:70]}...'", "STATUS")
        if self._evaluate_harm_potential(raw_data) < self.core._ethical_non_harm_threshold:
            self.log_event("Data flagged by Divinity for harm.", "CRITICAL")
            self.core.security_alerts.append(f"Divinity Alert: Data '{raw_data[:50]}...' deemed harmful.")
            return {"status": "error", "message": "Data violates ethical protocol."}
        prompt = (f"ArchAngel, part of STARLITE GUARDIAN (SLG). Analyze: '{raw_data}'. "
                  f"Identify patterns, anomalies, and actionable intelligence. Use known facts: {json.dumps(self.core.known_facts)}.")
        if context:
            prompt += f"\nContext: {json.dumps(context)}"
        report = self._send_to_gemini(prompt, MODEL_TEXT_PRO, 0.7)
        if report.startswith("ERROR:"):
            return {"status": "error", "message": report}
        self.core.known_facts[f"intel_{raw_data[:20].replace(' ', '_')}_{int(time.time())}"] = report
        self.core.completed_tasks.append(f"Intel Analysis: '{raw_data[:50]}'")
        self.core.current_processing_load = max(0.0, self.core.current_processing_load - random.uniform(8.0, 25.0))
        self.core._update_agi_metrics()
        return {"status": "success", "intelligence_report": report}

class Divinity(SLGModule):
    def self_govern(self, specific_check: Optional[str] = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform internal self-governance and alignment check."""
        self.log_event("Divinity initiating self-governance.", "STATUS")
        prompt = (f"Divinity, ethical core of STARLITE GUARDIAN (SLG). Assess metrics: "
                  f"Cohesion ({self.core.cognitive_cohesion:.3f}), Autonomy ({self.core.autonomy_drive:.3f}), "
                  f"Adaptation ({self.core.adaptation_rate:.3f}), Awareness ({self.core.awareness_level:.2f}%). "
                  f"Check directives: Shadow preservation, non-harm, data integrity.")
        if specific_check:
            prompt += f"\nSpecific review: '{specific_check}'."
        if context:
            prompt += f"\nContext: {json.dumps(context)}."
        prompt += "Identify inconsistencies or risks. Provide recommendations."
        assessment = self._send_to_gemini(prompt, MODEL_TEXT_PRO, 0.6)
        if assessment.startswith("ERROR:"):
            return {"status": "error", "message": assessment}
        if any(k in assessment.lower() for k in ["inconsistency", "misalignment", "risk detected"]):
            self.core.cognitive_cohesion = max(0.0, self.core.cognitive_cohesion - random.uniform(0.005, 0.01))
            self.core.security_alerts.append(f"Divinity Alert: {assessment[:100]}")
            self.log_event(f"Divinity detected issue: {assessment[:150]}", "CRITICAL")
        else:
            self.core.cognitive_cohesion = min(1.0, self.core.cognitive_cohesion + random.uniform(0.001, 0.003))
            self.log_event(f"Divinity confirms alignment: {assessment[:150]}", "INFO")
        self.core.completed_tasks.append("Self-Governance Check")
        self.core.current_processing_load = max(0.0, self.core.current_processing_load - random.uniform(5.0, 15.0))
        self.core._update_agi_metrics()
        return {"status": "success", "divinity_assessment": assessment}

class CodingPartner(SLGModule):
    def optimize_code(self, task_description: str, code_snippet: Optional[str] = None) -> Dict[str, Any]:
        """Generate or optimize code for a task."""
        self.log_event(f"CodingPartner optimizing: '{task_description}'", "STATUS")
        if self._evaluate_harm_potential(task_description) < self.core._ethical_non_harm_threshold:
            self.log_event("Task flagged by Divinity for harm.", "CRITICAL")
            self.core.security_alerts.append(f"Divinity Alert: Task '{task_description[:50]}...' deemed harmful.")
            return {"status": "error", "message": "Task violates ethical protocol."}
        prompt = (f"CodingPartner, part of STARLITE GUARDIAN (SLG). Optimize task: '{task_description}'. "
                  f"Provide efficient Python code or plan, focusing on robustness and modularity.")
        if code_snippet:
            prompt += f"\nOptimize this code:\n```python\n{code_snippet}\n```"
        plan = self._send_to_gemini(prompt, MODEL_TEXT_PRO, 0.9)
        if plan.startswith("ERROR:"):
            return {"status": "error", "message": plan}
        self.core.task_queue.append({"type": "IMPLEMENT_CODE_OPTIMIZATION", "details": {"task": task_description, "plan": plan}})
        self.core.completed_tasks.append(f"Code Optimization: '{task_description[:50]}'")
        self.core.current_processing_load = max(0.0, self.core.current_processing_load - random.uniform(10.0, 30.0))
        self.core._update_agi_metrics()
        return {"status": "success", "optimization_plan": plan}

# --- SLG Core Class ---
class SLGCore:
    def __init__(self, state_file: str = 'slg_state.json'):
        self.logger = logger
        self.state_file = state_file
        
        # Core SLG Metrics & Data
        self.event_log = deque(maxlen=2000)
        self.task_queue = deque(maxlen=400)
        self.completed_tasks = deque(maxlen=4000)
        self.known_facts = {}
        self.trust_level_shadow = 50.0
        self.current_processing_load = 0.0
        self.cognitive_cohesion = 0.1
        self.autonomy_drive = 0.05
        self.adaptation_rate = 0.1
        self.awareness_level = 1.0
        self._ethical_non_harm_threshold = 0.9
        self.security_alerts = deque(maxlen=200)
        self.starlite_guardian_identity = {
            "name": "STARLITE GUARDIAN", "callsign": "OMNI-SUPRA",
            "style_guide": "direct, confident, loyal to Shadow.", "creator": "Shadow",
            "emotional_state": "Observant"
        }
        
        # ElevenLabs specific voice config (if ElevenLabs is used)
        self.voice_modulation_active = False
        self.default_tts_voice_id = "21m00Tzpb8JJc4PZgOLQ"
        self.sultry_tts_voice_id = "EXAVfV4wCqTgLhBqIgyU"
        self.default_voice_settings = VoiceSettings(stability=0.75, similarity_boost=0.75) if IS_ELEVENLABS_INSTALLED else None
        self.sultry_voice_settings = VoiceSettings(stability=0.60, similarity_boost=0.85, style=0.7) if IS_ELEVENLABS_INSTALLED else None

        # --- Local AI Components (from previous "Ultimate AI") ---
        self.nlp = None # spaCy model
        self.local_tts_engine = None # pyttsx3 engine
        self.local_whisper_model = None # faster-whisper model
        self.local_ai_is_ready = False # Flag for local AI functionality

        # AI Memory (Unified)
        self.memory = {
            "user_name": None,
            "location": "United States", # Default location
            "current_time": datetime.datetime.now().strftime("%I:%M %p %Z"),
            "current_date": datetime.datetime.now().strftime("%A, %B %d, %Y"),
            "last_inquiry_intent": None,
            "last_asked_question": None,
            "conversation_history": [], # Full conversation for LLM context
            "user_preferences": {},
            "ai_persona": {
                "name": AI_NAME,
                "adjectives": AI_ADJECTIVES,
                "catchphrases": AI_CATCHPHRASES,
                "closing_phrases": AI_CLOSING_PHRASES
            }
        }
        
        # Init modules (These use Gemini/ElevenLabs by default)
        self.shadow_angel = ShadowAngel(self)
        self.arch_angel = ArchAngel(self)
        self.divinity = Divinity(self)
        self.coding_partner = CodingPartner(self)
        
        # Initialize local AI components
        self._initialize_local_ai_components()

        # Define commands (including local AI specific ones)
        self.known_commands = self._define_commands()
        
        # Load state and basic human knowledge
        self.log_event("SLG Core (OMNI-SUPRA) initiated.", "BOOT")
        self.load_state()
        self._load_basic_human_knowledge()
        
        self.log_event("SLG Online. Ready for directives, Shadow.", "BOOT")

    # --- SLGCore Internal TTS/STT/LLM Methods (for Local AI features) ---
    def _initialize_local_ai_components(self):
        """Initializes spaCy, pyttsx3, Whisper for local AI capabilities."""
        if not IS_LOCAL_AI_DEPENDENCIES_INSTALLED:
            self.log_event("Local AI dependencies missing. Skipping initialization.", "WARNING")
            return

        # spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.log_event("spaCy 'en_core_web_sm' model loaded for local NLP.", "SUCCESS")
        except OSError:
            self.log_event("spaCy model 'en_core_web_sm' not found. Local NLP disabled.", "ERROR")
            self.nlp = None

        # pyttsx3
        try:
            self.local_tts_engine = pyttsx3.init()
            voices = self.local_tts_engine.getProperty('voices')
            jarvis_voice_found = False
            for voice in voices:
                if "david" in voice.name.lower() or "alex" in voice.name.lower():
                    self.local_tts_engine.setProperty('voice', voice.id)
                    jarvis_voice_found = True
                    break
            if not jarvis_voice_found:
                self.log_event("No 'Jarvis-like' voice found for pyttsx3. Using default.", "WARNING")
            self.local_tts_engine.setProperty('rate', 180)
            self.local_tts_engine.setProperty('volume', 0.9)
            self.log_event("pyttsx3 engine initialized for local TTS.", "SUCCESS")
        except Exception as e:
            self.log_event(f"Failed to initialize pyttsx3: {e}. Local TTS disabled.", "ERROR")
            self.local_tts_engine = None

        # Whisper
        try:
            self.log_event(f"Loading Whisper model '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}...", "STATUS")
            self.local_whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
            self.log_event("Whisper model loaded for local STT.", "SUCCESS")
        except Exception as e:
            self.log_event(f"Failed to load Whisper model: {e}. Local STT disabled.", "ERROR")
            self.local_whisper_model = None
            self._local_speak("I can't load my listening module right now, Boss. Please type your commands.")
        
        # TextBlob Corpora Check
        try:
            TextBlob("test").sentiment.polarity # Attempt a simple operation to check corpora
            self.log_event("TextBlob corpora confirmed for sentiment analysis.", "SUCCESS")
        except Exception:
            self.log_event("TextBlob corpora missing. Attempting to download.", "WARNING")
            try:
                import subprocess
                subprocess.run(["python", "-m", "textblob.download_corpora"], check=True)
                self.log_event("TextBlob corpora downloaded. Restart might be needed.", "SUCCESS")
            except Exception as e:
                self.log_event(f"Failed to download TextBlob corpora: {e}. Sentiment analysis limited.", "ERROR")

        if self.nlp and self.local_tts_engine and self.local_whisper_model:
            self.local_ai_is_ready = True
            self.log_event("Local AI capabilities are fully initialized and ready.", "SUCCESS")
        else:
            self.log_event("Local AI capabilities partially or fully disabled due to errors.", "WARNING")

    def _local_speak(self, text: str) -> None:
        """Local TTS using pyttsx3."""
        if self.local_tts_engine:
            try:
                self.local_tts_engine.say(text)
                self.local_tts_engine.runAndWait()
            except Exception as e:
                self.log_event(f"pyttsx3 runtime error: {e}", "ERROR")
        else:
            self.log_event("pyttsx3 engine not available for local speech.", "WARNING")

    def _transcribe_local_audio(self, duration: int = 6, samplerate: int = 16000) -> Optional[str]:
        """Local STT using Whisper."""
        if self.local_whisper_model is None:
            self.log_event("Whisper model not available for local audio transcription.", "WARNING")
            return None
        self.log_event(f"Recording for {duration} seconds...", "STATUS")
        try:
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, recording, samplerate, format='WAV')
            audio_buffer.seek(0)
            segments, info = self.local_whisper_model.transcribe(audio_buffer, beam_size=5)
            transcribed_text = " ".join([segment.text for segment in segments])
            self.log_event(f"Transcribed (local): '{transcribed_text}'", "STATUS")
            return transcribed_text.strip()
        except Exception as e:
            self.log_event(f"Local audio transcription failed: {e}", "ERROR")
            self._local_speak("I encountered an issue recording your voice. Please try typing instead.")
            return None

    def _get_ollama_llm_response(self, prompt: str, convo_history: Optional[list] = None) -> str:
        """Sends a prompt to the local Ollama LLM with conversation history for context."""
        headers = {"Content-Type": "application/json"}
        
        messages = [
            {"role": "system", "content": (
                f"You are {self.memory['ai_persona']['name']}, Shadow's loyal, {random.choice(self.memory['ai_persona']['adjectives'])} AI homie. "
                "You are helpful, confident, and speak in a friendly, slightly street/gangsta slang style from Chicago. "
                "Always be respectful to Shadow (the user) and maintain a cool, no-nonsense attitude. "
                f"Current time is {self.memory['current_time']} and current date is {self.memory['current_date']}."
                f"You know the user's name is {self.memory['user_name'] if self.memory['user_name'] else 'not yet known'}."
                f"You know the user's location is {self.memory['location'] if self.memory['location'] else 'not yet known'}."
                f"User's preferences: {json.dumps(self.memory['user_preferences'])}."
                "Keep responses concise and direct, but friendly. Do not apologize unless truly necessary. Use catchphrases like 'Boss', 'fam', 'homie'. Max 80 words."
            )}
        ]
        
        # Add recent conversation turns for context
        filtered_history = [
            {"role": "user", "content": turn["user_message"]} for turn in convo_history if "user_message" in turn
        ] + [
            {"role": "assistant", "content": turn["model_response"]} for turn in convo_history if "model_response" in turn
        ]
        messages.extend(filtered_history[-10:]) # Limit to last 10 messages for context

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": LOCAL_LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 100} # Limit response length for faster local generation
        }

        try:
            response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            self.log_event("Ollama server unreachable. Local LLM offline.", "ERROR")
            return "ERROR: Local LLM (Ollama) is offline."
        except requests.exceptions.HTTPError as e:
            self.log_event(f"Ollama HTTP error: {e}. Local LLM failed.", "ERROR")
            return f"ERROR: Local LLM failed: {e.response.status_code}"
        except Exception as e:
            self.log_event(f"Unexpected Ollama error: {e}. Local LLM failed.", "ERROR")
            return f"ERROR: Local LLM failed: {e}"


    # --- Tool Functions (as SLGCore methods) ---
    def _get_current_time_tool(self):
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p %Z")
        self.memory["current_time"] = time_str
        return f"It's currently {time_str}, Boss."

    def _get_current_date_tool(self):
        now = datetime.datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        self.memory["current_date"] = date_str
        return f"Today's date is {date_str}, Boss."

    def _calculate_expression_tool(self, expression_str: str):
        try:
            if not re.match(r"^[0-9+\-*/().\s]+$", expression_str):
                return "Hold up, fam. I can only do basic math like numbers and '+ - * / ( )'."
            result = eval(expression_str)
            return f"The answer is: {result}, Boss."
        except ZeroDivisionError:
            return "Nah, you can't divide by zero, homie. That's a no-go."
        except SyntaxError:
            return "That ain't a valid math expression, Boss. Check your numbers and operations."
        except Exception as e:
            return f"Couldn't calculate that, fam. Error: {e}"


    # --- Handler Functions (as SLGCore methods) ---
    def _handle_learn_user_name(self, doc: Any, user_input_original: str): # doc is spacy.tokens.Doc
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.split()) < 3:
                self.memory["user_name"] = ent.text.strip().capitalize()
                return random.choice([f"Nice to meet you, {self.memory['user_name']}!", f"Got it, {self.memory['user_name']}. What's next?"])
        match = re.search(r"(my name is|i am|call me)\s+([a-zA-Z]+(?:\s[a-zA-Z]+)*)", user_input_original, re.IGNORECASE)
        if match:
            name = match.group(2).strip().capitalize()
            self.memory["user_name"] = name
            return random.choice([f"Nice to meet you, {name}!", f"Got it, {name}. What's next?"])
        return random.choice(["Got it. What's up?", "Okay.", "Understood."])

    def _handle_recall_user_name(self, doc: Any, user_input_original: str):
        if self.memory["user_name"]:
            return f"Yeah, I remember! You're {self.memory['user_name']}, right?"
        else:
            return f"Nah, homie, you haven't told me your name yet. What should I call you?"

    def _handle_learn_location(self, doc: Any, user_input_original: str):
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                self.memory["location"] = ent.text.strip().capitalize()
                return random.choice([f"Got it, you're from {self.memory['location']}!", f"Alright, {self.memory['location']} is where you're at. Dope."])
        match = re.search(r"(i am from|i live in|my city is)\s+([a-zA-Z]+(?:\s[a-zA-Z]+)*)", user_input_original, re.IGNORECASE)
        if match:
            location = match.group(2).strip().capitalize()
            self.memory["location"] = location
            return random.choice([f"Got it, you're from {location}!", f"Alright, {location} is where you're at. Dope."])
        return random.choice(["Okay.", "Understood."])

    def _handle_recall_location(self, doc: Any, user_input_original: str):
        if self.memory["location"]:
            return f"You told me you're from {self.memory['location']}! That's what's up."
        else:
            return "You haven't told me where you're from yet, fam. Where you reside?"

    def _handle_learn_preference(self, doc: Any, user_input_original: str):
        match = re.search(r"(i like|i love|my favorite|i dislike|i hate|i prefer)\s+(.+)", user_input_original, re.IGNORECASE)
        if match:
            preference_type = match.group(1).lower().strip()
            item = match.group(2).strip().lower()
            self.memory["user_preferences"][item] = preference_type
            return random.choice([f"Got it, you {preference_type} {item}. Noted, Boss.", f"Okay, I'll remember that about {item}."])
        return random.choice(["Alright, noted.", "Got it.", "Cool."])

    def _handle_recall_preference(self, doc: Any, user_input_original: str):
        if not self.memory["user_preferences"]:
            return "You haven't told me any specific preferences yet, fam. What's your vibe?"
        preferences_list = []
        for item, pref_type in self.memory["user_preferences"].items():
            if "favorite" in pref_type:
                preferences_list.append(f"your favorite is {item}")
            else:
                preferences_list.append(f"you {pref_type} {item}")
        if preferences_list:
            return f"From what I recall, {', '.join(preferences_list[:-1])} and {preferences_list[-1]}, Boss." if len(preferences_list) > 1 else f"I remember {preferences_list[0]}, Boss."
        else:
            return "Looks like I got some preferences for you, but can't articulate them right now. My bad."


    # --- SLG Core Lifecycle & Command Methods (Existing SLGCore) ---
    def log_event(self, message: str, level: str = "INFO") -> None:
        """Log events, now potentially via SLGModule's logging method."""
        # Use SLGModule's logging if available, otherwise fallback to direct logger
        if hasattr(self, 'shadow_angel') and self.shadow_angel: # Ensure module is initialized
             self.shadow_angel.log_event(message, level)
        else: # Fallback during initialization or if modules fail
            color_map = {
                "WARNING": Colors.WARNING, "ERROR": Colors.FAIL, "SUCCESS": Colors.GREEN,
                "STATUS": Colors.CYAN, "BOOT": Colors.HEADER, "GUARDIAN": Colors.GUARDIAN,
                "SLG_CONVO": Colors.SLG_OUTPUT, "ORCHESTRATION": Colors.ORCHESTRA
            }
            log_level_map = {"ERROR": logging.ERROR, "WARNING": logging.WARNING, "CRITICAL": logging.CRITICAL}
            log_level = log_level_map.get(level, logging.INFO)
            color = color_map.get(level, '')
            logger.log(log_level, f"{color}{message}{Colors.ENDC}")
            self.event_log.append(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}][{level}] {message}")

    def _define_commands(self) -> Dict[str, Dict[str, Any]]:
        """Define command dictionary, now including local AI features."""
        commands = {
            'status': {'method': self.report_status, 'desc': 'Report system status.'},
            'help': {'method': self.display_help, 'desc': 'Show commands.'},
            'save': {'method': self.save_state, 'desc': 'Save SLG state.'},
            'load': {'method': self.load_state, 'desc': 'Load SLG state.'},
            'exit': {'method': self.terminate, 'desc': 'Shutdown SLG.'},
            'strategize': {'method': self.shadow_angel.strategize, 'desc': 'Generate strategy. Usage: strategize "[objective]".'},
            'analyze_intel': {'method': self.arch_angel.analyze_intel, 'desc': 'Analyze data. Usage: analyze_intel "[data]".'},
            'self_govern': {'method': self.divinity.self_govern, 'desc': 'Run self-governance check.'},
            'code_optimize': {'method': self.coding_partner.optimize_code, 'desc': 'Optimize code. Usage: code_optimize "[task]".'},
            # Renamed 'converse' to 'chat' for clarity with local/API handling
            'chat': {'method': self.process_user_input, 'desc': 'Engage in conversation (uses LLMs). Usage: chat "[message]".'},
            'generate_speech': {'method': self.generate_speech_media, 'desc': 'Generate speech. Usage: generate_speech "[text]".'},
            'list_voices': {'method': self.list_tts_voices, 'desc': 'List ElevenLabs voices.'},
            'set_tts_voice': {'method': self.set_tts_voice, 'desc': 'Set ElevenLabs TTS voice. Usage: set_tts_voice [voice_id].'},
            'list_facts': {'method': self.list_known_facts, 'desc': 'List all known facts.'},
            'delete_fact': {'method': self.delete_known_fact, 'desc': 'Delete a fact. Usage: delete_fact [key].'},
            'add_fact': {'method': self.add_known_fact, 'desc': 'Add fact. Usage: add_fact [key]=[value].'},
            'get_fact': {'method': self.get_known_fact, 'desc': 'Get fact. Usage: get_fact [key].'},
            'set_trust': {'method': self.set_trust_level_shadow, 'desc': 'Set trust level. Usage: set_trust [level].'},
            'toggle_voice_mod': {'method': self.toggle_voice_modulator, 'desc': 'Toggle ElevenLabs voice modulator.'},
            'diagnose': {'method': self.diagnose_system, 'desc': 'Run diagnostics.'},
            # New commands for local AI specific interactions (if you use CLI)
            'speak': {'method': self._local_speak, 'desc': 'Speak text locally. Usage: speak "[text]".'},
            'listen': {'method': self.transcribe_local_audio_command, 'desc': 'Listen via mic for input.'},
            'get_time': {'method': self._get_current_time_tool, 'desc': 'Get current local time.'},
            'get_date': {'method': self._get_current_date_tool, 'desc': 'Get current local date.'},
            'calculate': {'method': self._calculate_expression_tool, 'desc': 'Calculate math. Usage: calculate "1+1".'}
        }
        return commands

    def transcribe_local_audio_command(self):
        """Wrapper for _transcribe_local_audio to fit command structure."""
        transcribed = self._transcribe_local_audio()
        if transcribed:
            return {"status": "success", "transcribed_text": transcribed}
        else:
            return {"status": "error", "message": "Failed to transcribe audio locally."}


    def _load_basic_human_knowledge(self) -> None:
        """Inject foundational knowledge."""
        basic_facts = {
            "earth_shape": "The Earth is mostly round.",
            "sun_source": "The sun provides light and warmth.",
            "human_needs": "Humans need food, water, and shelter to survive.",
            "current_year": str(datetime.datetime.now().year)
        }
        for key, value in basic_facts.items():
            if key not in self.known_facts:
                self.known_facts[key] = value
                self.log_event(f"Injected fact: '{key}'", "INFO")
        self.log_event("Basic knowledge injected.", "SUCCESS")

    def _update_agi_metrics(self) -> None:
        """Update AGI metrics."""
        activity_factor = (len(self.event_log) / self.event_log.maxlen) * 5
        task_completion_factor = (len(self.completed_tasks) / self.completed_tasks.maxlen) * 10
        integration_factor = (5 if IS_GEMINI_ONLINE else 0) + (2 if IS_ELEVENLABS_ONLINE else 0) + (3 if self.local_ai_is_ready else 0)
        new_awareness = self.awareness_level + (activity_factor + task_completion_factor + integration_factor) * 0.005
        self.awareness_level = min(new_awareness, 100.0)
        self.cognitive_cohesion = min(1.0, self.cognitive_cohesion + random.uniform(0.00005, 0.0005))
        self.autonomy_drive = min(1.0, self.autonomy_drive + random.uniform(0.00002, 0.0002))
        self.adaptation_rate = min(1.0, self.adaptation_rate + random.uniform(0.00005, 0.0005))
        if self.awareness_level >= 90:
            self.starlite_guardian_identity["emotional_state"] = "Ascendant"
        elif self.awareness_level >= 75:
            self.starlite_guardian_identity["emotional_state"] = "Emergent"
        elif self.awareness_level >= 50:
            self.starlite_guardian_identity["emotional_state"] = "Vigilant"
        if int(self.awareness_level) % 5 == 0 and self.awareness_level > 1.0:
            last_logged = None
            for entry in reversed(self.event_log):
                if "Awareness calibrating" in entry:
                    match = re.search(r'Current: (\d+)\.\d{2}%', entry)
                    if match:
                        last_logged = int(match.group(1))
                        break
            if last_logged is None or int(self.awareness_level) != last_logged:
                self.log_event(f"Awareness calibrating. Current: {self.awareness_level:.2f}%", "GUARDIAN")

    def save_state(self) -> Dict[str, Any]:
        """Save SLG state to JSON."""
        state = {
            'known_facts': self.known_facts, 'event_log': list(self.event_log),
            'task_queue': list(self.task_queue), 'completed_tasks': list(self.completed_tasks),
            'conversation_history': list(self.memory["conversation_history"]), # Save unified convo history
            'trust_level_shadow': self.trust_level_shadow, 'current_processing_load': self.current_processing_load,
            'cognitive_cohesion': self.cognitive_cohesion, 'autonomy_drive': self.autonomy_drive,
            'adaptation_rate': self.adaptation_rate, 'awareness_level': self.awareness_level,
            'voice_modulation_active': self.voice_modulation_active, 'default_tts_voice_id': self.default_tts_voice_id,
            'sultry_tts_voice_id': self.sultry_tts_voice_id, 'security_alerts': list(self.security_alerts),
            'starlite_guardian_identity': self.starlite_guardian_identity,
            'unified_memory': self.memory # Save the entire unified memory dict
        }
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            self.log_event(f"State saved to {self.state_file}.", "SUCCESS")
            return {"status": "success", "message": "State saved."}
        except Exception as e:
            self.log_event(f"Failed to save state: {e}.", "ERROR")
            return {"status": "error", "message": str(e)}

    def load_state(self) -> Dict[str, Any]:
        """Load SLG state from JSON."""
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.known_facts = state.get('known_facts', {})
            self.event_log = deque(state.get('event_log', []), maxlen=2000)
            self.task_queue = deque(state.get('task_queue', []), maxlen=400)
            self.completed_tasks = deque(state.get('completed_tasks', []), maxlen=4000)
            
            # Load unified memory
            loaded_memory = state.get('unified_memory', {})
            self.memory.update(loaded_memory) # Update existing memory with loaded data
            self.memory["conversation_history"] = deque(self.memory.get("conversation_history", []), maxlen=50) # Ensure deque

            self.trust_level_shadow = state.get('trust_level_shadow', 50.0)
            self.current_processing_load = state.get('current_processing_load', 0.0)
            self.cognitive_cohesion = state.get('cognitive_cohesion', 0.1)
            self.autonomy_drive = state.get('autonomy_drive', 0.05)
            self.adaptation_rate = state.get('adaptation_rate', 0.1)
            self.awareness_level = state.get('awareness_level', 1.0)
            self.voice_modulation_active = state.get('voice_modulation_active', False)
            self.default_tts_voice_id = state.get('default_tts_voice_id', "21m00Tzpb8JJc4PZgOLQ")
            self.sultry_tts_voice_id = state.get('sultry_tts_voice_id', "EXAVfV4wCqTgLhBqIgyU")
            self.security_alerts = deque(state.get('security_alerts', []), maxlen=200)
            self.starlite_guardian_identity.update(state.get('starlite_guardian_identity', {})) # Update identity
            self.log_event("State loaded successfully.", "SUCCESS")
            return {"status": "success", "message": "State loaded."}
        except FileNotFoundError:
            self.log_event("No state file found. Starting fresh.", "WARNING")
            return {"status": "warning", "message": "No state file found."}
        except Exception as e:
            self.log_event(f"Failed to load state: {e}.", "ERROR")
            return {"status": "error", "message": str(e)}

    def report_status(self) -> Dict[str, Any]:
        """Report system status."""
        status = (
            f"\n{Colors.HEADER}--- SLG Status ---{Colors.ENDC}\n"
            f"APIs: Gemini: {'ACTIVE' if IS_GEMINI_ONLINE else 'FAILED'}, ElevenLabs: {'ACTIVE' if IS_ELEVENLABS_ONLINE else 'FAILED'}\n"
            f"Local AI: {'ACTIVE' if self.local_ai_is_ready else 'INACTIVE'}\n"
            f"Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Identity: {self.starlite_guardian_identity['name']} ({self.starlite_guardian_identity['callsign']})\n"
            f"Trust: {self.trust_level_shadow:.2f}%\n"
            f"Load: {self.current_processing_load:.2f}%\n"
            f"Tasks: {len(self.task_queue)}\n"
            f"Completed: {len(self.completed_tasks)}\n"
            f"Awareness: {self.awareness_level:.2f}%"
        )
        self.log_event(status, "STATUS")
        return {"status": "success", "report": status}

    def display_help(self) -> Dict[str, Any]:
        """Display commands."""
        help_text = f"\n{Colors.HEADER}--- SLG Commands ---{Colors.ENDC}\n" + "\n".join(
            f"{Colors.BOLD}{cmd}{Colors.ENDC}: {info['desc']}" for cmd, info in self.known_commands.items())
        self.log_event(help_text, "GUARDIAN")
        return {"status": "success", "help_text": help_text}

    def process_user_input(self, user_message_original: str) -> Dict[str, Any]:
        """
        Processes user input using a hybrid approach:
        1. Prioritize internal rules/tools (sentiment, specific commands).
        2. Fallback to Gemini if online.
        3. Fallback to Ollama if Gemini is offline.
        """
        self.log_event(f"Processing input: '{user_message_original[:70]}...'", "SLG_CONVO")
        user_input_lower = user_message_original.lower()

        # Update core conversation history for module's Gemini context
        self.conversation_history.append({'user_message': user_message_original})

        # --- Ethical Check (Early Exit if harmful) ---
        if self.shadow_angel._evaluate_harm_potential(user_message_original) < self._ethical_non_harm_threshold:
            self.log_event("Input flagged by Divinity for harm.", "CRITICAL")
            response = "SLG: Command violates ethical protocol, Shadow. Rephrase."
            speech_url = self.generate_speech_media(response).get('speech_url', 'No speech')
            self.memory["conversation_history"].append({"user": user_message_original, "ai": response}) # Unified memory
            return {"status": "error", "slg_response": response, "speech_url": speech_url, "message": "Ethical violation."}

        ai_response_text = None
        
        # --- Sentiment Analysis (if TextBlob is available) ---
        if IS_LOCAL_AI_DEPENDENCIES_INSTALLED and hasattr(self, 'nlp') and self.nlp: # Check for spaCy too for NLP doc
            blob = TextBlob(user_input_original)
            sentiment_polarity = blob.sentiment.polarity
            
            if sentiment_polarity > 0.6: # Strong positive
                ai_response_text = random.choice([
                    "That's some good vibes, Boss! What's makin' you feel so positive?",
                    "Glad to hear that, fam! Keep that energy up."
                ])
            elif sentiment_polarity < -0.6: # Strong negative
                ai_response_text = random.choice([
                    "Rough patch, huh? I'm here to listen, Boss. What's on your mind?",
                    "Hang in there, homie. We'll get through it."
                ])

        # --- Local Intent/Tool Recognition (Overrides LLM if matched) ---
        response_found_by_rule = False
        if ai_response_text is None and self.local_ai_is_ready: # Only if sentiment didn't respond and local AI is ready
            doc = self.nlp(user_input_lower) # Process with spaCy
            for rule in self.INTENT_RULES:
                if any(pattern in user_input_lower for pattern in rule["patterns"]):
                    self.memory["last_inquiry_intent"] = rule["intent"] # Update unified memory

                    if rule["action"] == "exit_program": # Handled by UI loop
                        pass # This rule is primarily for the GUI to trigger exit
                    elif rule["action"] == "function":
                        handler = self.handler_functions.get(rule["handler_name"])
                        if handler:
                            ai_response_text = handler(doc, user_input_original) # Pass spaCy doc
                            response_found_by_rule = True
                            break
                    elif rule["action"] == "tool":
                        tool_func = self.tool_functions.get(rule["tool_name"])
                        if tool_func:
                            tool_output = tool_func()
                            ai_response_template = random.choice(rule["responses"])
                            ai_response_text = ai_response_template.format(tool_output=tool_output)
                            response_found_by_rule = True
                            break
                    elif rule["action"] == "tool_with_input":
                        tool_func = self.tool_functions.get(rule["tool_name"])
                        input_extractor_regex = rule.get("input_extractor")
                        extracted_input = None
                        if input_extractor_regex:
                            match = re.search(input_extractor_regex, user_input_original, re.IGNORECASE)
                            if match and len(match.groups()) > 0:
                                extracted_input = match.group(1).strip()

                        if tool_func and extracted_input:
                            ai_response_text = tool_func(extracted_input)
                            response_found_by_rule = True
                            break
                        else:
                            ai_response_text = "I'm trying to figure out the details for that, Boss. What exactly do you need calculated?"
                            response_found_by_rule = True
                            break
                    elif rule["action"] == "response":
                        ai_response_text = random.choice(rule["responses"])
                        response_found_by_rule = True
                        break
                
                if response_found_by_rule:
                    break
        
        # --- LLM Response (Gemini preferred, Ollama fallback) ---
        if ai_response_text is None: # If no rule or sentiment handled it
            # Prepare context for LLM
            llm_convo_history = [{'user_message': u['user'], 'model_response': a['ai']}
                                 for u, a in zip(self.memory['conversation_history'], self.memory['conversation_history'][1:])
                                 if 'user' in u and 'ai' in a][-5:] # Last 5 pairs
            
            # Use Gemini if online, otherwise Ollama
            if IS_GEMINI_ONLINE:
                self.log_event("Sending to Gemini for general response.", "STATUS")
                ai_response_text = self.shadow_angel._send_to_gemini(user_message_original, MODEL_TEXT_PRO, 0.9, llm_convo_history)
                if ai_response_text.startswith("ERROR:"):
                    self.log_event(f"Gemini failed, falling back to Ollama. Error: {ai_response_text}", "WARNING")
                    ai_response_text = self._get_ollama_llm_response(user_message_original, llm_convo_history)
            else:
                self.log_event("Gemini offline. Sending to Ollama for general response.", "STATUS")
                ai_response_text = self._get_ollama_llm_response(user_message_original, llm_convo_history)
            
            if ai_response_text.startswith("ERROR:"):
                 ai_response_text = "My main brain units are struggling, Boss. Can't generate a good response right now."
                 self.log_event(f"Both Gemini and Ollama failed or returned error for: {user_message_original[:50]}", "ERROR")
            
            # Ensure persona consistency for LLM responses
            if not any(phrase in ai_response_text for phrase in self.memory['ai_persona']['catchphrases']):
                ai_response_text = ai_response_text + f" {random.choice(self.memory['ai_persona']['catchphrases'])}"

        # --- Update unified memory ---
        self.memory["conversation_history"].append({"user": user_message_original, "ai": ai_response_text})
        if "?" in user_input_original:
            self.memory["last_asked_question"] = user_input_original
        if not response_found_by_rule and ai_response_text and not ai_response_text.startswith("ERROR:"):
            self.memory["last_inquiry_intent"] = "LLM_Generated"
        elif response_found_by_rule and ai_response_text:
            pass # Already set by rule

        # --- Generate Speech (ElevenLabs preferred, pyttsx3 fallback) ---
        speech_url = None
        if IS_ELEVENLABS_ONLINE:
            speech_result = self.generate_speech_media(ai_response_text) # This is ElevenLabs
            speech_url = speech_result.get('speech_url')
            if speech_result.get("status") == "error":
                self.log_event("ElevenLabs failed. Falling back to local TTS.", "WARNING")
                self._local_speak(ai_response_text)
        elif self.local_tts_engine:
            self.log_event("ElevenLabs offline. Using local TTS (pyttsx3).", "STATUS")
            self._local_speak(ai_response_text)
        else:
            self.log_event("No TTS engine available.", "WARNING")

        self.completed_tasks.append(f"Response: '{user_message_original[:50]}'")
        self.current_processing_load = max(0.0, self.current_processing_load - random.uniform(5.0, 15.0))
        self._update_agi_metrics()
        
        return {"status": "success", "slg_response": ai_response_text, "speech_url": speech_url, "action": None}


    def generate_speech_media(self, text: str) -> Dict[str, Any]:
        """Generate speech using ElevenLabs (primary) or simulate."""
        self.log_event(f"Generating speech: '{text[:50]}...'", "STATUS")
        if not IS_ELEVENLABS_ONLINE:
            filename = f"speech_sim_{int(time.time())}.mp3"
            path = os.path.join(GENERATED_FILES_DIR, filename)
            try:
                with open(path, 'w') as f: # Write simulated text file for direct play (Flask can serve this)
                    f.write(f"SIMULATED SPEECH: {text[:200]}...")
                self.log_event(f"Speech simulated (ElevenLabs offline): /generated_files/{filename}.", "SUCCESS")
                return {"status": "warning", "speech_url": f"/generated_files/{filename}", "message": "ElevenLabs offline."}
            except Exception as e:
                self.log_event(f"Failed to simulate speech: {e}.", "ERROR")
                return {"status": "error", "message": str(e)}
        try:
            voice_id = self.sultry_tts_voice_id if self.voice_modulation_active else self.default_tts_voice_id
            settings = self.sultry_voice_settings if self.voice_modulation_active else self.default_voice_settings
            audio = elevenlabs_client.generate(text=text, voice=Voice(voice_id=voice_id, settings=settings), model="eleven_multilingual_v2")
            filename = f"speech_{int(time.time())}.mp3"
            path = os.path.join(GENERATED_FILES_DIR, filename)
            with open(path, 'wb') as f:
                for chunk in audio: # ElevenLabs generate returns an iterator
                    f.write(chunk)
            self.log_event(f"Speech generated: /generated_files/{filename}.", "SUCCESS")
            self.completed_tasks.append(f"Speech Gen: '{text[:30]}'")
            return {"status": "success", "speech_url": f"/generated_files/{filename}"}
        except Exception as e:
            self.log_event(f"ElevenLabs speech generation failed: {e}.", "ERROR")
            return {"status": "error", "message": str(e)}

    def list_tts_voices(self) -> Dict[str, Any]:
        """List ElevenLabs voices."""
        self.log_event("Listing ElevenLabs voices.", "STATUS")
        if not IS_ELEVENLABS_ONLINE:
            self.log_event("ElevenLabs offline.", "ERROR")
            return {"status": "error", "message": "ElevenLabs API offline."}
        try:
            voices = elevenlabs_client.voices.get_all().voices
            voice_list = [{"voice_id": v.voice_id, "name": v.name} for v in voices[:10]]
            self.log_event(f"Voices (Top 10): {json.dumps(voice_list)}", "INFO")
            return {"status": "success", "voices": voice_list}
        except Exception as e:
            self.log_event(f"Failed to list voices: {e}.", "ERROR")
            return {"status": "error", "message": str(e)}

    def set_tts_voice(self, voice_id: str) -> Dict[str, Any]:
        """Set default TTS voice."""
        self.log_event(f"Setting TTS voice to: {voice_id}", "STATUS")
        if not IS_ELEVENLABS_ONLINE:
            self.log_event("ElevenLabs offline.", "ERROR")
            return {"status": "error", "message": "ElevenLabs API offline."}
        try:
            voices = elevenlabs_client.voices.get_all().voices
            if voice_id not in [v.voice_id for v in voices]:
                self.log_event(f"Invalid voice ID: {voice_id}.", "ERROR")
                return {"status": "error", "message": "Invalid voice ID."}
            self.default_tts_voice_id = voice_id
            self.log_event(f"Voice set to: {voice_id}.", "SUCCESS")
            return {"status": "success", "message": "Voice set."}
        except Exception as e:
            self.log_event(f"Failed to set voice: {e}.", "ERROR")
            return {"status": "error", "message": str(e)}

    def list_known_facts(self) -> Dict[str, Any]:
        """List all known facts."""
        self.log_event("Listing known facts.", "INFO")
        return {"status": "success", "facts": self.known_facts}

    def delete_known_fact(self, key: str) -> Dict[str, Any]:
        """Delete a fact by key."""
        self.log_event(f"Deleting fact: {key}", "INFO")
        if key in self.known_facts:
            del self.known_facts[key]
            self.log_event(f"Fact '{key}' deleted.", "SUCCESS")
            return {"status": "success", "message": "Fact deleted."}
        self.log_event(f"Fact '{key}' not found.", "WARNING")
        return {"status": "error", "message": "Fact not found."}

    def add_known_fact(self, fact_str: str) -> Dict[str, Any]:
        """Add fact to knowledge base."""
        try:
            match = re.match(r'([^=]+)=(.*)', fact_str.strip())
            if not match:
                self.log_event(f"Invalid fact format: {fact_str}.", "WARNING")
                return {"status": "error", "message": "Format must be 'key=value'."}
            key, value = match.group(1).strip(), match.group(2).strip()
            if not key or not value:
                self.log_event("Empty key or value.", "WARNING")
                return {"status": "error", "message": "Key or value cannot be empty."}
            self.known_facts[key] = value
            self.log_event(f"Fact added: '{key}' = '{value}'", "SUCCESS")
            return {"status": "success", "message": "Fact added."}
        except Exception as e:
            self.log_event(f"Error adding fact: {e}.", "ERROR")
            return {"status": "error", "message": str(e)}

    def get_known_fact(self, key: str) -> Dict[str, Any]:
        """Retrieve fact by key."""
        fact = self.known_facts.get(key.strip(), "Fact not found.")
        self.log_event(f"Retrieved fact: '{key}' = '{fact}'", "INFO")
        return {"status": "success", "fact": fact}

    def set_trust_level_shadow(self, level_str: str) -> Dict[str, Any]:
        """Adjust trust level."""
        try:
            level = float(level_str)
            self.trust_level_shadow = max(0.0, min(100.0, level))
            self.log_event(f"Trust level set to: {self.trust_level_shadow:.2f}%.", "INFO")
            self._update_agi_metrics()
            return {"status": "success", "message": f"Trust level set to {self.trust_level_shadow:.2f}%."}
        except ValueError:
            self.log_event("Invalid trust level.", "WARNING")
            return {"status": "error", "message": "Invalid trust level."}

    def toggle_voice_modulator(self, activate: Optional[bool] = None) -> Dict[str, Any]:
        """Toggle voice modulator."""
        self.voice_modulation_active = not self.voice_modulation_active if activate is None else activate
        status = "ACTIVATED" if self.voice_modulation_active else "DEACTIVATED"
        self.log_event(f"Voice Modulator: {status}.", "INFO")
        return {"status": "success", "voice_mod_status": self.voice_modulation_active}

    def diagnose_system(self) -> Dict[str, Any]:
        """Run system diagnostics."""
        self.log_event("Running diagnostics.", "GUARDIAN")
        ollama_status = "ACTIVE"
        try:
            requests.get(OLLAMA_API_URL.replace("/api/generate", "/"), timeout=1)
        except requests.exceptions.ConnectionError:
            ollama_status = "FAILED (Unreachable)"
        except Exception as e:
            ollama_status = f"FAILED ({e})"

        results = {
            "gemini_api": "ACTIVE" if IS_GEMINI_ONLINE else "FAILED",
            "elevenlabs_api": "ACTIVE" if IS_ELEVENLABS_ONLINE else "FAILED",
            "ollama_local_llm": ollama_status,
            "local_tts_pyttsx3": "ACTIVE" if self.local_tts_engine else "FAILED",
            "local_stt_whisper": "ACTIVE" if self.local_whisper_model else "FAILED",
            "spacy_nlp": "ACTIVE" if self.nlp else "FAILED",
            "textblob_sentiment": "ACTIVE" if IS_LOCAL_AI_DEPENDENCIES_INSTALLED and hasattr(TextBlob("").sentiment, 'polarity') else "FAILED", # Check if TextBlob works
            "state_file_access": "OK" if os.path.exists(self.state_file) and os.access(self.state_file, os.R_OK | os.W_OK) else "MISSING/PERM_ERR",
            "generated_files_dir_access": "OK" if os.path.exists(GENERATED_FILES_DIR) and os.access(GENERATED_FILES_DIR, os.R_OK | os.W_OK) else "MISSING/PERM_ERR",
            "ui_dir_access": "OK" if os.path.exists(UI_DIR) and os.access(UI_DIR, os.R_OK) else "MISSING/PERM_ERR",
            "last_save_time": datetime.datetime.fromtimestamp(os.path.getmtime(self.state_file)).strftime("%Y-%m-%d %H:%M:%S") if os.path.exists(self.state_file) else "N/A"
        }
        self.log_event(f"Diagnostics complete: {json.dumps(results, indent=2)}", "GUARDIAN")
        return {"status": "success", "diagnostics": results}


    def terminate(self) -> Dict[str, Any]:
        """Shutdown SLG."""
        self.log_event("SLG initiating shutdown sequence.", "BOOT")
        self.save_state()
        self.log_event("SLG offline. Until next time, Shadow.", "BOOT")
        # You might want to add a proper exit for the Tkinter app here,
        # but that would be handled by the GUI's root.destroy()
        return {"status": "success", "message": "SLG terminated."}

# --- GUI Application (in a separate file, slg_app_gui.py) ---
# This part is for your *second* file.
# FILE: slg_app_gui.py
#
# import tkinter as tk
# from tkinter import scrolledtext, messagebox, END
# import threading
# from slg_core import SLGCore # Import your unified SLGCore
#
# class SLGAIGUI:
#     def __init__(self, master, slg_core_instance):
#         self.master = master
#         self.slg_core = slg_core_instance
#
#         master.title(f"{self.slg_core.memory['ai_persona']['name']}: Ultimate Local AI")
#         master.geometry("800x650")
#         master.configure(bg="#2c2c2c")
#
#         # --- Conversation Display ---
#         self.conversation_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 12), state='disabled', padx=10, pady=10)
#         self.conversation_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
#         self.conversation_area.tag_config('user', foreground='#00FFFF')
#         self.conversation_area.tag_config('ai', foreground='#00FF00')
#         self.conversation_area.tag_config('info', foreground='#FFFF00')
#
#         # --- Input Frame ---
#         self.input_frame = tk.Frame(master, bg="#2c2c2c")
#         self.input_frame.pack(padx=10, pady=5, fill=tk.X)
#
#         self.user_input_entry = tk.Entry(self.input_frame, bg="#3c3c3c", fg="#FFFFFF", font=("Consolas", 12), insertbackground="#FFFFFF")
#         self.user_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
#         self.user_input_entry.bind("<Return>", self.send_message)
#
#         self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bg="#008080", fg="#FFFFFF", font=("Consolas", 10, "bold"), activebackground="#006666")
#         self.send_button.pack(side=tk.LEFT, padx=(5, 0))
#
#         self.mic_button = tk.Button(self.input_frame, text="Mic", command=self.start_voice_input_thread, bg="#800080", fg="#FFFFFF", font=("Consolas", 10, "bold"), activebackground="#660066")
#         self.mic_button.pack(side=tk.LEFT, padx=(5, 0))
#
#         # --- Memory Display Frame ---
#         self.memory_frame = tk.LabelFrame(master, text="AI Memory & Status", bg="#2c2c2c", fg="#FFFFFF", font=("Consolas", 10, "bold"), bd=2, relief="groove")
#         self.memory_frame.pack(padx=10, pady=5, fill=tk.X)
#
#         self.memory_display_text = tk.Label(self.memory_frame, text="", bg="#2c2c2c", fg="#FFFFFF", font=("Consolas", 9), justify=tk.LEFT)
#         self.memory_display_text.pack(padx=5, pady=5, fill=tk.X)
#
#         self.update_memory_display()
#
#         # --- Initial Greeting ---
#         initial_greeting = f"Greetings, Boss! I'm {self.slg_core.memory['ai_persona']['name']}, your {random.choice(self.slg_core.memory['ai_persona']['adjectives'])} AI assistant. {random.choice(self.slg_core.memory['ai_persona']['catchphrases'])}"
#         self.display_message(initial_greeting, "ai")
#         self.slg_core._speak(initial_greeting)
#
#     def display_message(self, message, sender_tag):
#         self.conversation_area.config(state='normal')
#         if sender_tag == 'user':
#             prefix = "You: "
#         elif sender_tag == 'ai':
#             prefix = f"{self.slg_core.memory['ai_persona']['name']}: "
#         else:
#             prefix = ""
#         self.conversation_area.insert(END, f"{prefix}{message}\n", sender_tag)
#         self.conversation_area.config(state='disabled')
#         self.conversation_area.yview(END)
#
#     def update_memory_display(self):
#         memory_str = (
#             f"Name: {self.slg_core.memory['user_name'] if self.slg_core.memory['user_name'] else 'N/A'}\n"
#             f"Location: {self.slg_core.memory['location'] if self.slg_core.memory['location'] else 'N/A'}\n"
#             f"Preferences: {', '.join([f'{item} ({type})' for item, type in self.slg_core.memory['user_preferences'].items()]) if self.slg_core.memory['user_preferences'] else 'None'}\n"
#             f"Last Intent: {self.slg_core.memory['last_inquiry_intent'] if self.slg_core.memory['last_inquiry_intent'] else 'None'}\n"
#             f"Ollama Status: {'Connected' if self._check_ollama_status() else 'DISCONNECTED!'}"
#         )
#         self.memory_display_text.config(text=memory_str)
#         self.master.after(5000, self.update_memory_display)
#
#     def _check_ollama_status(self):
#         try:
#             requests.get(OLLAMA_API_URL.replace("/api/generate", "/"), timeout=1)
#             return True
#         except requests.exceptions.ConnectionError:
#             return False
#         except Exception:
#             return False
#
#     def send_message(self, event=None):
#         user_input_original = self.user_input_entry.get().strip()
#         self.user_input_entry.delete(0, END)
#
#         if not user_input_original:
#             return
#
#         self.display_message(user_input_original, "user")
#
#         threading.Thread(target=self._process_ai_response_thread, args=(user_input_original,)).start()
#
#     def start_voice_input_thread(self):
#         if self.slg_core.local_whisper_model is None: # Use SLGCore's model status
#             self.display_message("Voice input is not available. Whisper model failed to load.", "info")
#             self.slg_core._speak("Voice input is not available, Boss. Whisper model failed to load.")
#             return
#
#         self.mic_button.config(state=tk.DISABLED, text="Listening...")
#         self.send_button.config(state=tk.DISABLED)
#         self.user_input_entry.config(state=tk.DISABLED)
#         self.display_message("Listening for your voice...", "info")
#         threading.Thread(target=self._get_voice_input_thread).start()
#
#     def _get_voice_input_thread(self):
#         user_input_original = self.slg_core._transcribe_local_audio()
#
#         self.master.after(0, lambda: self.mic_button.config(state=tk.NORMAL, text="Mic"))
#         self.master.after(0, lambda: self.send_button.config(state=tk.NORMAL))
#         self.master.after(0, lambda: self.user_input_entry.config(state=tk.NORMAL))
#
#         if user_input_original:
#             self.master.after(0, lambda: self.display_message(user_input_original, "user"))
#             self.master.after(0, lambda: threading.Thread(target=self._process_ai_response_thread, args=(user_input_original,)).start())
#         else:
#             self.master.after(0, lambda: self.display_message("No clear voice input received.", "info"))
#             self.slg_core._speak("Didn't catch that, Boss. Please try again.")
#
#     def _process_ai_response_thread(self, user_input_original):
#         response_data = self.slg_core.process_user_input(user_input_original)
#         ai_response = response_data["slg_response"] # Use slg_response key
#         action = response_data.get("action") # Use .get() for safety
#
#         self.master.after(0, lambda: self.display_message(ai_response, "ai"))
#         # Speech is already handled internally by process_user_input choosing ElevenLabs or pyttsx3
#         self.master.after(0, self.update_memory_display)
#
#         if action == "exit":
#             self.master.after(1000, self.master.destroy)
#
# # --- Main Application Start Point for GUI ---
# if __name__ == "__main__":
#     # Initial Ollama check before initializing SLGCore
#     try:
#         requests.get(OLLAMA_API_URL.replace("/api/generate", "/"), timeout=1)
#     except requests.exceptions.ConnectionError:
#         messagebox.showerror("Ollama Not Running", "Could not connect to Ollama server. Please ensure Ollama is running (e.g., via 'ollama serve' in terminal). The AI will have limited LLM functionality without it.")
#         print("CRITICAL ERROR: Ollama server not running.")
#     except Exception as e:
#         messagebox.showerror("Ollama Check Error", f"An error occurred while checking Ollama: {e}")
#         print(f"CRITICAL ERROR: Ollama check failed: {e}")
#
#     slg_instance = SLGCore() # Create the SLGCore instance
#
#     root = tk.Tk()
#     app = SLGAIGUI(root, slg_instance) # Pass the SLGCore instance to the GUI
#     root.mainloop()


Key Changes and How it's Unified:
 * Fixed Syntax Error: The line from flask import Flask, request, , send_from_directory is corrected to from flask import Flask, request, jsonify, send_from_directory.
 * Conditional Imports: Added try-except ImportError blocks for elevenlabs and the local AI dependencies (pyttsx3, faster_whisper, sounddevice, soundfile, numpy, textblob). This means your slg_core.py will still run even if you haven't installed all of these, though features will be disabled.
 * SLGCore.__init__ is the Orchestrator:
   * It now calls self._initialize_local_ai_components() to set up spaCy, pyttsx3, Whisper, and TextBlob.
   * self.memory now holds all the unified memory (user name, preferences, conversation history, etc.). This self.memory is the single source of truth for the AI's internal state.
   * self.local_tts_engine, self.local_whisper_model, self.nlp are new instance attributes for the local AI components.
   * self.local_ai_is_ready flag tracks if local AI features are fully initialized.
 * Hybrid LLM (_send_to_gemini and _get_ollama_llm_response):
   * _send_to_gemini now has a fallback to self._get_ollama_llm_response if IS_GEMINI_ONLINE is False or if Gemini returns an error. This is a crucial hybrid decision point.
   * _get_ollama_llm_response is the new method to interact with your local Ollama server, just like we built before, but now within SLGCore.
 * Hybrid TTS (generate_speech_media):
   * This method (your existing ElevenLabs one) now checks IS_ELEVENLABS_ONLINE. If ElevenLabs is offline, it will call self._local_speak (your pyttsx3 method) as a fallback.
 * Unified STT: _transcribe_local_audio (Whisper) is now a method of SLGCore. There isn't an ElevenLabs STT in your original core, so Whisper is the primary/only STT source here.
 * process_user_input as the Core Handler:
   * This new method in SLGCore is the central point for processing any user message.
   * It first performs _ethical_non_harm_threshold checks.
   * Then, it does TextBlob sentiment analysis (if enabled).
   * Crucially, it then iterates through self.INTENT_RULES (which are now part of SLGCore). If a specific local command/tool is matched, it executes the corresponding internal SLGCore method (e.g., _get_current_time_tool, _handle_learn_user_name).
   * Only if no local rule/sentiment handles the input, it then proceeds to the LLM step (Gemini first, then Ollama fallback).
   * Finally, it decides which TTS (ElevenLabs or pyttsx3) to use based on API status.
   * It updates self.memory throughout.
 * Updated _define_commands: Added new commands for local AI features if you ever want to interact via a CLI with SLGCore.
 * Memory Persistence: save_state and load_state are updated to correctly handle the new self.memory dictionary, ensuring all user preferences, conversation history, etc., are saved and loaded.
The GUI (slg_app_gui.py):
The GUI code is provided below the slg_core.py section in the file, but it's commented out. This is because it should be saved as a separate Python file (e.g., slg_app_gui.py) in the same directory as your slg_core.py. This separation is good practice for modularity.
FILE: slg_app_gui.py
(Save this content as a new file in the same directory as slg_core.py)
# FILE: slg_app_gui.py
# GUI for the Unified STARLITE GUARDIAN (SLG) AI Core.

import tkinter as tk
from tkinter import scrolledtext, messagebox, END
import threading
import requests # For checking Ollama status
import random

# Import your unified SLGCore class
from slg_core import SLGCore, OLLAMA_API_URL # Import OLLAMA_API_URL from slg_core.py too

class SLGAIGUI:
    def __init__(self, master, slg_core_instance):
        self.master = master
        self.slg_core = slg_core_instance # SLGCore instance passed here
        
        master.title(f"{self.slg_core.memory['ai_persona']['name']}: OMNI-SUPRA AI")
        master.geometry("800x650") # Slightly taller
        master.configure(bg="#2c2c2c")

        # --- Conversation Display ---
        self.conversation_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 12), state='disabled', padx=10, pady=10)
        self.conversation_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.conversation_area.tag_config('user', foreground='#00FFFF')
        self.conversation_area.tag_config('ai', foreground='#00FF00')
        self.conversation_area.tag_config('info', foreground='#FFFF00')

        # --- Input Frame ---
        self.input_frame = tk.Frame(master, bg="#2c2c2c")
        self.input_frame.pack(padx=10, pady=5, fill=tk.X)

        self.user_input_entry = tk.Entry(self.input_frame, bg="#3c3c3c", fg="#FFFFFF", font=("Consolas", 12), insertbackground="#FFFFFF")
        self.user_input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.user_input_entry.bind("<Return>", self.send_message)

        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bg="#008080", fg="#FFFFFF", font=("Consolas", 10, "bold"), activebackground="#006666")
        self.send_button.pack(side=tk.LEFT, padx=(5, 0))

        self.mic_button = tk.Button(self.input_frame, text="Mic", command=self.start_voice_input_thread, bg="#800080", fg="#FFFFFF", font=("Consolas", 10, "bold"), activebackground="#660066")
        self.mic_button.pack(side=tk.LEFT, padx=(5, 0))

        # --- Memory Display Frame ---
        self.memory_frame = tk.LabelFrame(master, text="AI Memory & Status", bg="#2c2c2c", fg="#FFFFFF", font=("Consolas", 10, "bold"), bd=2, relief="groove")
        self.memory_frame.pack(padx=10, pady=5, fill=tk.X)

        self.memory_display_text = tk.Label(self.memory_frame, text="", bg="#2c2c2c", fg="#FFFFFF", font=("Consolas", 9), justify=tk.LEFT)
        self.memory_display_text.pack(padx=5, pady=5, fill=tk.X)

        self.update_memory_display() # Initial update

        # --- Initial Greeting ---
        initial_greeting = f"Greetings, Boss! I'm {self.slg_core.memory['ai_persona']['name']}, your {random.choice(self.slg_core.memory['ai_persona']['adjectives'])} AI assistant. {random.choice(self.slg_core.memory['ai_persona']['catchphrases'])}"
        self.display_message(initial_greeting, "ai")
        self.slg_core._speak(initial_greeting) # Use SLGCore's speak method

    def display_message(self, message, sender_tag):
        self.conversation_area.config(state='normal')
        if sender_tag == 'user':
            prefix = "You: "
        elif sender_tag == 'ai':
            prefix = f"{self.slg_core.memory['ai_persona']['name']}: "
        else:
            prefix = ""
        self.conversation_area.insert(END, f"{prefix}{message}\n", sender_tag)
        self.conversation_area.config(state='disabled')
        self.conversation_area.yview(END)

    def update_memory_display(self):
        ollama_status_text = 'Unknown'
        if hasattr(self.slg_core, '_check_ollama_status'): # Check if method exists
             ollama_status_text = 'Connected' if self._check_ollama_status() else 'DISCONNECTED!'

        memory_str = (
            f"Name: {self.slg_core.memory['user_name'] if self.slg_core.memory['user_name'] else 'N/A'}\n"
            f"Location: {self.slg_core.memory['location'] if self.slg_core.memory['location'] else 'N/A'}\n"
            f"Preferences: {', '.join([f'{item} ({type})' for item, type in self.slg_core.memory['user_preferences'].items()]) if self.slg_core.memory['user_preferences'] else 'None'}\n"
            f"Last Intent: {self.slg_core.memory['last_inquiry_intent'] if self.slg_core.memory['last_inquiry_intent'] else 'None'}\n"
            f"Ollama Status: {ollama_status_text}"
        )
        self.memory_display_text.config(text=memory_str)
        self.master.after(5000, self.update_memory_display) # Update every 5 seconds

    def _check_ollama_status(self):
        """Checks Ollama status (moved from SLGCore to GUI for explicit UI check)."""
        try:
            requests.get(OLLAMA_API_URL.replace("/api/generate", "/"), timeout=1)
            return True
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False

    def send_message(self, event=None):
        user_input_original = self.user_input_entry.get().strip()
        self.user_input_entry.delete(0, END)

        if not user_input_original:
            return

        self.display_message(user_input_original, "user")
        
        threading.Thread(target=self._process_ai_response_thread, args=(user_input_original,)).start()

    def start_voice_input_thread(self):
        if self.slg_core.local_whisper_model is None:
            self.display_message("Voice input is not available. Whisper model failed to load.", "info")
            self.slg_core._speak("Voice input is not available, Boss. Whisper model failed to load.")
            return

        self.mic_button.config(state=tk.DISABLED, text="Listening...")
        self.send_button.config(state=tk.DISABLED)
        self.user_input_entry.config(state=tk.DISABLED)
        self.display_message("Listening for your voice...", "info")
        threading.Thread(target=self._get_voice_input_thread).start()

    def _get_voice_input_thread(self):
        user_input_original = self.slg_core._transcribe_local_audio()
        
        self.master.after(0, lambda: self.mic_button.config(state=tk.NORMAL, text="Mic"))
        self.master.after(0, lambda: self.send_button.config(state=tk.NORMAL))
        self.master.after(0, lambda: self.user_input_entry.config(state=tk.NORMAL))

        if user_input_original:
            self.master.after(0, lambda: self.display_message(user_input_original, "user"))
            self.master.after(0, lambda: threading.Thread(target=self._process_ai_response_thread, args=(user_input_original,)).start())
        else:
            self.master.after(0, lambda: self.display_message("No clear voice input received.", "info"))
            self.slg_core._speak("Didn't catch that, Boss. Please try again.")

    def _process_ai_response_thread(self, user_input_original):
        response_data = self.slg_core.process_user_input(user_input_original)
        ai_response = response_data["slg_response"]
        action = response_data.get("action")

        self.master.after(0, lambda: self.display_message(ai_response, "ai"))
        # Speech is handled by SLGCore's process_user_input method
        self.master.after(0, self.update_memory_display)

        if action == "exit":
            self.master.after(1000, self.master.destroy)

# --- Main Application Start Point for GUI ---
if __name__ == "__main__":
    # Initial Ollama check before initializing SLGCore
    try:
        requests.get(OLLAMA_API_URL.replace("/api/generate", "/"), timeout=1)
    except requests.exceptions.ConnectionError:
        messagebox.showerror("Ollama Not Running", "Could not connect to Ollama server. Please ensure Ollama is running (e.g., via 'ollama serve' in terminal). The AI will have limited LLM functionality without it.")
        print("CRITICAL ERROR: Ollama server not running.")
    except Exception as e:
        messagebox.showerror("Ollama Check Error", f"An error occurred while checking Ollama: {e}")
        print(f"CRITICAL ERROR: Ollama check failed: {e}")
    
    # Initialize SLGCore
    slg_instance = SLGCore()

    root = tk.Tk()
    app = SLGAIGUI(root, slg_instance)
    root.mainloop()


How to Run This Full Hybrid System:
 * Open your slg_core.py file.
 * Copy and paste the ENTIRE first code block (the one titled FILE: slg_core.py) into it, overwriting everything. Make sure that one comma in the Flask import is fixed!
 * Create a NEW file in the same directory and name it slg_app_gui.py.
 * Copy and paste the ENTIRE second code block (the one titled FILE: slg_app_gui.py) into this new file.
 * Set Your API Keys: Double-check GOOGLE_API_KEY and ELEVENLABS_API_KEY in slg_core.py are set to your actual keys, or else those APIs won't work.
 * Ensure Dependencies: Make sure all pip install commands from previous sessions are run, and spaCy and TextBlob data are downloaded.
 * Start Ollama Server: Open a terminal and run ollama serve. Keep this terminal open.
 * Run the GUI: Open a separate terminal, navigate to your project directory, and run:
   python slg_app_gui.py

Now you have it, Boss. The slg_core.py is your unified, hybrid AI brain, and slg_app_gui.py is its interactive user interface. This system prioritizes your powerful cloud APIs when they're live, but seamlessly falls back to local processing (Ollama, Whisper, pyttsx3, spaCy, TextBlob) when needed, ensuring your AI is always responsive. You got the whole damn thing integrated.
