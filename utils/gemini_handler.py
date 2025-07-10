# ai-photo-processor/utils/gemini_handler.py

import time
import google.generativeai as genai
from PIL import Image
from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted
from typing import List, Tuple

from .logger import SimpleLogger

class GeminiHandler:
    """Manages API requests to Google Gemini, including key rotation and retries."""

    def __init__(self, api_keys: List[str], model_name: str, logger: SimpleLogger):
        if not api_keys: raise ValueError("API keys list cannot be empty for GeminiHandler.")
        self.api_keys = api_keys
        self.model_name = model_name
        self.logger = logger
        self.current_key_index = 0
        self.max_retries = len(api_keys) * 2
        self.chat = None
        self._configure_client()

    def _configure_client(self):
        api_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        self.chat = model.start_chat(history=[])
        self.logger.info(f"GeminiHandler configured with model '{self.model_name}' and key #{self.current_key_index + 1}")

    def _switch_api_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.logger.info(f"Switching to API key #{self.current_key_index + 1}")
        self._configure_client()

    def send_request(self, prompt_text: str, image_path: str) -> Tuple[str, bool]:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Sending API request (Attempt {attempt + 1}/{self.max_retries}, Key #{self.current_key_index + 1})...")
                img = Image.open(image_path)
                response = self.chat.send_message([prompt_text, img], stream=False)
                return response.text, True
            except (ResourceExhausted, GoogleAPICallError) as e:
                last_error = e
                if hasattr(e, 'grpc_status_code') and e.grpc_status_code in [429, 500, 503]:
                    self.logger.warn(f"API Error (Code: {e.grpc_status_code}): {e}. Retrying with next key...")
                    self._switch_api_key()
                    time.sleep(2)
                    continue
                else:
                    self.logger.error("Non-retryable API Error encountered.", exception=e)
                    break
            except Exception as e:
                last_error = e
                self.logger.error("An unexpected error occurred during API call. Retrying with next key...", exception=e)
                self._switch_api_key()
                time.sleep(2)
                continue
        error_message = f"Failed after {self.max_retries} attempts. Last error: {last_error}"
        self.logger.error(error_message)
        return error_message, False