# ai-photo-processor/utils/gemini_handler.py

import time
from google import genai
from google.genai.errors import APIError
from PIL import Image
from typing import List, Tuple, Optional

from .logger import SimpleLogger


class GeminiHandler:
    """Manages API requests to Google Gemini, including key rotation and retries."""

    def __init__(self, api_keys: List[str], model_name: str, logger: SimpleLogger):
        if not api_keys:
            raise ValueError("API keys list cannot be empty for GeminiHandler.")
        self.api_keys = api_keys
        self.model_name = model_name
        self.logger = logger
        self.current_key_index = 0
        self.max_retries = len(api_keys) * 2
        self.client: Optional[genai.Client] = None
        self._configure_client()

    def _configure_client(self):
        """Creates a new Gemini client with the current API key."""
        api_key = self.api_keys[self.current_key_index]
        self.client = genai.Client(api_key=api_key)
        self.logger.info(f"GeminiHandler configured with model '{self.model_name}' and key #{self.current_key_index + 1}")

    def _switch_api_key(self):
        """Rotates to the next API key and reconfigures the client."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.logger.info(f"Switching to API key #{self.current_key_index + 1}")
        self._configure_client()

    def send_request(self, prompt_text: str, images: List[Image.Image]) -> Tuple[str, bool]:
        """
        Sends a request to the Gemini API with the given prompt and images.

        Args:
            prompt_text: The text prompt to send to the model.
            images: List of PIL Image objects to include in the request.

        Returns:
            A tuple of (response_text, success_bool).
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Sending API request with {len(images)} images (Attempt {attempt + 1}/{self.max_retries}, Key #{self.current_key_index + 1})...")

                # Build contents list: prompt first, then all images
                contents = [prompt_text] + images

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents
                )

                return response.text, True

            except APIError as e:
                last_error = e
                error_code = getattr(e, 'code', None)

                # Check for retryable errors (rate limit, server errors)
                if error_code in [429, 500, 503]:
                    self.logger.warn(f"API Error (Code: {error_code}): {e.message}. Retrying with next key...")
                    self._switch_api_key()
                    time.sleep(2)
                    continue
                else:
                    self.logger.error(f"Non-retryable API Error (Code: {error_code}): {e.message}")
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
