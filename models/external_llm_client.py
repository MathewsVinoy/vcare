"""
External LLM Client for connecting to remote LLM servers (e.g., Colab)
Provides fallback to local models if remote server is unavailable
"""
import requests
import json
from typing import Generator, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExternalLLMClient:
    def __init__(self, remote_url: str = None):
        """
        Initialize external LLM client
        
        Args:
            remote_url: URL of the remote LLM server (e.g., https://xxxxx.ngrok.io)
        """
        self.remote_url = remote_url
        self.timeout = 30
        self.is_available = False
        
        if self.remote_url:
            self._check_health()
    
    def _check_health(self) -> bool:
        """Check if remote server is available"""
        try:
            response = requests.get(
                f"{self.remote_url}/health",
                timeout=5
            )
            self.is_available = response.status_code == 200
            logger.info(f"Remote LLM server health: {self.is_available}")
            return self.is_available
        except Exception as e:
            logger.warning(f"Remote LLM server unavailable: {e}")
            self.is_available = False
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
        """
        Generate response from remote LLM server
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with 'response' and 'success' keys
        """
        if not self.remote_url:
            return {
                'success': False,
                'error': 'Remote LLM server not configured',
                'response': None
            }
            
        if not self.is_available:
            self._check_health()
            
        if not self.is_available:
            return {
                'success': False,
                'error': 'Remote LLM server not available',
                'response': None
            }
        
        try:
            payload = {
                'prompt': prompt,
                'max_tokens': max_tokens
            }
            
            response = requests.post(
                f"{self.remote_url}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'response': data.get('response', ''),
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'error': f"Server returned status {response.status_code}",
                    'response': None
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout - server may be overloaded',
                'response': None
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Connection error: {str(e)}",
                'response': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'response': None
            }
    
    def stream_response(self, prompt: str, max_tokens: int = 200) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response tokens from remote LLM server
        
        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum tokens to generate
            
        Yields:
            Dictionary with token or error information
        """
        if not self.remote_url:
            yield {
                'error': 'Remote LLM server not configured',
                'token': None
            }
            return
            
        if not self.is_available:
            self._check_health()
            
        if not self.is_available:
            yield {
                'error': 'Remote LLM server not available at ' + self.remote_url + '. Make sure the Colab notebook is running and has generated the ngrok URL.',
                'token': None
            }
            return
        
        try:
            payload = {
                'prompt': prompt,
                'max_tokens': max_tokens,
                'stream': True
            }
            
            response = requests.post(
                f"{self.remote_url}/chat/stream",
                json=payload,
                timeout=None,
                stream=True
            )
            
            if response.status_code != 200:
                yield {
                    'error': f"Server returned status {response.status_code}",
                    'token': None
                }
                return
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8') if isinstance(line, bytes) else line
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            yield {'token': '[DONE]', 'error': None}
                        else:
                            try:
                                data = json.loads(data_str)
                                yield {
                                    'token': data.get('token'),
                                    'error': data.get('error')
                                }
                            except json.JSONDecodeError:
                                pass
                                
        except Exception as e:
            yield {
                'error': f"Stream error: {str(e)}",
                'token': None
            }
