"""
Task executor with multiple circuit breakers and cost controls.
Pre-flight token estimation, timeout handling, and cost tracking.
"""
import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json

import requests
from requests.exceptions import Timeout, RequestException
import tiktoken
from pydantic import BaseModel, Field, validator

from .crypto_utils import AgentKeyManager

logger = logging.getLogger(__name__)

@dataclass
class ExecutionResult:
    """Structured result of task execution."""
    success: bool
    output: Dict[str, Any]
    cost_usd: float
    token_count: int
    execution_time_ms: int
    error_message: Optional[str] = None
    receipt_hash: Optional[str] = None
    
    def to_firestore_dict(self) -> Dict[str, Any]:
        """Convert to Firestore-friendly dictionary."""
        return {
            'success': self.success,
            'output': self.output,
            'cost_usd': self.cost_usd,
            'token_count': self.token_count,
            'execution_time_ms': self.execution_time_ms,
            'error_message': self.error_message,
            'receipt_hash': self.receipt_hash,
            'completed_at': datetime.now(timezone.utc)
        }

class CircuitBreakerError(Exception):
    """Raised when circuit breaker trips."""
    pass

class TaskCostEstimator:
    """Estimates token counts and costs before execution."""
    
    # Token prices in USD (approximate as of 2024)
    PRICES = {
        'claude-3-haiku': {'input': 0.25 / 1_000_000, 'output': 1.25 / 1_000_000},
        'claude-3-sonnet': {'input': 3.0 / 1_000_000, 'output': 15.0 / 1_000_000},
        'gpt-4-turbo': {'input': 10.0 / 1_000_000, 'output': 30.0 / 1_000_000}
    }
    
    @staticmethod
    def estimate_tokens(text: str, model: str = 'claude-3-haiku') -> int:
        """
        Estimate token count for text.
        Uses tiktoken for OpenAI models, approximation for Claude.
        """
        try:
            if 'gpt' in model:
                encoding = tiktoken.encoding_for_model(model)
                return len(encoding.encode(text))
            else:
                # Approximation: 1 token ≈ 4 characters for English
                return len(text) // 4
        except Exception as e:
            logger.warning(f"Token estimation failed: {e}, using fallback")
            return len(text) // 4  # Conservative fallback
    
    @classmethod
    def estimate_cost(cls, prompt: str, completion_estimate: int