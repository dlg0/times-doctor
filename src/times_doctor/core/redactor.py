"""Redact sensitive information from logs and output."""

import re
import os
from typing import Any, Dict


def redact_api_key(text: str) -> str:
    """Redact API keys from text.
    
    Args:
        text: Text that may contain API keys
        
    Returns:
        Text with API keys redacted
    """
    if not text:
        return text
    
    patterns = [
        (r'sk-proj-[A-Za-z0-9_-]{20,}', 'sk-proj-***REDACTED***'),
        (r'sk-[A-Za-z0-9]{20,}', 'sk-***REDACTED***'),
        (r'sk-ant-[A-Za-z0-9_-]{20,}', 'sk-ant-***REDACTED***'),
    ]
    
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    
    return result


def redact_env_keys() -> Dict[str, str]:
    """Get redacted versions of sensitive environment variables.
    
    Returns:
        Dict mapping env var names to redacted values
    """
    sensitive_keys = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'AMP_API_KEY',
    ]
    
    redacted = {}
    for key in sensitive_keys:
        value = os.environ.get(key)
        if value:
            redacted[key] = f"{value[:8]}***REDACTED***"
        else:
            redacted[key] = None
    
    return redacted


def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redact sensitive information from a dict.
    
    Args:
        data: Dictionary that may contain sensitive data
        
    Returns:
        Dictionary with sensitive data redacted
    """
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = redact_dict(value)
        elif isinstance(value, str):
            if any(k in key.lower() for k in ['key', 'token', 'authorization']):
                result[key] = redact_api_key(value)
            else:
                result[key] = value
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item) if isinstance(item, dict)
                else redact_api_key(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    
    return result
