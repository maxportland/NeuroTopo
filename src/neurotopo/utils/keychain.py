"""
Mac Keychain utilities for secure secret storage.
"""
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

SERVICE_NAME = "NeuroTopo"


def get_from_keychain(account: str, service: str = SERVICE_NAME) -> str | None:
    """
    Retrieve a secret from Mac Keychain.
    
    Args:
        account: The account name (e.g., "OPENAI_API_KEY")
        service: The service name (default: "NeuroTopo")
        
    Returns:
        The secret value, or None if not found
    """
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            logger.debug(f"Secret not found in keychain: {service}/{account}")
            return None
            
    except FileNotFoundError:
        logger.debug("'security' command not found - not on macOS?")
        return None
    except Exception as e:
        logger.warning(f"Error retrieving from keychain: {e}")
        return None


def get_openai_api_key() -> str | None:
    """
    Get OpenAI API key from environment or Keychain.
    
    Priority:
    1. OPENAI_API_KEY environment variable
    2. Mac Keychain (NeuroTopo/OPENAI_API_KEY)
    
    Returns:
        API key or None if not found
    """
    # First check environment variable
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        logger.debug("Using OPENAI_API_KEY from environment")
        return env_key
    
    # Fall back to keychain
    keychain_key = get_from_keychain("OPENAI_API_KEY")
    if keychain_key:
        logger.debug("Using OPENAI_API_KEY from Keychain")
        return keychain_key
    
    return None


def get_anthropic_api_key() -> str | None:
    """
    Get Anthropic API key from environment or Keychain.
    
    Priority:
    1. ANTHROPIC_API_KEY environment variable
    2. Mac Keychain (NeuroTopo/ANTHROPIC_API_KEY)
    
    Returns:
        API key or None if not found
    """
    # First check environment variable
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        logger.debug("Using ANTHROPIC_API_KEY from environment")
        return env_key
    
    # Fall back to keychain
    keychain_key = get_from_keychain("ANTHROPIC_API_KEY")
    if keychain_key:
        logger.debug("Using ANTHROPIC_API_KEY from Keychain")
        return keychain_key
    
    return None


def ensure_api_key(provider: str = "openai") -> str:
    """
    Get API key for the specified provider, raising an error if not found.
    
    Args:
        provider: "openai" or "anthropic"
        
    Returns:
        The API key
        
    Raises:
        ValueError: If no API key is found
    """
    if provider == "openai":
        key = get_openai_api_key()
        env_var = "OPENAI_API_KEY"
    elif provider == "anthropic":
        key = get_anthropic_api_key()
        env_var = "ANTHROPIC_API_KEY"
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    if not key:
        raise ValueError(
            f"No API key found for {provider}. "
            f"Set {env_var} environment variable or run: "
            f"python scripts/store_api_key.py"
        )
    
    return key
