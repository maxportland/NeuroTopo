#!/usr/bin/env python3
"""
Store OpenAI API key in Mac Keychain.

Usage:
    python store_api_key.py
    
You will be prompted to enter the API key securely.
"""
import subprocess
import sys
import getpass

SERVICE_NAME = "NeuroTopo"
ACCOUNT_NAME = "OPENAI_API_KEY"


def store_in_keychain(service: str, account: str, password: str) -> bool:
    """Store a secret in Mac Keychain."""
    try:
        # First, try to delete any existing entry
        subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True,
            check=False  # Don't fail if it doesn't exist
        )
        
        # Add the new entry
        result = subprocess.run(
            [
                "security", "add-generic-password",
                "-s", service,
                "-a", account,
                "-w", password,
                "-U"  # Update if exists
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error storing in keychain: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def verify_keychain_entry(service: str, account: str) -> bool:
    """Verify the secret can be retrieved."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  Store OpenAI API Key in Mac Keychain")
    print("=" * 60)
    print()
    print(f"Service: {SERVICE_NAME}")
    print(f"Account: {ACCOUNT_NAME}")
    print()
    
    # Get the API key securely (hidden input)
    api_key = getpass.getpass("Enter your OpenAI API key: ")
    
    if not api_key:
        print("Error: No API key provided")
        sys.exit(1)
    
    if not api_key.startswith("sk-"):
        print("Warning: API key doesn't start with 'sk-', continuing anyway...")
    
    print()
    print("Storing in Keychain...")
    
    if store_in_keychain(SERVICE_NAME, ACCOUNT_NAME, api_key):
        print("✓ API key stored successfully!")
        
        # Verify it can be retrieved
        if verify_keychain_entry(SERVICE_NAME, ACCOUNT_NAME):
            print("✓ Verified: Key can be retrieved from Keychain")
        else:
            print("⚠ Warning: Could not verify key retrieval")
    else:
        print("✗ Failed to store API key")
        sys.exit(1)
    
    print()
    print("You can now use the test harness without setting OPENAI_API_KEY env var.")
    print(f"The key is stored under service '{SERVICE_NAME}', account '{ACCOUNT_NAME}'")


if __name__ == "__main__":
    main()
