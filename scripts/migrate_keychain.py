#!/usr/bin/env python3
"""
Migrate API keys from old MeshRepair keychain service to NeuroTopo.

This script:
1. Reads API keys from the old "MeshRepair" keychain service
2. Stores them under the new "NeuroTopo" service
3. Optionally deletes the old entries
"""
import subprocess
import sys


OLD_SERVICE = "MeshRepair"
NEW_SERVICE = "NeuroTopo"
ACCOUNTS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]


def get_from_keychain(account: str, service: str) -> str | None:
    """Retrieve a secret from Mac Keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def store_in_keychain(account: str, service: str, value: str) -> bool:
    """Store a secret in Mac Keychain."""
    try:
        # First try to delete any existing entry
        subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True
        )
        # Add the new entry
        result = subprocess.run(
            ["security", "add-generic-password", "-s", service, "-a", account, "-w", value],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False


def delete_from_keychain(account: str, service: str) -> bool:
    """Delete a secret from Mac Keychain."""
    try:
        result = subprocess.run(
            ["security", "delete-generic-password", "-s", service, "-a", account],
            capture_output=True
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    print("=" * 50)
    print("NeuroTopo Keychain Migration")
    print(f"From: {OLD_SERVICE} → To: {NEW_SERVICE}")
    print("=" * 50)
    
    migrated = 0
    skipped = 0
    
    for account in ACCOUNTS:
        print(f"\n{account}:")
        
        # Check if already exists in new service
        new_value = get_from_keychain(account, NEW_SERVICE)
        if new_value:
            print(f"  ✓ Already exists in {NEW_SERVICE}")
            skipped += 1
            continue
        
        # Try to get from old service
        old_value = get_from_keychain(account, OLD_SERVICE)
        if not old_value:
            print(f"  - Not found in {OLD_SERVICE}")
            skipped += 1
            continue
        
        # Migrate to new service
        print(f"  Found in {OLD_SERVICE} ({len(old_value)} chars)")
        if store_in_keychain(account, NEW_SERVICE, old_value):
            print(f"  ✓ Copied to {NEW_SERVICE}")
            migrated += 1
        else:
            print(f"  ✗ Failed to store in {NEW_SERVICE}")
    
    print("\n" + "=" * 50)
    print(f"Migration complete: {migrated} migrated, {skipped} skipped")
    
    # Ask about cleanup
    if migrated > 0:
        print(f"\nDelete old entries from '{OLD_SERVICE}'? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            for account in ACCOUNTS:
                if get_from_keychain(account, OLD_SERVICE):
                    if delete_from_keychain(account, OLD_SERVICE):
                        print(f"  Deleted {account} from {OLD_SERVICE}")
                    else:
                        print(f"  Failed to delete {account}")
            print("Cleanup complete.")
        else:
            print("Old entries preserved.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
