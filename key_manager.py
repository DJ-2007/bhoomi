"""
BhoomiSetu API Key Manager
Handles cryptographically secure key generation, validation, rotation, and CLI management.
"""

import os
import sys
import json
import time
import secrets
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

KEYS_FILE = Path("api_keys.json")
DEFAULT_KEY_PREFIX = "bs_live_"

def _load_keys() -> Dict[str, Any]:
    if not KEYS_FILE.exists():
        return {"keys": []}
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"keys": []}

def _save_keys(data: Dict[str, Any]) -> None:
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def generate_key(name: str = "Default User", role: str = "admin") -> Tuple[str, Dict[str, Any]]:
    """Generates a new secure API key with prefix bs_live_ and saves it."""
    data = _load_keys()
    
    # 24 bytes hex = 48 chars
    token = secrets.token_hex(24)
    full_key = f"{DEFAULT_KEY_PREFIX}{token}"
    key_id = f"key_{secrets.token_hex(4)}"
    
    record = {
        "key_id": key_id,
        "name": name,
        "role": role,
        "key": full_key,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "status": "active",
        "last_used_at": None,
        "request_count": 0
    }
    
    data["keys"].append(record)
    _save_keys(data)
    
    # Also update or set in .env if desired
    _sync_env_file(full_key)
    return full_key, record

def _sync_env_file(primary_key: str) -> None:
    env_path = Path(".env")
    env_content = ""
    if env_path.exists():
        env_content = env_path.read_text(encoding="utf-8")
    
    if "BHOOMI_API_KEY=" in env_content:
        lines = env_content.splitlines()
        new_lines = []
        for line in lines:
            if line.startswith("BHOOMI_API_KEY="):
                new_lines.append(f"BHOOMI_API_KEY={primary_key}")
            else:
                new_lines.append(line)
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    else:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"\nBHOOMI_API_KEY={primary_key}\n")

def get_or_create_default_key() -> str:
    """Returns an existing active key or generates a default one."""
    env_key = os.environ.get("BHOOMI_API_KEY")
    data = _load_keys()
    
    # Check if env key is registered
    if env_key:
        for k in data.get("keys", []):
            if k.get("key") == env_key and k.get("status") == "active":
                return env_key

    # Check active keys in storage
    active_keys = [k for k in data.get("keys", []) if k.get("status") == "active"]
    if active_keys:
        return active_keys[0]["key"]
        
    # Generate default
    new_key, _ = generate_key(name="Primary Master Key", role="admin")
    return new_key

def validate_key(api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validates provided API key against persistent store and environment."""
    if not api_key:
        return False, None
    
    # Check .env first for speed
    env_key = os.environ.get("BHOOMI_API_KEY")
    if env_key and secrets.compare_digest(api_key, env_key):
        return True, {"name": "Master Key (Env)", "role": "admin", "status": "active"}

    data = _load_keys()
    for record in data.get("keys", []):
        if secrets.compare_digest(record.get("key", ""), api_key):
            if record.get("status") == "active":
                # Update usage stats
                record["last_used_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
                record["request_count"] = record.get("request_count", 0) + 1
                _save_keys(data)
                return True, record
            return False, None
            
    return False, None

def list_keys() -> None:
    data = _load_keys()
    keys = data.get("keys", [])
    if not keys:
        print("[!] No API keys found. Generate one with: python key_manager.py create")
        return
    print(f"\n{'='*75}")
    print(f"{'KEY ID':<12} | {'NAME':<20} | {'STATUS':<8} | {'REQUESTS':<8} | KEY")
    print(f"{'='*75}")
    for k in keys:
        raw_key = k["key"]
        masked = f"{raw_key[:12]}...{raw_key[-4:]}"
        print(f"{k.get('key_id', ''):<12} | {k.get('name', ''):<20} | {k.get('status', ''):<8} | {k.get('request_count', 0):<8} | {masked}")
    print(f"{'='*75}\n")

def revoke_key(key_identifier: str) -> bool:
    data = _load_keys()
    updated = False
    for k in data.get("keys", []):
        if k.get("key") == key_identifier or k.get("key_id") == key_identifier:
            k["status"] = "revoked"
            updated = True
    if updated:
        _save_keys(data)
        print(f"[+] Successfully revoked key '{key_identifier}'")
        return True
    print(f"[-] Key identifier '{key_identifier}' not found.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BhoomiSetu API Key Manager")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Generate a new API key")
    create_parser.add_argument("--name", default="Client Application", help="Identifier name for this key")
    create_parser.add_argument("--role", default="user", help="Role (admin/user)")

    subparsers.add_parser("list", help="List all API keys")
    
    subparsers.add_parser("get-default", help="Retrieve active default API key")

    revoke_parser = subparsers.add_parser("revoke", help="Revoke an existing API key")
    revoke_parser.add_argument("--key", required=True, help="Full key string or key_id")

    args = parser.parse_args()

    if args.command == "create":
        key, record = generate_key(name=args.name, role=args.role)
        print("\n[+] NEW API KEY GENERATED SUCCESSFULLY!")
        print(f"    Key ID   : {record['key_id']}")
        print(f"    Name     : {record['name']}")
        print(f"    API Key  : {key}")
        print("\n[!] Keep this key secure. Pass it in HTTP headers as 'X-API-Key' or 'Authorization: Bearer <key>'.\n")
    elif args.command == "list":
        list_keys()
    elif args.command == "get-default":
        key = get_or_create_default_key()
        print(f"\nActive Default Key: {key}\n")
    elif args.command == "revoke":
        revoke_key(args.key)
    else:
        parser.print_help()
