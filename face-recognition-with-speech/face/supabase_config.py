"""
Supabase Configuration Module
Initializes and configures Supabase client connection
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_KEY)

def get_supabase_client() -> Client:
    """
    Initialize and return Supabase client
    
    Returns:
        Client: Supabase client instance
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return client

def get_supabase_admin_client() -> Client:
    """
    Initialize and return Supabase admin client (for privileged operations)
    
    Returns:
        Client: Supabase admin client instance
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env file")
    
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return client
