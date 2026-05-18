import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

NEXT_PUBLIC_SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase() -> Client:
    if not NEXT_PUBLIC_SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise ValueError("NEXT_PUBLIC_SUPABASE_URL 또는 SUPABASE_SERVICE_ROLE_KEY가 설정되지 않았습니다.")

    return create_client(NEXT_PUBLIC_SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)