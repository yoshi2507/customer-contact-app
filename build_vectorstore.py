# build_vectorstore.py

from utils import create_vectorstore
from constants import DB_POLICY_PATH, DB_SUSTAINABILITY_PATH, DB_NAMES

# ベクトルDBを作成
create_vectorstore(DB_NAMES[DB_POLICY_PATH], DB_POLICY_PATH)
create_vectorstore(DB_NAMES[DB_SUSTAINABILITY_PATH], DB_SUSTAINABILITY_PATH)

print("ベクトルDBの作成が完了しました。")
