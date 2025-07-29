"""
最もシンプルなテストスクリプト
"""

import os
import sys
from pathlib import Path

# プロジェクトルートに移動
os.chdir(Path(__file__).parent)

# srcディレクトリをパスに追加
sys.path.insert(0, "src")

def test_imports():
    """必要なモジュールのインポートテスト"""
    try:
        print("📦 モジュールのインポートテスト...")
        
        # 基本ライブラリ
        from dotenv import load_dotenv
        load_dotenv()
        print("  ✅ dotenv")
        
        # プロジェクトのモジュール
        import constants as ct
        print("  ✅ constants")
        
        import utils
        print("  ✅ utils")
        
        # 環境変数の確認
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print(f"  ✅ OPENAI_API_KEY (先頭10文字: {openai_key[:10]}...)")
        else:
            print("  ❌ OPENAI_API_KEY が設定されていません")
            return False
        
        # 認証ファイルの確認
        auth_file = Path("secrets/service_account.json")
        if auth_file.exists():
            print("  ✅ service_account.json")
        else:
            print("  ❌ secrets/service_account.json が見つかりません")
            return False
        
        print("🎉 すべてのインポートテストが成功しました！")
        return True
        
    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_google_sheets():
    """Google Sheetsの接続テスト"""
    try:
        print("\n📊 Google Sheets接続テスト...")
        
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        import constants as ct
        
        # 認証
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            'secrets/service_account.json', scope
        )
        client = gspread.authorize(creds)
        
        # シートを開く
        sheet = client.open_by_url(ct.GOOGLE_SHEET_URL).sheet1
        rows = sheet.get_all_records()
        
        print(f"  ✅ スプレッドシート接続成功: {len(rows)}行のデータを取得")
        
        # 最初の3行を表示
        for i, row in enumerate(rows[:3], 1):
            q = row.get("質問", "")
            a = row.get("回答", "")
            print(f"  {i}. Q: {q[:30]}...")
            print(f"     A: {a[:30]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Sheets接続エラー: {e}")
        return False

if __name__ == "__main__":
    print("🧪 シンプルテストを開始します\n")
    
    # Step 1: インポートテスト
    if not test_imports():
        print("💥 インポートテストで失敗しました")
        sys.exit(1)
    
    # Step 2: Google Sheets接続テスト
    if not test_google_sheets():
        print("💥 Google Sheets接続テストで失敗しました")
        sys.exit(1)
    
    print("\n🎉 すべてのテストが成功しました！")
    print("👉 次のステップ: Streamlitアプリを起動してテストしてください")
    print("   コマンド: streamlit run src/main.py")