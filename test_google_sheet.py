import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ====== スプレッドシートのURLをここに貼ってください ======
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1l4EXisQ4QHEQ0MDw5kpIp-7EJFy4PnSSqBc01bKbzhA"

# Google Sheets API に必要なスコープ（固定でOK）
SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# service_account.json を読み込んで認証
creds = ServiceAccountCredentials.from_json_keyfile_name(
    'secrets/service_account.json',
    SCOPES
)

# gspread クライアントの作成
client = gspread.authorize(creds)

# スプレッドシートを開く
sheet = client.open_by_url(SPREADSHEET_URL)

# 最初のシート（1枚目）を取得
worksheet = sheet.sheet1

# すべてのデータ（行）を取得
records = worksheet.get_all_records()

# 取得結果を表示（デバッグ用）
for i, row in enumerate(records, 1):
    print(f"{i}行目: {row}")
