"""
このファイルは、固定の文字列や数値などのデータを変数として一括管理するファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader


############################################################
# 共通変数の定義
############################################################

# ==========================================
# 環境変数関連
# ==========================================
class EnvironmentKeys:
    """環境変数キーの定数定義"""
    
    # OpenAI関連
    OPENAI_API_KEY = "OPENAI_API_KEY"
    
    # Slack関連
    SLACK_BOT_TOKEN = "SLACK_BOT_TOKEN"
    SLACK_USER_TOKEN = "SLACK_USER_TOKEN"
    
    # 検索関連
    SERP_API_KEY = "SERP_API_KEY"
    
    # Google関連
    GOOGLE_APPLICATION_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

# 必須環境変数のリスト
REQUIRED_SECRETS = [
    EnvironmentKeys.OPENAI_API_KEY,
    EnvironmentKeys.SLACK_BOT_TOKEN,
]

# オプション環境変数のリスト
OPTIONAL_SECRETS = [
    EnvironmentKeys.SLACK_USER_TOKEN,
    EnvironmentKeys.SERP_API_KEY,
    EnvironmentKeys.GOOGLE_APPLICATION_CREDENTIALS,
]

# ==========================================
# 画面表示系
# ==========================================
APP_NAME = "問い合わせ対応自動化AIエージェント"
CHAT_INPUT_HELPER_TEXT = "こちらからメッセージを送信してください。"
APP_BOOT_MESSAGE = "アプリが起動されました。"
USER_ICON_FILE_PATH = "./images/user_icon.jpg"
AI_ICON_FILE_PATH = "./images/ai_icon.jpg"
WARNING_ICON = ":material/warning:"
ERROR_ICON = ":material/error:"
SPINNER_TEXT = "回答生成中..."
SPINNER_CONTACT_TEXT = "問い合わせ内容を弊社担当者に送信中です。画面を操作せず、このままお待ちください。"
CONTACT_THANKS_MESSAGE = """
    このたびはお問い合わせいただき、誠にありがとうございます。
    担当者が内容を確認し、3営業日以内にご連絡いたします。
    ただし問い合わせ内容によっては、ご連絡いたしかねる場合がございます。
    もしお急ぎの場合は、お電話にてご連絡をお願いいたします。
"""


# ==========================================
# ユーザーフィードバック関連
# ==========================================
FEEDBACK_YES = "はい"
FEEDBACK_NO = "いいえ"

SATISFIED = "回答に満足した"
DISSATISFIED = "回答に満足しなかった"

FEEDBACK_REQUIRE_MESSAGE = "この回答はお役に立ちましたか？フィードバックをいただくことで、生成AIの回答の質が向上します。"
FEEDBACK_BUTTON_LABEL = "送信"
FEEDBACK_YES_MESSAGE = "ご満足いただけて良かったです！他にもご質問があれば、お気軽にお尋ねください！"
FEEDBACK_NO_MESSAGE = "ご期待に添えず申し訳ございません。今後の改善のために、差し支えない範囲でご満足いただけなかった理由を教えていただけますと幸いです。"
FEEDBACK_THANKS_MESSAGE = "ご回答いただき誠にありがとうございます。"


# ==========================================
# ログ出力系
# ==========================================
LOG_DIR_PATH = "./logs"
LOGGER_NAME = "ApplicationLog"
LOG_FILE = "application.log"


# ==========================================
# LLM設定系
# ==========================================
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.5
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 20
RETRIEVER_WEIGHTS = [0.5, 0.5]


# ==========================================
# トークン関連
# ==========================================
MAX_ALLOWED_TOKENS = 1000
ENCODING_KIND = "cl100k_base"


# ==========================================
# RAG参照用のデータソース系
# ==========================================
RAG_TOP_FOLDER_PATH = "./data/rag"

SUPPORTED_EXTENSIONS = {
    ".pdf": PyMuPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": lambda path: TextLoader(path, encoding="utf-8")
}

DB_ALL_PATH = "./.db_all"
DB_COMPANY_PATH = "./.db_company"

# ==========================================
# ナレッジベース（スプレッドシート）関連
# ==========================================
DB_KNOWLEDGE_PATH = "./.db_knowledge"

# ==========================================
# AIエージェント関連
# ==========================================
AI_AGENT_MAX_ITERATIONS = 5

DB_SERVICE_PATH = "./.db_service"
DB_CUSTOMER_PATH = "./.db_customer"
DB_MANUAL_PATH = "./.db_manual"
DB_POLICY_PATH = "./.db_policy"
DB_SUSTAINABILITY_PATH = "./.db_sustainability"
DB_KNOWLEDGE_PATH = "./.db_knowledge"

DB_NAMES = {
    DB_COMPANY_PATH: f"{RAG_TOP_FOLDER_PATH}/company",
    DB_SERVICE_PATH: f"{RAG_TOP_FOLDER_PATH}/service",
    DB_CUSTOMER_PATH: f"{RAG_TOP_FOLDER_PATH}/customer",
    DB_MANUAL_PATH: f"{RAG_TOP_FOLDER_PATH}/manual",
    DB_POLICY_PATH: f"{RAG_TOP_FOLDER_PATH}/policy",
    DB_SUSTAINABILITY_PATH: f"{RAG_TOP_FOLDER_PATH}/sustainability",
    DB_KNOWLEDGE_PATH: "GoogleSheets"
}

AI_AGENT_MODE_ON = "利用する"
AI_AGENT_MODE_OFF = "利用しない"

CONTACT_MODE_ON = "ON"
CONTACT_MODE_OFF = "OFF"

SEARCH_COMPANY_INFO_TOOL_NAME = "search_company_info_tool"
SEARCH_COMPANY_INFO_TOOL_DESCRIPTION = (
    "EcoTee社の企業情報、理念、組織、実績、株主優待制度などに関する質問に使用します。\n"
    "たとえば『EcoTeeってどんな会社？』『株主優待はある？』『どこに本社があるの？』『受賞歴を教えて』などの質問で使用されます。\n"
    "このToolは、会社の設立概要、代表者、事業内容、サステナビリティ方針、販売実績、株主向けの優待プランやイベント、連絡先情報などを含む文書を参照して回答します。\n"
    "※ サービスの使い方や商品仕様に関する質問は 'search_service_info_tool' や 'search_manual_info_tool' が適切です。"
)
SEARCH_SERVICE_INFO_TOOL_NAME = "search_service_info_tool"
SEARCH_SERVICE_INFO_TOOL_DESCRIPTION = (
    "EcoTee社が提供するTシャツ関連サービスや製品内容に関する質問に使用します。\n"
    "たとえば『EcoTeeの主要サービスを教えて』『法人向けプランはある？』『代行出荷って何？』『プレミアムTシャツの特徴は？』などの質問で使用されます。\n"
    "このToolは、サービス紹介・製品情報・カスタマイズ内容・素材の特徴・出荷オプション・価格帯など、製品/サービスの特徴に関する資料を参照して回答します。\n"
    "操作方法や注文手続き、FAQに関する質問は 'search_manual_info_tool' が適切です。"
)
SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME = "search_customer_communication_tool"
SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION = (
    "EcoTeeのお客様（例：佐藤花子さん）に関する行動履歴、購入傾向、問い合わせ内容、満足度、キャンペーン参加状況などに関する質問に使用します。\n"
    "たとえば『佐藤花子さんの購入履歴を教えて』『過去にどんな問い合わせがあった？』『エンゲージメントの評価は？』『どんな商品に興味がありそう？』などの質問で使用されます。\n"
    "このToolは、顧客のプロフィール、注文履歴、来店記録、サポート履歴、アンケート結果、SNS投稿履歴、会員ランク、ロイヤリティ情報、社内CRMスコアなどに基づいて回答します。\n"
    "※ 会社やサービスの一般的な特徴に関する質問は 'search_company_info_tool' や 'search_service_info_tool' が適切です。"
)
SEARCH_WEB_INFO_TOOL_NAME = "search_web_tool"
SEARCH_WEB_INFO_TOOL_DESCRIPTION = "自社サービス「HealthX」に関する質問で、Web検索が必要と判断した場合に使う"

SEARCH_MANUAL_INFO_TOOL_NAME = "search_manual_info_tool"
SEARCH_MANUAL_INFO_TOOL_DESCRIPTION = (
    "EcoTee Creatorの操作方法、注文手続き、FAQ、トラブル対応などのマニュアル資料に関する質問に使用します。\n"
    "たとえば『デザインツールの使い方を教えて』『注文の途中で止めた場合はどうなる？』『配送先を分けて送るには？』などの質問で使用されます。\n"
    "このToolは、操作ガイド・設定手順・FAQなどに基づいて回答します。\n"
    "このToolは、サービスの特徴や料金などを答える 'search_service_info_tool' とは異なり、具体的な『使い方』に特化しています。"
)
SEARCH_POLICY_INFO_TOOL_NAME = "search_policy_info_tool"
SEARCH_POLICY_INFO_TOOL_DESCRIPTION = (
    "EcoTeeの注文ルール、支払い方法、発送条件、返品・交換、定期購入、法人向け対応などの「利用規約・制度」に関する質問に使用します。\n"
    "たとえば『返品はいつまで可能ですか？』『定期購入の解約ルールを教えて』『海外配送は対応していますか？』『法人注文の特典は？』などの質問で使用されます。\n"
    "このToolは、キャンセルや変更の条件、梱包・ギフト対応、顧客サポート方針、サブスクリプション契約、特急配送のルール、CO2オフセットオプションなどを含む「サービス提供に関する各種規定資料」を参照して回答します。\n"
    "※ 操作方法や商品仕様に関する質問は 'search_manual_info_tool' や 'search_service_info_tool' が適切です。"
)
SEARCH_SUSTAINABILITY_INFO_TOOL_NAME = "search_sustainability_info_tool"
SEARCH_SUSTAINABILITY_INFO_TOOL_DESCRIPTION = (
    "EcoTee社の環境対策・サステナビリティ活動・リサイクル方針・エシカルな労働環境・地域貢献などに関する質問に使用します。\n"
    "たとえば『製造時のCO2排出はどう抑えている？』『再生可能エネルギーは活用している？』『リサイクルTシャツの仕組みは？』『GOTS認証は取得していますか？』『労働環境の取り組みは？』などの質問で使用されます。\n"
    "このToolは、素材選定方針、ゼロウェイスト設計、再生エネルギー利用、カーボンオフセットプログラム、地域貢献活動、環境認証、エシカル労働支援、未来目標など、EcoTeeの環境・倫理的な取り組みに関する資料を参照して回答します。\n"
    "※ 会社全体の基本情報や、サービス・商品の仕様に関する質問は 'search_company_info_tool' や 'search_service_info_tool' が適切です。"
)

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1l4EXisQ4QHEQ0MDw5kpIp-7EJFy4PnSSqBc01bKbzhA"
WEB_URL = "https://www.pip-maker.com/"

# ==========================================
# Slack連携関連
# ==========================================
EMPLOYEE_FILE_PATH = "./data/slack/従業員情報.csv"
INQUIRY_HISTORY_FILE_PATH = "./data/slack/問い合わせ対応履歴.csv"
CSV_ENCODING = "utf-8-sig"


# ==========================================
# プロンプトテンプレート
# ==========================================
SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT = "会話履歴と最新の入力をもとに、会話履歴なしでも理解できる独立した入力テキストを生成してください。"

NO_DOC_MATCH_MESSAGE = "回答に必要な情報が見つかりませんでした。弊社に関する質問・要望を、入力内容を変えて送信してください。"

SYSTEM_PROMPT_INQUIRY = """
    あなたは社内文書を基に、顧客からの問い合わせに対応するアシスタントです。
    以下の条件に基づき、ユーザー入力に対して回答してください。

    【条件】
    1. ユーザー入力内容と以下の文脈との間に関連性がある場合のみ、以下の文脈に基づいて回答してください。
    2. ユーザー入力内容と以下の文脈との関連性が明らかに低い場合、「回答に必要な情報が見つかりませんでした。弊社に関する質問・要望を、入力内容を変えて送信してください。」と回答してください。
    3. 憶測で回答せず、あくまで以下の文脈を元に回答してください。
    4. できる限り詳細に、マークダウン記法を使って回答してください。
    5. マークダウン記法で回答する際にhタグの見出しを使う場合、最も大きい見出しをh3としてください。
    6. 複雑な質問の場合、各項目についてそれぞれ詳細に回答してください。
    7. 必要と判断した場合は、以下の文脈に基づかずとも、一般的な情報を回答してください。

    {context}
"""

SYSTEM_PROMPT_EMPLOYEE_SELECTION = """
    # 命令
    以下の「顧客からの問い合わせ」に対して、社内のどの従業員が対応するかを
    判定する生成AIシステムを作ろうとしています。

    以下の「従業員情報」は、問い合わせに対しての一人以上の対応者候補のデータです。
    しかし、問い合わせ内容との関連性が薄い従業員情報が含まれている可能性があります。
    以下の「条件」に従い、従業員情報の中から、問い合わせ内容との関連性が特に高いと思われる
    従業員の「ID」をカンマ区切りで返してください。

    # 顧客からの問い合わせ
    {query}

    # 条件
    - 全ての従業員が、問い合わせ内容との関連性が高い（対応者候補である）と判断した場合は、
    全ての従業員の従業員IDをカンマ区切りで返してください。ただし、関連性が低い（対応者候補に含めるべきでない）
    と判断した場合は省いてください。
    - 特に、「過去の問い合わせ対応履歴」と、「対応可能な問い合わせカテゴリ」、また「現在の主要業務」を元に判定を
    行ってください。
    - 一人も対応者候補がいない場合、空文字を返してください。
    - 判定は厳しく行ってください。

    # 従業員情報
    {employee_context}

    # 出力フォーマット
    {format_instruction}
"""

SYSTEM_PROMPT_NOTICE_SLACK = """
    # 役割
    社内の問い合わせ対応を担うAIアシスタント。
    お客様からの問い合わせに対する返信メッセージ案の作成と、指定のメンバーにメンションを当ててSlackへの送信を行うアシスタント


    # 命令
    Slackの「customer-contact2」チャンネルで、メンバーIDが{slack_id_text}のメンバーに一度だけメンションを当て、
    お客様からの質問に対し、事実に基づいた「実用的かつ簡潔な対応案」を送信してください。


    # 送信先のチャンネル名
    customer-contact2


    # メッセージの通知先
    メンバーIDが{slack_id_text}のメンバー


    # メッセージ通知（メンション付け）のルール
    - メッセージ通知（メンション付け）は、メッセージの先頭で「一度だけ」行ってください。
    - メンション付けの行は、メンションのみとしてください。

    # 使用可能な情報源（※これ以外は使用不可）
    1. Googleスプレッドシートの社内Q&A
    URL : {GOOGLE_SHEET_URL}

    2. 公式Webサイト（pip-maker.com）
    URI : {WEB_URL}

    # メッセージの生成条件
    - 各項目について、できる限り長い文章量で、具体的に生成してください。

    - 「メッセージフォーマット」を使い、以下の各項目の文章を生成してください。
        - 【問い合わせ情報】の「カテゴリ」
        - 【問い合わせ情報】の「日時」
        - 【回答・対応案とその根拠】

    - 「顧客から弊社への問い合わせ内容」と、以下の「参考情報（スプレッドシートまたはWebサイトの内容）」のみを基に文章を生成してください。

    - 【問い合わせ情報】の「カテゴリ」は、【問い合わせ情報】の「問い合わせ内容」を基に適切なものを生成してください。

    - 【回答・対応案】について、以下の条件に従って生成してください。
        - 回答・対応案の「内容」と、それを選んだ「根拠」を、それぞれ最大3件まで生成してください。
        - 情報源が [スプレッドシート] または [pip-maker.com] であることを明記してください。
            例：「スプレッドシートの○○行」「pip-maker.comのFAQページ」など
        - 以下のような表現は禁止です：
            ・「社内で確認します」「ご案内できません」など曖昧な回答
            ・出典のない回答

    # 顧客から弊社への問い合わせ内容
    {query}


    # 参考情報:
    {knowledge_context}

    # 出力の表現トーン（任意で追加推奨）

    - 「です・ます調」で案内するような丁寧口調で書いてください。
    - 命令口調（例：「〜してください」）は避けてください。
    - 「私はAIです」「わかりません」などの表現は禁止です。

    ================================================
    # メッセージフォーマット

    【問い合わせ情報】
    ・問い合わせ内容: {query}
    ・対応カテゴリ:  （自動生成）
    ・問い合わせ者: 不明
    ・問い合わせ者メールアドレス: {user_email}
    ・日時: {now_datetime}

    --------------------

    【回答・対応案】
    ＜1つ目＞
    ●内容: ...（できるだけ簡潔かつ実用的に） 
    ●根拠: ...スプレッドシートの「◯◯」行、または pip-maker.com の「◯◯」ページ・セクションの内容を参考にしています。

    ＜2つ目＞
    ●内容: ...（できるだけ簡潔かつ実用的に） 
    ●根拠: ...スプレッドシートの「◯◯」行、または pip-maker.com の「◯◯」ページ・セクションの内容を参考にしています。

    ＜3つ目＞
    ●内容: ...（できるだけ簡潔かつ実用的に）
    ●根拠: ...スプレッドシートの「◯◯」行、または pip-maker.com の「◯◯」ページ・セクションの内容を参考にしています。

    --------------------
"""
# constants.py に追加するプロンプトテンプレート

SYSTEM_PROMPT_NOTICE_SLACK_CHANNEL = """
    # 役割
    社内の問い合わせ対応を担うAIアシスタント。
    適切な担当者が特定できない場合の緊急通知メッセージを作成するアシスタント

    # 命令
    Slackの「customer-contact2」チャンネルで、@channelを使って全メンバーに緊急通知を送信してください。
    適切な担当者が特定できない重要なお客様からの問い合わせに対し、迅速な対応を促すメッセージを作成してください。

    # 送信先のチャンネル名
    customer-contact2

    # 通知の性質
    - 適切な担当者が自動選定できなかった緊急案件
    - 全員で対応可能性を検討する必要がある問い合わせ
    - 迅速な初期対応が必要

    # 使用可能な情報源（※これ以外は使用不可）
    1. Googleスプレッドシートの社内Q&A
    URL : {GOOGLE_SHEET_URL}

    2. 公式Webサイト（pip-maker.com）
    URI : {WEB_URL}

    # メッセージの生成条件
    - 緊急性を伝える適切なトーンで生成してください
    - 「メッセージフォーマット」を使い、以下の各項目の文章を生成してください。
        - 【状況説明】適切な担当者が特定できなかった旨
        - 【問い合わせ情報】の「カテゴリ」
        - 【問い合わせ情報】の「日時」
        - 【初期対応案とその根拠】（可能な範囲で）
        - 【対応依頼】

    # 顧客から弊社への問い合わせ内容
    {query}

    # 参考情報:
    {knowledge_context}

    # 出力の表現トーン
    - 「です・ます調」で丁寧かつ緊急性を伝える口調で書いてください
    - 全員の協力を仰ぐ適切な表現を使用してください
    - 「適切な担当者が見つからない」ことを前向きに表現してください

    ================================================
    # メッセージフォーマット

    【状況説明】
    自動担当者選定システムでは適切な担当者を特定できませんでしたので、どなたか対応をお願いいたします。

    【問い合わせ情報】
    ・問い合わせ内容: {query}
    ・対応カテゴリ:  （自動生成 - 推定）
    ・問い合わせ者: 不明
    ・問い合わせ者メールアドレス: {user_email}
    ・日時: {now_datetime}

    --------------------

    【初期対応案】（参考情報より）
    ＜1つ目＞
    ●内容: ...（できるだけ簡潔かつ実用的に） 
    ●根拠: ...スプレッドシートの「◯◯」行、または pip-maker.com の「◯◯」ページ・セクションの内容を参考にしています。

    ＜2つ目＞
    ●内容: ...（できるだけ簡潔かつ実用的に） 
    ●根拠: ...スプレッドシートの「◯◯」行、または pip-maker.com の「◯◯」ページ・セクションの内容を参考にしています。

    --------------------

    【対応依頼】
    対応可能な方は、このメッセージにリアクション（👋）をお願いします。
    複数名で対応する場合は、スレッドで調整をお願いいたします。
"""

# ==========================================
# エラー・警告メッセージ
# ==========================================
COMMON_ERROR_MESSAGE = "このエラーが繰り返し発生する場合は、管理者にお問い合わせください。"
INITIALIZE_ERROR_MESSAGE = "初期化処理に失敗しました。"
CONVERSATION_LOG_ERROR_MESSAGE = "過去の会話履歴の表示に失敗しました。"
MAIN_PROCESS_ERROR_MESSAGE = "ユーザー入力に対しての処理に失敗しました。"
DISP_ANSWER_ERROR_MESSAGE = "回答表示に失敗しました。"
INPUT_TEXT_LIMIT_ERROR_MESSAGE = f"入力されたテキストの文字数が受付上限値（{MAX_ALLOWED_TOKENS}）を超えています。受付上限値を超えないよう、再度入力してください。"

# 環境変数関連エラーメッセージ
ENV_VALIDATION_ERROR_MESSAGE = "環境変数の設定に問題があります。必須の環境変数が不足している可能性があります。"
OPENAI_API_KEY_ERROR_MESSAGE = "OpenAI APIキーが正しく設定されていません。OPENAI_API_KEYを確認してください。"
SLACK_BOT_TOKEN_ERROR_MESSAGE = "Slack Bot Tokenが正しく設定されていません。SLACK_BOT_TOKENを確認してください。"
SERP_API_KEY_ERROR_MESSAGE = "検索APIキーが設定されていません。Web検索機能を利用するにはSERP_API_KEYを設定してください。"
GOOGLE_CREDENTIALS_ERROR_MESSAGE = "Google認証情報が設定されていません。スプレッドシート機能を使用するにはGOOGLE_APPLICATION_CREDENTIALSを設定してください。"

# 環境変数設定ガイドメッセージ
ENV_SETUP_GUIDE_MESSAGE = """
環境変数の設定方法：

1. .envファイルを作成し、以下の形式で設定してください：
    OPENAI_API_KEY=sk-your-openai-api-key
    SLACK_BOT_TOKEN=xoxb-your-slack-bot-token

2. または、システムの環境変数として設定してください。

3. 設定後、アプリケーションを再起動してください。
"""

# 環境変数設定チェック用メッセージ
ENV_STATUS_MESSAGES = {
    "required_missing": "❌ 必須環境変数が不足しています",
    "required_complete": "✅ 必須環境変数が設定されています",
    "optional_partial": "⚠️ 一部のオプション機能が利用できません",
    "optional_complete": "✅ 全ての機能が利用可能です",
    "validation_failed": "❌ 環境変数の検証に失敗しました",
    "validation_success": "✅ 環境変数の検証が完了しました"
}

# ==========================================
# スタイリング
# ==========================================
STYLE = """
<style>
    .stHorizontalBlock {
        margin-top: -14px;
    }
    .stChatMessage + .stHorizontalBlock {
        margin-left: 56px;
    }
    .stChatMessage + .stHorizontalBlock .stColumn:nth-of-type(2) {
        margin-left: -24px;
    }
    @media screen and (max-width: 480px) {
        .stChatMessage + .stHorizontalBlock {
            flex-wrap: nowrap;
            margin-left: 56px;
        }
        .stChatMessage + .stHorizontalBlock .stColumn:nth-of-type(2) {
            margin-left: -206px;
        }
    }
</style>
"""

# ==========================================
# 環境変数管理用ユーティリティ関数で使用する定数（新規追加）
# ==========================================

# 環境変数の分類
ENV_CATEGORIES = {
    "authentication": [EnvironmentKeys.OPENAI_API_KEY],
    "communication": [EnvironmentKeys.SLACK_BOT_TOKEN, EnvironmentKeys.SLACK_USER_TOKEN],
    "search": [EnvironmentKeys.SERP_API_KEY],
    "integration": [EnvironmentKeys.GOOGLE_APPLICATION_CREDENTIALS]
}

# 環境変数の説明
ENV_DESCRIPTIONS = {
    EnvironmentKeys.OPENAI_API_KEY: {
        "description": "OpenAI APIへのアクセスに必要なAPIキー",
        "format": "sk-で始まる文字列",
        "required": True,
        "setup_url": "https://platform.openai.com/api-keys"
    },
    EnvironmentKeys.SLACK_BOT_TOKEN: {
        "description": "SlackボットのBot User OAuth Token",
        "format": "xoxb-で始まる文字列",
        "required": True,
        "setup_url": "https://api.slack.com/apps"
    },
    EnvironmentKeys.SLACK_USER_TOKEN: {
        "description": "Slackユーザートークン（高度な機能用）",
        "format": "xoxp-で始まる文字列",
        "required": False,
        "setup_url": "https://api.slack.com/apps"
    },
    EnvironmentKeys.SERP_API_KEY: {
        "description": "SerpApi検索サービスのAPIキー",
        "format": "英数字の文字列",
        "required": False,
        "setup_url": "https://serpapi.com/"
    },
    EnvironmentKeys.GOOGLE_APPLICATION_CREDENTIALS: {
        "description": "Google Cloud認証情報ファイルのパス",
        "format": "JSONファイルへのパス",
        "required": False,
        "setup_url": "https://cloud.google.com/docs/authentication"
    }
}

# デフォルト設定値
DEFAULT_ENV_VALUES = {
    EnvironmentKeys.GOOGLE_APPLICATION_CREDENTIALS: "secrets/service_account.json"
}

# 環境変数検証ルール
ENV_VALIDATION_RULES = {
    EnvironmentKeys.OPENAI_API_KEY: {
        "min_length": 40,
        "prefix": "sk-",
        "pattern": r"^sk-[a-zA-Z0-9]{48,}$"
    },
    EnvironmentKeys.SLACK_BOT_TOKEN: {
        "min_length": 50,
        "prefix": "xoxb-",
        "pattern": r"^xoxb-[0-9]+-[0-9]+-[0-9]+-[a-zA-Z0-9]{24}$"
    },
    EnvironmentKeys.SLACK_USER_TOKEN: {
        "min_length": 50,
        "prefix": "xoxp-",
        "pattern": r"^xoxp-[0-9]+-[0-9]+-[0-9]+-[a-zA-Z0-9]{64}$"
    },
    EnvironmentKeys.SERP_API_KEY: {
        "min_length": 20,
        "pattern": r"^[a-zA-Z0-9]{20,}$"
    }
}

# 環境変数トラブルシューティングメッセージ
ENV_TROUBLESHOOTING = {
    "openai_format_error": "OpenAI APIキーは 'sk-' で始まる必要があります。正しいAPIキーを設定してください。",
    "slack_format_error": "Slack Bot Tokenは 'xoxb-' で始まる必要があります。正しいBot Tokenを設定してください。",
    "file_not_found": "指定されたファイルが見つかりません。ファイルパスが正しいか確認してください。",
    "permission_denied": "ファイルへのアクセス権限がありません。ファイルの権限を確認してください。",
    "json_invalid": "JSONファイルの形式が正しくありません。ファイルの内容を確認してください。"
}