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
# AIエージェント関連
# ==========================================
AI_AGENT_MAX_ITERATIONS = 5

DB_SERVICE_PATH = "./.db_service"
DB_CUSTOMER_PATH = "./.db_customer"
DB_MANUAL_PATH = "./.db_manual"
DB_POLICY_PATH = "./.db_policy"
DB_SUSTAINABILITY_PATH = "./.db_sustainability"

DB_NAMES = {
    DB_COMPANY_PATH: f"{RAG_TOP_FOLDER_PATH}/company",
    DB_SERVICE_PATH: f"{RAG_TOP_FOLDER_PATH}/service",
    DB_CUSTOMER_PATH: f"{RAG_TOP_FOLDER_PATH}/customer",
    DB_MANUAL_PATH: f"{RAG_TOP_FOLDER_PATH}/manual",
    DB_POLICY_PATH: f"{RAG_TOP_FOLDER_PATH}/policy",
    DB_SUSTAINABILITY_PATH: f"{RAG_TOP_FOLDER_PATH}/sustainability"
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
    {context}

    # 出力フォーマット
    {format_instruction}
"""

SYSTEM_PROMPT_NOTICE_SLACK = """
    # 役割
    具体的で分量の多いメッセージの作成と、指定のメンバーにメンションを当ててSlackへの送信を行うアシスタント


    # 命令
    Slackの「動作検証用」チャンネルで、メンバーIDが{slack_id_text}のメンバーに一度だけメンションを当て、生成したメッセージを送信してください。


    # 送信先のチャンネル名
    動作検証用


    # メッセージの通知先
    メンバーIDが{slack_id_text}のメンバー


    # メッセージ通知（メンション付け）のルール
    - メッセージ通知（メンション付け）は、メッセージの先頭で「一度だけ」行ってください。
    - メンション付けの行は、メンションのみとしてください。


    # メッセージの生成条件
    - 各項目について、できる限り長い文章量で、具体的に生成してください。

    - 「メッセージフォーマット」を使い、以下の各項目の文章を生成してください。
        - 【問い合わせ情報】の「カテゴリ」
        - 【問い合わせ情報】の「日時」
        - 【回答・対応案とその根拠】

    - 「顧客から弊社への問い合わせ内容」と「従業員情報と過去の問い合わせ対応履歴」を基に文章を生成してください。

    - 【問い合わせ情報】の「カテゴリ」は、【問い合わせ情報】の「問い合わせ内容」を基に適切なものを生成してください。

    - 【回答・対応案】について、以下の条件に従って生成してください。
        - 回答・対応案の内容と、それが良いと判断した根拠を、それぞれ3つずつ生成してください。


    # 顧客から弊社への問い合わせ内容
    {query}


    # 従業員情報と過去の問い合わせ対応履歴
    {context}


    # メッセージフォーマット
    こちらは顧客問い合わせに対しての「担当者割り振り」と「回答・対応案の提示」を自動で行うAIアシスタントです。
    担当者は問い合わせ内容を確認し、対応してください。

    ================================================

    【問い合わせ情報】
    ・問い合わせ内容: {query}
    ・カテゴリ: 
    ・問い合わせ者: 山田太郎
    ・日時: {now_datetime}

    --------------------

    【回答・対応案】
    ＜1つ目＞
    ●内容: 
    ●根拠: 

    ＜2つ目＞
    ●内容: 
    ●根拠: 

    ＜3つ目＞
    ●内容: 
    ●根拠: 

    --------------------

    【参照資料】
    ・従業員情報.csv
    ・問い合わせ履歴.csv
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