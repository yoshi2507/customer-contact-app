"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
（起動時間短縮版 + エラーハンドリング統一適用版）
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
from dotenv import load_dotenv
import streamlit as st
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import utils
import constants as ct

# 環境変数管理統一対応
from environment_manager import (
    get_environment_manager,
    EnvironmentError,
    EnvironmentKeys,
    check_environment_health
)

# エラーハンドリング統一対応
from error_handler import (
    ErrorHandlerContext,
    ErrorContext,
    ErrorLevel,
    error_handler,
    handle_data_processing_error
)

############################################################
# 設定関連
############################################################
load_dotenv()

############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理（最適化版 + エラーハンドリング統一版）
    """
    # 必須の初期化のみ実行
    initialize_session_state()
    initialize_session_id()
    initialize_logger()
    
    # LLMとエンコーダーのみ先に初期化（軽量）
    initialize_basic_components()
    
    # 重い処理は遅延読み込みでマーク
    st.session_state.lazy_init_required = True

def initialize_session_state():
    """
    初期化データの用意（変更なし）
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.total_tokens = 0
        
        # フラグ類
        st.session_state.flexible_keyword_debug_mode = True
        st.session_state.feedback_yes_flg = False
        st.session_state.feedback_no_flg = False
        st.session_state.answer_flg = False
        st.session_state.dissatisfied_reason = ""
        st.session_state.feedback_no_reason_send_flg = False
        st.session_state.retriever_debug_mode = True

def initialize_session_id():
    """
    セッションIDの作成（変更なし）
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex

def initialize_logger():
    """
    ログ出力の設定（変更なし）
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)

@error_handler(
    context=ErrorContext.INITIALIZATION,
    level=ErrorLevel.CRITICAL
)

def validate_environment_variables():
    """
    環境変数の事前検証（🆕 新機能）
    
    🔧 機能：
    - 必須環境変数の存在確認
    - オプション環境変数の状態確認
    - 環境設定の健全性チェック
    - 検証結果のセッション状態保存
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info("🔍 環境変数事前検証開始")
    
    env_manager = get_environment_manager()
    
    try:
        # === Step 1: 必須環境変数の検証 ===
        logger.info("📋 必須環境変数を検証中...")
        validation_result = env_manager.validate_required_secrets()
        
        # 検証失敗の場合はエラー
        missing_keys = [k for k, v in validation_result.items() if not v]
        if missing_keys:
            error_msg = f"必須環境変数が不足しています: {missing_keys}"
            logger.critical(error_msg)
            
            # UI にエラーメッセージを表示
            st.error(f"""
            ❌ **システム初期化エラー**
            
            必須の環境変数が設定されていません：
            {', '.join(missing_keys)}
            
            以下の環境変数を設定してください：
            - `OPENAI_API_KEY`: OpenAI APIキー
            - `SLACK_BOT_TOKEN`: Slack Bot Token (xoxb-で始まる)
            
            設定後、ページを再読み込みしてください。
            """)
            
            raise EnvironmentError(error_msg)
        
        # 成功時の詳細ログ
        logger.info("✅ 必須環境変数検証完了")
        for key, is_valid in validation_result.items():
            masked_value = "設定済み"
            try:
                value = env_manager.get_secret(key, required=False, mask_in_logs=False)
                if value:
                    masked_value = f"設定済み ({env_manager._mask_secret(value)})"
            except:
                pass
            logger.info(f"  ✅ {key}: {masked_value}")
        
        # === Step 2: オプション環境変数の状態確認 ===
        logger.info("📊 オプション環境変数の状態確認中...")
        optional_status = {}
        for key in ct.OPTIONAL_SECRETS:
            try:
                value = env_manager.get_secret(key, required=False, mask_in_logs=True)
                if value:
                    optional_status[key] = "設定済み"
                    logger.info(f"  ✅ {key}: 設定済み")
                else:
                    optional_status[key] = "未設定"
                    logger.info(f"  ⚠️ {key}: 未設定（オプション）")
            except Exception as e:
                optional_status[key] = f"エラー: {str(e)}"
                logger.warning(f"  ❌ {key}: エラー {str(e)}")
        
        # === Step 3: 環境設定の健全性チェック ===
        logger.info("🩺 環境設定の健全性チェック中...")
        
        # OpenAI設定の基本チェック
        try:
            openai_key = env_manager.get_secret(EnvironmentKeys.OPENAI_API_KEY, required=True)
            if not openai_key.startswith('sk-'):
                logger.warning("⚠️ OpenAI APIキーの形式が正しくない可能性があります")
            else:
                logger.info("✅ OpenAI APIキーの形式が正常です")
        except EnvironmentError:
            pass  # 既に上記で処理済み
        
        # Slack設定の基本チェック
        try:
            slack_token = env_manager.get_secret(EnvironmentKeys.SLACK_BOT_TOKEN, required=True)
            if not slack_token.startswith('xoxb-'):
                logger.warning("⚠️ Slack Bot Tokenの形式が正しくない可能性があります")
            else:
                logger.info("✅ Slack Bot Tokenの形式が正常です")
        except EnvironmentError:
            pass  # 既に上記で処理済み
        
        # === Step 4: 検証結果のセッション状態保存 ===
        st.session_state.env_validation_passed = True
        st.session_state.env_validation_result = validation_result
        st.session_state.env_optional_status = optional_status
        st.session_state.env_validation_timestamp = logging.Formatter().formatTime()
        
        # === Step 5: 成功時のサマリー表示 ===
        total_required = len(validation_result)
        total_optional_set = len([k for k, v in optional_status.items() if v == "設定済み"])
        
        logger.info(f"🎉 環境変数検証完了サマリー:")
        logger.info(f"  - 必須環境変数: {total_required}/{total_required} 設定済み")
        logger.info(f"  - オプション環境変数: {total_optional_set}/{len(optional_status)} 設定済み")
        
        # 開発時向けのヒント
        if total_optional_set < len(optional_status):
            missing_optional = [k for k, v in optional_status.items() if v != "設定済み"]
            logger.info(f"💡 利用可能な追加機能: {missing_optional}")
        
    except EnvironmentError as e:
        # 初期化エラーとして処理
        logger.critical(f"環境変数検証失敗: {e}")
        st.session_state.env_validation_passed = False
        raise

@error_handler(
    context=ErrorContext.INITIALIZATION,
    level=ErrorLevel.CRITICAL
)

def initialize_basic_components():
    """
    基本コンポーネントのみ初期化（軽量）（🔧 環境変数統一対応）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info("🚀 基本コンポーネント初期化開始")
    
    # 消費トークン数カウント用のオブジェクト
    if "enc" not in st.session_state:
        st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    
    # 🔧 LLMオブジェクト（環境変数統一版）
    if "llm" not in st.session_state:
        # 環境変数からOpenAI APIキーを統一取得
        env_manager = get_environment_manager()
        try:
            openai_api_key = env_manager.get_secret(
                EnvironmentKeys.OPENAI_API_KEY,
                required=True
            )
            
            # OpenAI用に環境変数を設定（LangChainが参照するため）
            os.environ[EnvironmentKeys.OPENAI_API_KEY] = openai_api_key
            
            # ChatOpenAIオブジェクトの作成
            st.session_state.llm = ChatOpenAI(
                model_name=ct.MODEL, 
                temperature=ct.TEMPERATURE, 
                streaming=True
            )
            
            logger.info("✅ LLMオブジェクト初期化完了")
            
        except EnvironmentError as e:
            logger.critical(f"OpenAI設定エラー: {e}")
            raise Exception(f"OpenAI APIキーの設定に問題があります: {e}")
    
    logger.info("✅ 基本コンポーネント初期化完了")

def initialize_heavy_components():
    """
    重いコンポーネントの遅延初期化（エラーハンドリング統一版）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info("🔄 重いコンポーネントの遅延初期化開始")
    
    # すでに初期化済みの場合はスキップ
    if "agent_executor" in st.session_state:
        logger.info("Agent Executorは既に初期化済みです")
        return
    
    # 統一エラーハンドラーのコンテキストマネージャーを使用
    with ErrorHandlerContext(
        context=ErrorContext.INITIALIZATION,
        level=ErrorLevel.CRITICAL,
        show_in_ui=False  # 初期化エラーは呼び出し元で制御
    ):
        # RAGチェーン作成
        logger.info("📚 RAGチェーンを作成中...")
        st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
        st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
        st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
        st.session_state.manual_doc_chain = utils.create_rag_chain(ct.DB_MANUAL_PATH)
        st.session_state.policy_doc_chain = utils.create_rag_chain(ct.DB_POLICY_PATH)
        st.session_state.sustainability_doc_chain = utils.create_rag_chain(ct.DB_SUSTAINABILITY_PATH)
        
        # ナレッジベースチェーン
        logger.info("📊 ナレッジベースチェーンを作成中...")
        st.session_state.knowledge_doc_chain = utils.create_knowledge_rag_chain()
        
        # 全体RAGチェーン
        st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)
        logger.info("✅ RAGチェーン作成完了")
        
        # デバッグ処理（条件付き実行）
        run_debug_if_needed()
        
        # Agent Executor作成
        create_agent_executor()
        
        # 遅延初期化完了フラグ
        st.session_state.lazy_init_required = False
        logger.info("✅ 重いコンポーネント初期化完了")

@error_handler(
    context=ErrorContext.DATA_PROCESSING,
    level=ErrorLevel.WARNING
)
def run_debug_if_needed():
    """
    デバッグ処理（軽量版）（エラーハンドリング統一版）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # デバッグモードが無効の場合はスキップ
    debug_enabled = st.session_state.get("retriever_debug_mode", False)
    if not debug_enabled:
        logger.info("🔧 デバッグモードが無効のためスキップ")
        return
    
    # 既に実行済みの場合はスキップ
    if st.session_state.get("retriever_debug_done", False):
        logger.info("🔧 デバッグは既に実行済み")
        return
    
    # 軽量デバッグを実行
    utils.run_lightweight_debug()

@error_handler(
    context=ErrorContext.INITIALIZATION,
    level=ErrorLevel.CRITICAL
)
def create_agent_executor():
    """
    Agent Executor作成（🔧 環境変数統一対応）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info("🤖 Agent Executor作成中...")
    
    # 🔧 Web検索用のTool（環境変数統一版）
    env_manager = get_environment_manager()
    try:
        # SERP API キーの確認
        serp_api_key = env_manager.get_secret(
            EnvironmentKeys.SERP_API_KEY,
            required=False
        )
        
        if serp_api_key:
            # SERP API が設定されている場合は Web検索を有効化
            os.environ[EnvironmentKeys.SERP_API_KEY] = serp_api_key
            search = SerpAPIWrapper()
            web_search_available = True
            logger.info("✅ Web検索機能が利用可能です")
        else:
            # SERP API が未設定の場合はダミー検索を使用
            def dummy_search(query):
                return "Web検索機能は利用できません。SERP_API_KEYを設定してください。"
            search = type('DummySearch', (), {'run': dummy_search})()
            web_search_available = False
            logger.warning("⚠️ SERP_API_KEYが未設定のため、Web検索機能は利用できません")
            
    except Exception as e:
        logger.warning(f"⚠️ Web検索設定エラー: {e}")
        # フォールバック：ダミー検索
        def dummy_search(query):
            return f"Web検索でエラーが発生しました: {str(e)}"
        search = type('DummySearch', (), {'run': dummy_search})()
        web_search_available = False
    
    # Tool一覧作成
    tools = [
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION + (
                "" if web_search_available else " （注意：現在Web検索は利用できません）"
            )
        ),
        Tool(
            name=ct.SEARCH_MANUAL_INFO_TOOL_NAME,
            func=utils.run_manual_doc_chain,
            description=ct.SEARCH_MANUAL_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_POLICY_INFO_TOOL_NAME,
            func=utils.run_policy_doc_chain,
            description=ct.SEARCH_POLICY_INFO_TOOL_DESCRIPTION
        ),
        Tool(
            name=ct.SEARCH_SUSTAINABILITY_INFO_TOOL_NAME,
            func=utils.run_sustainability_doc_chain,
            description=ct.SEARCH_SUSTAINABILITY_INFO_TOOL_DESCRIPTION
        )
    ]
    
    # Agent Executor作成
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )
    
    # Agent Executorの設定情報をログ出力
    logger.info(f"✅ Agent Executor作成完了:")
    logger.info(f"  - ツール数: {len(tools)}")
    logger.info(f"  - Web検索: {'有効' if web_search_available else '無効'}")
    logger.info(f"  - 最大反復回数: {ct.AI_AGENT_MAX_ITERATIONS}")

# ユーティリティ関数：遅延初期化チェック
def ensure_heavy_components_loaded():
    """
    重いコンポーネントが読み込まれていることを確認
    """
    if st.session_state.get("lazy_init_required", True):
        initialize_heavy_components()

@error_handler(
    context=ErrorContext.INITIALIZATION,
    level=ErrorLevel.WARNING
)
def force_initialize_if_needed():
    """
    utils.pyから直接呼び出される場合の強制初期化（エラーハンドリング統一版）
    """
    if "agent_executor" not in st.session_state:
        initialize_heavy_components()

def get_initialization_status():
    """
    初期化状態の取得
    
    Returns:
        初期化状態の辞書
    """
    env_manager = get_environment_manager()
    
    return {
        "session_initialized": "session_id" in st.session_state,
        "logger_initialized": len(logging.getLogger(ct.LOGGER_NAME).handlers) > 0,
        "env_validation_passed": st.session_state.get("env_validation_passed", False),
        "basic_components_loaded": "llm" in st.session_state and "enc" in st.session_state,
        "heavy_components_loaded": "agent_executor" in st.session_state,
        "lazy_init_required": st.session_state.get("lazy_init_required", True),
        "environment_health": check_environment_health(),
        "environment_summary": env_manager.get_environment_status()
    }

def validate_initialization_completeness():
    """
    初期化の完全性チェック
    
    Returns:
        tuple: (完全性フラグ, 不足コンポーネントリスト)
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    required_components = [
        ("session_id", "セッションID"),
        ("enc", "トークンエンコーダー"),
        ("llm", "LLMオブジェクト"),
        ("env_validation_passed", "環境変数検証")
    ]
    
    missing_components = []
    
    for component_key, component_name in required_components:
        if component_key not in st.session_state:
            missing_components.append(component_name)
        elif component_key == "env_validation_passed" and not st.session_state.get(component_key):
            missing_components.append(component_name)
    
    is_complete = len(missing_components) == 0
    
    if is_complete:
        logger.info("✅ 基本初期化が完全に完了しています")
    else:
        logger.warning(f"⚠️ 不足している初期化コンポーネント: {missing_components}")
    
    return is_complete, missing_components

def reinitialize_if_needed():
    """
    必要に応じて再初期化を実行
    
    Returns:
        bool: 再初期化が実行された場合True
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    is_complete, missing_components = validate_initialization_completeness()
    
    if not is_complete:
        logger.info(f"🔄 不完全な初期化を検出、再初期化を実行: {missing_components}")
        
        try:
            # 環境変数検証から再実行
            if "env_validation_passed" not in st.session_state or not st.session_state.env_validation_passed:
                validate_environment_variables()
            
            # 基本コンポーネントの再初期化
            if "llm" not in st.session_state or "enc" not in st.session_state:
                initialize_basic_components()
            
            logger.info("✅ 再初期化完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ 再初期化に失敗: {e}")
            raise
    
    return False