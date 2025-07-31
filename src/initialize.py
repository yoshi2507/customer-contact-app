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
def initialize_basic_components():
    """
    基本コンポーネントのみ初期化（軽量）（エラーハンドリング統一版）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info("🚀 基本コンポーネント初期化開始")
    
    # 消費トークン数カウント用のオブジェクト
    if "enc" not in st.session_state:
        st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    
    # LLMオブジェクト
    if "llm" not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model_name=ct.MODEL, 
            temperature=ct.TEMPERATURE, 
            streaming=True
        )
    
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
    Agent Executor作成（エラーハンドリング統一版）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    logger.info("🤖 Agent Executor作成中...")
    
    # Web検索用のTool
    search = SerpAPIWrapper()
    
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
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
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
    
    logger.info("✅ Agent Executor作成完了")

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