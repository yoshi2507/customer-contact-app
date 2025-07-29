"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
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



############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # Agent Executorを作成
    initialize_agent_executor()


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []
        # 会話履歴の合計トークン数を加算する用の変数
        st.session_state.total_tokens = 0
        
        # 🆕 柔軟なキーワードマッチングのデバッグモード
        st.session_state.flexible_keyword_debug_mode = True
        # フィードバックボタンで「はい」を押下した後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_yes_flg = False
        # フィードバックボタンで「いいえ」を押下した後に入力エリアを表示するためのフラグ
        st.session_state.feedback_no_flg = False
        # LLMによる回答生成後、フィードバックボタンを表示するためのフラグ
        st.session_state.answer_flg = False
        # フィードバックボタンで「いいえ」を押下後、フィードバックを送信するための入力エリアからの入力を受け付ける変数
        st.session_state.dissatisfied_reason = ""
        # フィードバック送信後にThanksメッセージを表示するためのフラグ
        st.session_state.feedback_no_reason_send_flg = False
        # Retrieverデバッグモード（任意でON/OFF切り替え）
        st.session_state.retriever_debug_mode = True


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_logger():
    """
    ログ出力の設定
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


def initialize_agent_executor():
    """
    画面読み込み時にAgent Executor（AIエージェント機能の実行を担当するオブジェクト）を作成
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # === 追加: 初期化開始のログ ===
    logger.info("🚀 Agent Executor初期化を開始します")

    # すでにAgent Executorが作成済みの場合、後続の処理を中断
    if "agent_executor" in st.session_state:
        logger.info("Agent Executorは既に初期化済みです")
        return
    
    # 消費トークン数カウント用のオブジェクトを用意
    st.session_state.enc = tiktoken.get_encoding(ct.ENCODING_KIND)
    
    st.session_state.llm = ChatOpenAI(model_name=ct.MODEL, temperature=ct.TEMPERATURE, streaming=True)

    # 各Tool用のChainを作成
    logger.info("📚 RAGチェーンを作成中...")
    st.session_state.customer_doc_chain = utils.create_rag_chain(ct.DB_CUSTOMER_PATH)
    st.session_state.service_doc_chain = utils.create_rag_chain(ct.DB_SERVICE_PATH)
    st.session_state.company_doc_chain = utils.create_rag_chain(ct.DB_COMPANY_PATH)
    st.session_state.manual_doc_chain = utils.create_rag_chain(ct.DB_MANUAL_PATH)
    st.session_state.policy_doc_chain = utils.create_rag_chain(ct.DB_POLICY_PATH)
    st.session_state.sustainability_doc_chain = utils.create_rag_chain(ct.DB_SUSTAINABILITY_PATH)

    # ===== 🆕 ナレッジベース（スプレッドシート）のチェーン作成 =====
    logger.info("📊 ナレッジベース（スプレッドシート）チェーンを作成中...")
    st.session_state.knowledge_doc_chain = utils.create_knowledge_rag_chain()
    if st.session_state.knowledge_doc_chain is None:
        logger.warning("⚠️ ナレッジベースチェーンの作成に失敗しました")
    else:
        logger.info("✅ ナレッジベースチェーン作成完了")
        
    retriever = utils.create_retriever(ct.DB_ALL_PATH)
    st.session_state.rag_chain = utils.create_rag_chain(ct.DB_ALL_PATH)
    logger.info("✅ RAGチェーン作成完了")

    # ✅ 強制的にデバッグモードを実行（フラグに関係なく）
    logger.info("🔧 強制デバッグモードを開始します")
    
    try:
        # まず基本的なretrieverテストを実行
        test_queries = ["受賞歴を教えてください", "SNS投稿に関する特典はありますか？"]
        for query in test_queries:
            logger.info(f"🧪 テストクエリ: {query}")
            docs = retriever.get_relevant_documents(query)
            logger.info(f"📊 取得文書数: {len(docs)}")
            
            # 最初の3件の文書の詳細を確認
            for i, doc in enumerate(docs[:3]):
                logger.info(f"[{i+1}] ファイル: {doc.metadata.get('file_name', '不明')}")
                logger.info(f"    キーワード: {doc.metadata.get('top_keywords', '未設定')}")
                logger.info(f"    内容: {doc.page_content[:100]}...")
        
        # 柔軟マッチングテストを実行
        utils.test_flexible_keyword_filter()
        
    except Exception as e:
        logger.error(f"デバッグモード実行中にエラー: {e}")
        import traceback
        logger.error(f"詳細エラー: {traceback.format_exc()}")
    
    logger.info("🔧 強制デバッグモード完了")

    # Web検索用のToolを設定するためのオブジェクトを用意
    search = SerpAPIWrapper()

    # ✅ RetrieverデバッグモードがONかつ未実行の場合、一度だけ実行
    if st.session_state.get("retriever_debug_mode") and not st.session_state.get("retriever_debug_done"):
        logger.info("🔧 Retrieverデバッグモードを開始します")
        
        # 修正: 新しいテスト関数を使用
        try:
            utils.test_keyword_filter()
        except Exception as e:
            logger.error(f"デバッグテスト中にエラー: {e}")
            # fallback: 元のテスト関数を使用
            test_queries = [
                "SNS投稿に関する特典はありますか？",
                "海外配送は対応していますか？",
                "地域貢献活動はありますか？",
                "受賞歴を教えてください",
                "EcoTeeは地域社会への貢献活動をしていますか？"
            ]
            for query in test_queries:
                try:
                    utils.debug_retriever_with_keywords(query, retriever)
                except Exception as e2:
                    logger.error(f"個別クエリテストでエラー ({query}): {e2}")
        
        st.session_state.retriever_debug_done = True
        logger.info("🔧 Retrieverデバッグモード完了")

        # ✅ 🆕 柔軟なキーワードマッチングデバッグモードがONかつ未実行の場合、一度だけ実行
    if st.session_state.get("flexible_keyword_debug_mode") and not st.session_state.get("flexible_keyword_debug_done"):
        logger.info("🔧 柔軟なキーワードマッチングデバッグモードを開始します")
        
        try:
            # 新しい柔軟なマッチング機能をテスト
            utils.test_flexible_keyword_filter()
            
            # 特に「受賞歴を教えてください」の問題を重点的にテスト
            logger.info("🎯 特別テスト: 受賞歴クエリの詳細分析")
            utils.debug_flexible_keyword_matching("受賞歴を教えてください", retriever)
            
        except Exception as e:
            logger.error(f"柔軟マッチングテスト中にエラー: {e}")
            # fallback: 既存のテスト関数を使用
            try:
                utils.test_keyword_filter()
            except Exception as e2:
                logger.error(f"fallbackテストもエラー: {e2}")
        
        st.session_state.flexible_keyword_debug_done = True
        logger.info("🔧 柔軟なキーワードマッチングデバッグモード完了")

    # Web検索用のToolを設定するためのオブジェクトを用意
    search = SerpAPIWrapper()
    # Agent Executorに渡すTool一覧を用意
    tools = [
        # 会社に関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_COMPANY_INFO_TOOL_NAME,
            func=utils.run_company_doc_chain,
            description=ct.SEARCH_COMPANY_INFO_TOOL_DESCRIPTION
        ),
        # サービスに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_SERVICE_INFO_TOOL_NAME,
            func=utils.run_service_doc_chain,
            description=ct.SEARCH_SERVICE_INFO_TOOL_DESCRIPTION
        ),
        # 顧客とのやり取りに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_NAME,
            func=utils.run_customer_doc_chain,
            description=ct.SEARCH_CUSTOMER_COMMUNICATION_INFO_TOOL_DESCRIPTION
        ),
        # Web検索用のTool
        Tool(
            name = ct.SEARCH_WEB_INFO_TOOL_NAME,
            func=search.run,
            description=ct.SEARCH_WEB_INFO_TOOL_DESCRIPTION
        ),
        # マニュアルに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_MANUAL_INFO_TOOL_NAME,
            func=utils.run_manual_doc_chain,
            description=ct.SEARCH_MANUAL_INFO_TOOL_DESCRIPTION
        ),
        # 利用規約に関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_POLICY_INFO_TOOL_NAME,
            func=utils.run_policy_doc_chain,
            description=ct.SEARCH_POLICY_INFO_TOOL_DESCRIPTION
        ),
        # サステナビリティに関するデータ検索用のTool
        Tool(
            name=ct.SEARCH_SUSTAINABILITY_INFO_TOOL_NAME,
            func=utils.run_sustainability_doc_chain,
            description=ct.SEARCH_SUSTAINABILITY_INFO_TOOL_DESCRIPTION
        )
    ]

    # Agent Executorの作成
    st.session_state.agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=ct.AI_AGENT_MAX_ITERATIONS,
        early_stopping_method="generate",
        handle_parsing_errors=True
    )