"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
（エラーハンドリング統一適用版）
"""

############################################################
# ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import components as cn
import constants as ct
import os
# エラーハンドリング統一対応
from error_handler import (
    handle_initialization_error,
    ErrorHandlerContext,
    ErrorContext,
    ErrorLevel
)

############################################################
# 設定関連
############################################################
st.set_page_config(
    page_title=ct.APP_NAME
)

load_dotenv()

logger = logging.getLogger(ct.LOGGER_NAME)

############################################################
# 初期化処理
############################################################
try:
    initialize()
except Exception as e:
    # 統一エラーハンドラーを使用（stop=Trueで例外時に停止）
    handle_initialization_error(e, stop=True)

# アプリ起動時のログ出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)

############################################################
# 初期表示
############################################################
# タイトル表示
cn.display_app_title()

# サイドバー表示
cn.display_sidebar()

# AIメッセージの初期表示
cn.display_initial_ai_message()

############################################################
# スタイリング処理
############################################################
# 画面装飾を行う「CSS」を記述
st.markdown(ct.STYLE, unsafe_allow_html=True)

############################################################
# メールアドレス入力欄
############################################################
email = None
if st.session_state.contact_mode == ct.CONTACT_MODE_ON:
    email = st.text_input("ご連絡先メールアドレスを入力してください（必須）", key="user_email")

############################################################
# チャット入力の受け付け
############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)

############################################################
# 会話ログの表示
############################################################
# 統一エラーハンドラーのコンテキストマネージャーを使用
with ErrorHandlerContext(
    context=ErrorContext.UI_DISPLAY,
    show_in_ui=True,
    stop_execution=True
):
    cn.display_conversation_log(chat_message)

############################################################
# チャット送信時の処理
############################################################
if chat_message:
    if st.session_state.contact_mode == ct.CONTACT_MODE_ON and not email:
        st.error("お問い合わせを送信するには、メールアドレスを入力してください。")
        st.stop()
    
    # ==========================================
    # 会話履歴の上限を超えた場合、受け付けない
    # ==========================================
    # ユーザーメッセージのトークン数を取得
    input_tokens = len(st.session_state.enc.encode(chat_message))
    # トークン数が、受付上限を超えている場合にエラーメッセージを表示
    if input_tokens > ct.MAX_ALLOWED_TOKENS:
        with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
            st.error(ct.INPUT_TEXT_LIMIT_ERROR_MESSAGE)
            st.stop()
    # トークン数が受付上限を超えていない場合、会話ログ全体のトークン数に加算
    st.session_state.total_tokens += input_tokens

    # ==========================================
    # 1. ユーザーメッセージの表示
    # ==========================================
    logger.info({"message": chat_message})

    res_box = st.empty()
    with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
        st.markdown(chat_message)
    
    # ==========================================
    # 2. LLMからの回答取得 or 問い合わせ処理
    # ==========================================
    res_box = st.empty()
    
    # 統一エラーハンドラーのコンテキストマネージャーを使用
    with ErrorHandlerContext(
        context=ErrorContext.MAIN_PROCESS,
        show_in_ui=True,
        stop_execution=True,
        additional_info={
            "contact_mode": st.session_state.contact_mode,
            "message_length": len(chat_message),
            "input_tokens": input_tokens
        }
    ):
        if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
            with st.spinner(ct.SPINNER_TEXT):
                result = utils.execute_agent_or_chain(chat_message)
        else:
            with st.spinner(ct.SPINNER_CONTACT_TEXT):
                # === Debug出力（既存のまま保持） ===
                print(f"🔍 Slack通知モード開始: {chat_message}")
                print(f"🔍 環境変数確認:")
                print(f"  - SLACK_BOT_TOKEN: {utils.check_env_var_status('SLACK_BOT_TOKEN')}")
                print(f"  - SLACK_USER_TOKEN: {utils.check_env_var_status('SLACK_USER_TOKEN')}")
                print(f"  - SERP_API_KEY: {utils.check_env_var_status('SERP_API_KEY')}")
                
                result = utils.notice_slack(chat_message)
                print(f"🔍 Slack通知完了: {result}")
    
    # ==========================================
    # 3. 古い会話履歴を削除
    # ==========================================
    utils.delete_old_conversation_log(result)

    # ==========================================
    # 4. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        # 統一エラーハンドラーのコンテキストマネージャーを使用
        with ErrorHandlerContext(
            context=ErrorContext.UI_DISPLAY,
            show_in_ui=True,
            stop_execution=True
        ):
            cn.display_llm_response(result)
            logger.info({"message": result})
    
    # ==========================================
    # 5. 会話ログへの追加
    # ==========================================
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": result})

############################################################
# 6. ユーザーフィードバックのボタン表示
############################################################
if st.session_state.contact_mode == ct.CONTACT_MODE_OFF:
    cn.display_feedback_button()