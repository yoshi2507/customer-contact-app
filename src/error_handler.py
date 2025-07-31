"""
統一エラーハンドラーモジュール

このモジュールは、アプリケーション全体のエラーハンドリングを統一するための
クラスと関数を提供します。

主な機能:
- 統一されたログ出力形式
- コンテキスト別エラーメッセージ
- Streamlit UI統合
- デコレーター/コンテキストマネージャー対応
"""

import logging
import traceback
import streamlit as st
from typing import Optional, Any, Callable
from enum import Enum
import constants as ct

class ErrorLevel(Enum):
    """エラーレベルの定義"""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ErrorContext(Enum):
    """エラー発生コンテキストの定義"""
    INITIALIZATION = "initialization"
    MAIN_PROCESS = "main_process"
    SLACK_NOTIFICATION = "slack_notification"
    RAG_CHAIN = "rag_chain"
    AGENT_EXECUTION = "agent_execution"
    DATA_PROCESSING = "data_processing"
    UI_DISPLAY = "ui_display"
    VECTORSTORE_CREATION = "vectorstore_creation"

class UnifiedErrorHandler:
    """統一エラーハンドラークラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(ct.LOGGER_NAME)
    
    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        level: ErrorLevel = ErrorLevel.ERROR,
        user_message: Optional[str] = None,
        show_in_ui: bool = False,
        stop_execution: bool = False,
        return_value: Any = None,
        additional_info: Optional[dict] = None
    ) -> Any:
        """
        統一エラーハンドリング
        
        Args:
            error: 発生した例外
            context: エラー発生コンテキスト
            level: エラーレベル
            user_message: ユーザー向けメッセージ
            show_in_ui: UIにエラーを表示するか
            stop_execution: 実行を停止するか
            return_value: エラー時の戻り値
            additional_info: 追加情報
        
        Returns:
            return_valueまたはNone
        """
        
        # 1. ログ出力の統一
        error_details = self._build_error_details(error, context, additional_info)
        self._log_error(error_details, level)
        
        # 2. UI表示の統一
        if show_in_ui:
            self._show_ui_error(user_message or error_details["user_friendly_message"])
        
        # 3. 実行制御
        if stop_execution:
            st.stop()
        
        return return_value
    
    def _build_error_details(
        self, 
        error: Exception, 
        context: ErrorContext, 
        additional_info: Optional[dict]
    ) -> dict:
        """エラー詳細情報の構築"""
        
        details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context.value,
            "traceback": traceback.format_exc(),
            "session_id": getattr(st.session_state, 'session_id', 'unknown'),
        }
        
        # コンテキスト別のメッセージ設定
        context_messages = {
            ErrorContext.INITIALIZATION: {
                "log_message": "初期化処理でエラー発生",
                "user_friendly_message": ct.INITIALIZE_ERROR_MESSAGE
            },
            ErrorContext.MAIN_PROCESS: {
                "log_message": "メイン処理でエラー発生",
                "user_friendly_message": ct.MAIN_PROCESS_ERROR_MESSAGE
            },
            ErrorContext.SLACK_NOTIFICATION: {
                "log_message": "Slack通知処理でエラー発生",
                "user_friendly_message": "お問い合わせを受け付けましたが、システムエラーが発生しました。直接お電話でお問い合わせください。"
            },
            ErrorContext.RAG_CHAIN: {
                "log_message": "RAGチェーン処理でエラー発生",
                "user_friendly_message": "申し訳ございませんが、回答生成でエラーが発生しました。"
            },
            ErrorContext.AGENT_EXECUTION: {
                "log_message": "AIエージェント実行でエラー発生",
                "user_friendly_message": "申し訳ございませんが、AIエージェント処理でエラーが発生しました。通常モードで再試行してください。"
            },
            ErrorContext.DATA_PROCESSING: {
                "log_message": "データ処理でエラー発生",
                "user_friendly_message": "データ処理でエラーが発生しました。"
            },
            ErrorContext.UI_DISPLAY: {
                "log_message": "UI表示でエラー発生",
                "user_friendly_message": ct.DISP_ANSWER_ERROR_MESSAGE
            },
            ErrorContext.VECTORSTORE_CREATION: {
                "log_message": "ベクトルストア作成でエラー発生",
                "user_friendly_message": "データベース初期化でエラーが発生しました。"
            }
        }
        
        messages = context_messages.get(context, {
            "log_message": f"不明なコンテキスト({context.value})でエラー発生",
            "user_friendly_message": "システムエラーが発生しました。"
        })
        
        details.update(messages)
        
        # 追加情報の付与
        if additional_info:
            details["additional_info"] = additional_info
        
        return details
    
    def _log_error(self, details: dict, level: ErrorLevel):
        """ログ出力の統一処理"""
        
        # 基本ログメッセージの構築
        log_message = (
            f"❌ {details['log_message']}\n"
            f"   エラータイプ: {details['error_type']}\n"
            f"   エラーメッセージ: {details['error_message']}\n"
            f"   コンテキスト: {details['context']}\n"
            f"   セッションID: {details['session_id']}"
        )
        
        # 追加情報があれば付与
        if "additional_info" in details:
            log_message += f"\n   追加情報: {details['additional_info']}"
        
        # レベル別ログ出力
        if level == ErrorLevel.WARNING:
            self.logger.warning(log_message)
        elif level == ErrorLevel.ERROR:
            self.logger.error(log_message)
            self.logger.error(f"詳細スタックトレース:\n{details['traceback']}")
        elif level == ErrorLevel.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"詳細スタックトレース:\n{details['traceback']}")
        
        # 標準出力にも出力（Streamlit Cloud対応）
        if level in [ErrorLevel.ERROR, ErrorLevel.CRITICAL]:
            print(f"🔍 統一エラーハンドラー出力: {details['error_type']} - {details['error_message']}")
            print(f"🔍 コンテキスト: {details['context']}")
    
    def _show_ui_error(self, message: str):
        """UI エラー表示の統一処理"""
        # constants.pyのbuild_error_message関数と同等の処理
        full_message = f"{message}\n{ct.COMMON_ERROR_MESSAGE}"
        st.error(full_message, icon=ct.ERROR_ICON)

# デコレーター版のエラーハンドラー
def error_handler(
    context: ErrorContext,
    level: ErrorLevel = ErrorLevel.ERROR,
    user_message: Optional[str] = None,
    show_in_ui: bool = False,
    stop_execution: bool = False,
    return_value: Any = None
):
    """
    デコレーター形式のエラーハンドラー
    
    使用例:
    @error_handler(
        context=ErrorContext.SLACK_NOTIFICATION,
        return_value="エラー時のデフォルト戻り値"
    )
    def some_function():
        # 処理...
        pass
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            handler = UnifiedErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handler.handle_error(
                    error=e,
                    context=context,
                    level=level,
                    user_message=user_message,
                    show_in_ui=show_in_ui,
                    stop_execution=stop_execution,
                    return_value=return_value,
                    additional_info={
                        "function_name": func.__name__,
                        "args_count": len(args) if args else 0,
                        "kwargs_keys": list(kwargs.keys()) if kwargs else []
                    }
                )
        return wrapper
    return decorator

# コンテキストマネージャー版
class ErrorHandlerContext:
    """
    コンテキストマネージャー形式のエラーハンドラー
    
    使用例:
    with ErrorHandlerContext(
        context=ErrorContext.RAG_CHAIN,
        show_in_ui=True,
        additional_info={"query": "test"}
    ) as handler:
        # 処理...
        pass
    """
    
    def __init__(
        self,
        context: ErrorContext,
        level: ErrorLevel = ErrorLevel.ERROR,
        user_message: Optional[str] = None,
        show_in_ui: bool = False,
        stop_execution: bool = False,
        return_value: Any = None,
        additional_info: Optional[dict] = None  # ← 追加
    ):
        self.handler = UnifiedErrorHandler()
        self.context = context
        self.level = level
        self.user_message = user_message
        self.show_in_ui = show_in_ui
        self.stop_execution = stop_execution
        self.return_value = return_value
        self.additional_info = additional_info  # ← 追加
        self.exception_occurred = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception_occurred = True
            self.handler.handle_error(
                error=exc_val,
                context=self.context,
                level=self.level,
                user_message=self.user_message,
                show_in_ui=self.show_in_ui,
                stop_execution=self.stop_execution,
                return_value=self.return_value,
                additional_info=self.additional_info  # ← 追加
            )
            return True  # 例外を抑制
        return False

# 便利な関数群（既存コードとの互換性を保つため）
def handle_initialization_error(error: Exception, stop: bool = True):
    """初期化エラーの簡易ハンドラー"""
    handler = UnifiedErrorHandler()
    return handler.handle_error(
        error=error,
        context=ErrorContext.INITIALIZATION,
        level=ErrorLevel.CRITICAL,
        show_in_ui=True,
        stop_execution=stop
    )

def handle_slack_error(error: Exception, fallback_message: str):
    """Slackエラーの簡易ハンドラー"""
    handler = UnifiedErrorHandler()
    return handler.handle_error(
        error=error,
        context=ErrorContext.SLACK_NOTIFICATION,
        level=ErrorLevel.ERROR,
        return_value=fallback_message
    )

def handle_rag_error(error: Exception, query: str):
    """RAGエラーの簡易ハンドラー"""
    handler = UnifiedErrorHandler()
    return handler.handle_error(
        error=error,
        context=ErrorContext.RAG_CHAIN,
        level=ErrorLevel.ERROR,
        return_value="申し訳ございませんが、回答生成でエラーが発生しました。",
        additional_info={"query": query}
    )

def handle_agent_error(error: Exception, fallback_message: str = None):
    """AIエージェントエラーの簡易ハンドラー"""
    handler = UnifiedErrorHandler()
    return handler.handle_error(
        error=error,
        context=ErrorContext.AGENT_EXECUTION,
        level=ErrorLevel.ERROR,
        return_value=fallback_message or "申し訳ございませんが、AIエージェント処理でエラーが発生しました。通常モードで再試行してください。"
    )

def handle_data_processing_error(error: Exception, return_value: Any = None):
    """データ処理エラーの簡易ハンドラー"""
    handler = UnifiedErrorHandler()
    return handler.handle_error(
        error=error,
        context=ErrorContext.DATA_PROCESSING,
        level=ErrorLevel.ERROR,
        return_value=return_value
    )