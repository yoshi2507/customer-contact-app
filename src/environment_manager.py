"""
統一環境変数管理モジュール

このモジュールは、アプリケーション全体の環境変数取得を統一管理し、
セキュリティとエラーハンドリングの一元化を提供します。

主な機能:
- 安全な環境変数取得
- 機密情報の自動マスキング
- キャッシュ機能によるパフォーマンス向上
- 必須環境変数の検証
"""

import os
import time
import logging
from typing import Optional, Dict, List, Any
from enum import Enum
import constants as ct

class EnvironmentError(Exception):
    """環境変数関連のカスタム例外"""
    pass

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

class EnvironmentManager:
    """統一環境変数管理クラス（シングルトン）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(ct.LOGGER_NAME)
        self._secrets_cache = {}
        self._timed_cache = {}  # (value, timestamp) のタプル
        self._access_count = {}  # アクセス頻度追跡
        self._initialized = True
        
        self.logger.info("🔧 EnvironmentManager初期化完了")
    
    def get_secret(
        self, 
        key: str, 
        required: bool = True,
        default: Optional[str] = None,
        mask_in_logs: bool = True,
        use_cache: bool = True
    ) -> Optional[str]:
        """
        安全な環境変数取得
        
        Args:
            key: 環境変数名
            required: 必須フラグ
            default: デフォルト値
            mask_in_logs: ログでマスクするか
            use_cache: キャッシュを使用するか
        
        Returns:
            環境変数の値またはNone
        
        Raises:
            EnvironmentError: 必須環境変数が見つからない場合
        """
        
        # アクセス回数をカウント
        self._access_count[key] = self._access_count.get(key, 0) + 1
        
        # キャッシュから取得を試行
        if use_cache and key in self._secrets_cache:
            self.logger.debug(f"環境変数 '{key}' をキャッシュから取得")
            return self._secrets_cache[key]
        
        # 環境変数から取得
        value = os.environ.get(key, default)
        
        # 必須チェック
        if required and not value:
            error_msg = f"必須環境変数 '{key}' が設定されていません"
            self.logger.critical(error_msg)
            raise EnvironmentError(error_msg)
        
        # ログ出力（マスク処理）
        if value:
            if mask_in_logs:
                masked_value = self._mask_secret(value)
                self.logger.info(f"環境変数 '{key}' を取得: {masked_value}")
            else:
                self.logger.info(f"環境変数 '{key}' を取得: 設定済み")
        else:
            self.logger.info(f"環境変数 '{key}': 未設定")
        
        # キャッシュに保存
        if use_cache:
            self._secrets_cache[key] = value
            self._timed_cache[key] = (value, time.time())
        
        return value
    
    def get_secret_with_fallback(
        self, 
        key: str, 
        fallback_keys: List[str] = None,
        required: bool = True
    ) -> Optional[str]:
        """
        フォールバック付き環境変数取得
        
        Args:
            key: メイン環境変数名
            fallback_keys: フォールバック変数名のリスト
            required: 必須フラグ
            
        Returns:
            取得できた環境変数の値
            
        Raises:
            EnvironmentError: 全ての環境変数が見つからない場合
        """
        try:
            return self.get_secret(key, required=True)
        except EnvironmentError as e:
            if fallback_keys:
                for fallback_key in fallback_keys:
                    try:
                        self.logger.info(f"フォールバック: {key} -> {fallback_key}")
                        return self.get_secret(fallback_key, required=True)
                    except EnvironmentError:
                        continue
            
            if required:
                all_keys = [key] + (fallback_keys or [])
                error_msg = f"環境変数が見つかりません: {all_keys}"
                self.logger.critical(error_msg)
                raise EnvironmentError(error_msg)
            
            return None
    
    def validate_required_secrets(self, required_keys: List[str] = None) -> Dict[str, bool]:
        """
        必須環境変数の一括検証
        
        Args:
            required_keys: 検証対象のキーリスト（Noneの場合はデフォルト）
            
        Returns:
            {環境変数名: 存在フラグ} の辞書
        """
        if required_keys is None:
            required_keys = getattr(ct, 'REQUIRED_SECRETS', [
                'OPENAI_API_KEY',
                'SLACK_BOT_TOKEN'
            ])
        
        validation_result = {}
        
        for key in required_keys:
            try:
                value = self.get_secret(key, required=False, mask_in_logs=True)
                validation_result[key] = bool(value)
            except Exception as e:
                self.logger.error(f"環境変数 '{key}' の検証でエラー: {e}")
                validation_result[key] = False
        
        # 検証結果をログ出力
        missing_keys = [k for k, v in validation_result.items() if not v]
        if missing_keys:
            self.logger.warning(f"❌ 不足している必須環境変数: {missing_keys}")
        else:
            self.logger.info("✅ 全ての必須環境変数が設定済み")
        
        return validation_result
    
    def get_environment_status(self, include_optional: bool = True) -> Dict[str, str]:
        """
        環境変数の状態取得（マスク済み）
        
        Args:
            include_optional: オプション環境変数も含めるか
            
        Returns:
            {環境変数名: ステータス} の辞書
        """
        status = {}
        
        # 必須環境変数
        required_keys = getattr(ct, 'REQUIRED_SECRETS', [
            'OPENAI_API_KEY',
            'SLACK_BOT_TOKEN'
        ])
        
        for key in required_keys:
            value = os.environ.get(key)
            if value:
                status[key] = f"設定済み ({self._mask_secret(value)})"
            else:
                status[key] = "❌ 未設定"
        
        # オプション環境変数
        if include_optional:
            optional_keys = getattr(ct, 'OPTIONAL_SECRETS', [
                'SLACK_USER_TOKEN',
                'SERP_API_KEY',
                'GOOGLE_APPLICATION_CREDENTIALS'
            ])
            
            for key in optional_keys:
                value = os.environ.get(key)
                if value:
                    status[key] = f"設定済み ({self._mask_secret(value)})"
                else:
                    status[key] = "未設定（オプション）"
        
        return status
    
    def _mask_secret(self, value: str, show_chars: int = 4) -> str:
        """
        機密情報のマスク処理
        
        Args:
            value: マスクする文字列
            show_chars: 両端に表示する文字数
            
        Returns:
            マスクされた文字列
        """
        if not value:
            return ""
        
        if len(value) <= show_chars * 2:
            return "*" * len(value)
        
        return f"{value[:show_chars]}{'*' * (len(value) - show_chars * 2)}{value[-show_chars:]}"
    
    def clear_cache(self):
        """キャッシュのクリア（セキュリティ対応）"""
        cache_size = len(self._secrets_cache)
        self._secrets_cache.clear()
        self._timed_cache.clear()
        self.logger.info(f"🧹 環境変数キャッシュをクリア ({cache_size}件)")
    
    def cleanup_old_cache(self, max_age_minutes: int = 30):
        """
        古いキャッシュエントリの削除
        
        Args:
            max_age_minutes: キャッシュの最大保持時間（分）
        """
        current_time = time.time()
        expired_keys = [
            key for key, (value, timestamp) in self._timed_cache.items()
            if (current_time - timestamp) > (max_age_minutes * 60)
        ]
        
        for key in expired_keys:
            if key in self._secrets_cache:
                del self._secrets_cache[key]
            if key in self._timed_cache:
                del self._timed_cache[key]
        
        if expired_keys:
            self.logger.info(f"🧹 期限切れキャッシュを削除: {expired_keys}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        キャッシュ統計情報の取得
        
        Returns:
            キャッシュ統計の辞書
        """
        return {
            "cache_size": len(self._secrets_cache),
            "access_count": dict(self._access_count),
            "cached_keys": list(self._secrets_cache.keys()),
            "most_accessed": max(self._access_count.items(), key=lambda x: x[1]) if self._access_count else None
        }

# グローバル関数（既存コードとの互換性のため）
_env_manager_instance = None

def get_environment_manager() -> EnvironmentManager:
    """環境変数マネージャーの取得（シングルトン）"""
    global _env_manager_instance
    if _env_manager_instance is None:
        _env_manager_instance = EnvironmentManager()
    return _env_manager_instance

def safe_get_secret(
    key: str, 
    required: bool = True, 
    default: Optional[str] = None,
    mask_in_logs: bool = True
) -> Optional[str]:
    """
    安全な環境変数取得（ショートカット関数）
    
    Args:
        key: 環境変数名
        required: 必須フラグ
        default: デフォルト値
        mask_in_logs: ログでマスクするか
        
    Returns:
        環境変数の値またはNone
    """
    manager = get_environment_manager()
    return manager.get_secret(key, required, default, mask_in_logs)

def check_environment_health() -> bool:
    """
    環境変数の健全性チェック
    
    Returns:
        全ての必須環境変数が設定されている場合True
    """
    manager = get_environment_manager()
    validation_result = manager.validate_required_secrets()
    return all(validation_result.values())

def check_env_var_status(key: str) -> str:
    """
    環境変数の状態をチェック（既存コードとの互換性）
    
    Args:
        key: 環境変数名
        
    Returns:
        状態を示す文字列
    """
    manager = get_environment_manager()
    value = os.environ.get(key)
    if value:
        return f"設定済み ({manager._mask_secret(value)})"
    else:
        return "未設定"