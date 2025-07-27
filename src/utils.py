"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
（Faiss保存方法修正版）
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
# ChromaDBの代わりにFaissを使用
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from typing import List
from sudachipy import tokenizer, dictionary
from langchain_community.agent_toolkits import SlackToolkit
from langchain.agents import AgentType, initialize_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from docx import Document
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain import LLMChain
import datetime
import constants as ct
from pathlib import Path
from collections import Counter
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter as RegexTextSplitter
import pickle
import hashlib
import json

# ============================================================================
# 同義語辞書（必要に応じて拡張可能）
# ============================================================================

SYNONYM_DICT = {
    "受賞": ["受賞歴", "表彰", "アワード", "賞", "栄誉"],
    "実績": ["成果", "業績", "結果", "成績"],
    "会社": ["企業", "法人", "組織", "事業者"],
    "環境": ["エコ", "グリーン", "サステナブル", "持続可能"],
    "製品": ["商品", "サービス", "プロダクト"],
    "顧客": ["お客様", "利用者", "ユーザー", "クライアント"],
    "料金": ["価格", "費用", "コスト", "値段", "金額"],
    "配送": ["発送", "配達", "お届け", "輸送"],
    "注文": ["オーダー", "発注", "購入", "申込"]
}

############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# Faiss専用の保存・読み込み関数
############################################################

def save_faiss_index(db, base_path):
    """
    Faissインデックスを適切な方法で保存
    
    Args:
        db: FAISSベクトルストアオブジェクト
        base_path: 保存先のベースパス（拡張子なし）
    """
    try:
        # Faiss専用の保存方法を使用
        db.save_local(base_path)
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.info(f"✅ Faissインデックス保存完了: {base_path}")
        return True
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.error(f"❌ Faissインデックス保存エラー: {e}")
        return False

def load_faiss_index(base_path, embeddings):
    """
    Faissインデックスを適切な方法で読み込み
    
    Args:
        base_path: 読み込み元のベースパス（拡張子なし）
        embeddings: 埋め込みオブジェクト
        
    Returns:
        FAISSベクトルストアオブジェクト または None
    """
    try:
        # Faiss専用の読み込み方法を使用
        db = FAISS.load_local(base_path, embeddings, allow_dangerous_deserialization=True)
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.info(f"✅ Faissインデックス読み込み完了: {base_path}")
        return db
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.warning(f"⚠️ Faissインデックス読み込み失敗: {e}")
        return None

def calculate_docs_hash(docs):
    """
    ドキュメントリストのハッシュ値を計算（変更検知用）
    
    Args:
        docs: ドキュメントリスト
        
    Returns:
        ハッシュ値文字列
    """
    # ドキュメントの内容からハッシュを生成
    content_str = ""
    for doc in docs:
        content_str += doc.page_content + str(doc.metadata)
    
    return hashlib.md5(content_str.encode('utf-8')).hexdigest()

def should_rebuild_index(base_path, docs):
    """
    インデックスの再構築が必要かチェック
    
    Args:
        base_path: インデックスのベースパス
        docs: 現在のドキュメントリスト
        
    Returns:
        bool: 再構築が必要な場合True
    """
    metadata_file = f"{base_path}_metadata.json"
    
    # メタデータファイルが存在しない場合は再構築
    if not os.path.exists(metadata_file):
        return True
    
    # Faissインデックスファイルが存在しない場合は再構築
    if not os.path.exists(f"{base_path}.faiss"):
        return True
    
    try:
        # メタデータを読み込み
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # ドキュメントのハッシュを比較
        current_hash = calculate_docs_hash(docs)
        stored_hash = metadata.get('docs_hash', '')
        
        if current_hash != stored_hash:
            logger = logging.getLogger(ct.LOGGER_NAME)
            logger.info("📝 ドキュメントが変更されたため、インデックスを再構築します")
            return True
        
        return False
        
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.warning(f"⚠️ メタデータ読み込みエラー、再構築します: {e}")
        return True

def save_index_metadata(base_path, docs):
    """
    インデックスのメタデータを保存
    
    Args:
        base_path: インデックスのベースパス
        docs: ドキュメントリスト
    """
    try:
        metadata_file = f"{base_path}_metadata.json"
        metadata = {
            'docs_hash': calculate_docs_hash(docs),
            'doc_count': len(docs),
            'created_at': datetime.datetime.now().isoformat(),
            'version': '1.0'
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.info(f"✅ メタデータ保存完了: {metadata_file}")
        
    except Exception as e:
        logger = logging.getLogger(ct.LOGGER_NAME)
        logger.error(f"❌ メタデータ保存エラー: {e}")

############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])

def clean_keyword(keyword): 
    """
    キーワードから不正な文字を除去
    """
    import re
    # ゼロ幅文字やMarkdown記号を除去
    cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff●○■□▶◇◆\.]', '', keyword)
    # 前後の空白を除去
    cleaned = cleaned.strip()
    return cleaned

def expand_keywords_with_synonyms(keywords: List[str]) -> List[str]:
    """
    キーワードリストに同義語を追加して拡張
    
    Args:
        keywords: 元のキーワードリスト（例: ["受賞歴"]）
        
    Returns:
        同義語を含む拡張されたキーワードリスト（例: ["受賞歴", "受賞", "表彰", "アワード", "賞", "栄誉"]）
    """
    # 重複を避けるためsetを使用
    expanded = set(keywords)
    
    # 各キーワードについて同義語を探す
    for keyword in keywords:
        # 完全一致の同義語を追加
        for base_word, synonyms in SYNONYM_DICT.items():
            if keyword == base_word:
                # base_wordの同義語をすべて追加
                expanded.update(synonyms)
            elif keyword in synonyms:
                # keywordが同義語リストに含まれている場合
                expanded.add(base_word)
                expanded.update(synonyms)
        
        # 部分マッチの同義語を追加
        for base_word, synonyms in SYNONYM_DICT.items():
            if base_word in keyword or keyword in base_word:
                expanded.add(base_word)
                expanded.update(synonyms)
    
    return list(expanded)

def check_flexible_keyword_match(query_keywords: List[str], doc_keywords: List[str]) -> tuple:
    """
    柔軟なキーワードマッチングを実行
    
    Args:
        query_keywords: クエリから抽出されたキーワード
        doc_keywords: 文書から抽出されたキーワード
        
    Returns:
        (マッチしたかどうか, マッチしたキーワードのリスト, マッチタイプ)
    """
    matched_keywords = []  # マッチした組み合わせを記録
    match_types = []       # マッチの種類を記録
    
    # クエリキーワードを同義語で拡張
    expanded_query_keywords = expand_keywords_with_synonyms(query_keywords)
    
    # 拡張されたキーワードと文書キーワードを比較
    for query_kw in expanded_query_keywords:
        for doc_kw in doc_keywords:
            # 1. 完全一致チェック
            if query_kw == doc_kw:
                matched_keywords.append(f"{query_kw}={doc_kw}")
                match_types.append("完全一致")
                continue
            
            # 2. 部分一致チェック（クエリキーワードが文書キーワードに含まれる）
            if query_kw in doc_kw:
                matched_keywords.append(f"{query_kw}⊆{doc_kw}")
                match_types.append("部分一致")
                continue
            
            # 3. 逆部分一致チェック（文書キーワードがクエリキーワードに含まれる）
            if doc_kw in query_kw:
                matched_keywords.append(f"{doc_kw}⊆{query_kw}")
                match_types.append("逆部分一致")
                continue
    
    # マッチしたかどうかを判定
    is_match = len(matched_keywords) > 0
    return is_match, matched_keywords, match_types

def filter_chunks_by_flexible_keywords(docs, query):
    """
    柔軟なキーワードマッチングを使ってチャンクをフィルタリング
    
    Args:
        docs: 検索で得られたチャンクリスト（Documentオブジェクト）
        query: ユーザー入力
        
    Returns:
        フィルター後のチャンクリスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # ステップ1: クエリから名詞を抽出
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer.Tokenizer.SplitMode.C
        tokens = tokenizer_obj.tokenize(query, mode)
        query_nouns = [
            t.surface() 
            for t in tokens
            if "名詞" in t.part_of_speech() and len(t.surface()) > 1
        ]
        
        logger.info(f"📝 抽出されたクエリ名詞: {query_nouns}")
        
        # ステップ2: 各文書とマッチングを実行
        filtered_docs = []
        for doc in docs:
            # 文書のキーワードを取得
            top_keywords_str = doc.metadata.get("top_keywords", "")
            if top_keywords_str:
                top_keywords = [kw.strip() for kw in top_keywords_str.split(" / ") if kw.strip()]
                
                # 柔軟なキーワードマッチングを実行
                is_match, matched_keywords, match_types = check_flexible_keyword_match(
                    query_nouns, top_keywords
                )
                
                # マッチした場合は結果リストに追加
                if is_match:
                    filtered_docs.append(doc)
                    logger.info(f"✅ マッチ成功:")
                    logger.info(f"   ファイル: {doc.metadata.get('file_name', '不明')}")
                    logger.info(f"   マッチしたキーワード: {matched_keywords}")
                    logger.info(f"   マッチタイプ: {set(match_types)}")
        
        logger.info(f"📊 フィルター結果: {len(docs)} → {len(filtered_docs)} 件")
        
        # ステップ3: フォールバック処理
        if not filtered_docs:
            logger.info("⚠️ フィルター結果が空のため、元のdocsを返します")
            return docs  # 安全策：フィルター結果が空の場合は元のリストを返す
        
        return filtered_docs
        
    except Exception as e:
        logger.error(f"❌ フィルタリング処理でエラーが発生: {e}")
        return docs  # エラー時も安全策として元のリストを返す

def create_rag_chain(db_name):
    """
    引数として渡されたDB内を参照するRAGのChainを作成（Faiss修正版）

    Args:
        db_name: RAG化対象のデータを格納するデータベース名
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    # AIエージェント機能を使わない場合の処理
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        # 「data」フォルダ直下の各フォルダ名に対して処理
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            # フォルダ内の各ファイルのデータをリストに追加
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    # AIエージェント機能を使う場合の処理
    else:
        # データベース名に対応した、RAG化対象のデータ群が格納されているフォルダパスを取得
        folder_path = ct.DB_NAMES[db_name]
        # フォルダ内の各ファイルのデータをリストに追加
        add_docs(folder_path, docs_all)

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)
    
    # ✅ チャンク先頭にメタ情報を付加（retrieverと同じ処理）
    for doc in splitted_docs:
        file_name = doc.metadata.get("file_name", "不明")
        category = doc.metadata.get("category", "不明")
        heading = doc.metadata.get("first_heading", "")
        keywords_str = doc.metadata.get("top_keywords", "") 

        prefix = f"【カテゴリ: {category}】【ファイル名: {file_name}】"
        if heading:
            prefix += f"【見出し: {heading}】"
        if keywords_str:  # ← 修正：文字列をそのまま使用
            prefix += f"【キーワード: {keywords_str}】"

        doc.page_content = prefix + "\n" + doc.page_content

    embeddings = OpenAIEmbeddings()

    # Faissインデックスの作成（修正版）
    base_path = f"{db_name}_faiss"
    
    # 再構築が必要かチェック
    if should_rebuild_index(base_path, splitted_docs):
        logger.info(f"🔨 Faissインデックスを構築中: {base_path}")
        db = FAISS.from_documents(splitted_docs, embeddings)
        
        # Faiss専用の保存方法を使用
        if save_faiss_index(db, base_path):
            save_index_metadata(base_path, splitted_docs)
        
    else:
        # 既存のインデックスを読み込み
        db = load_faiss_index(base_path, embeddings)
        if db is None:
            # 読み込み失敗時は再作成
            logger.info("🔄 インデックス読み込み失敗、再作成します")
            db = FAISS.from_documents(splitted_docs, embeddings)
            if save_faiss_index(db, base_path):
                save_index_metadata(base_path, splitted_docs)
    
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def create_retriever(db_name):
    """
    指定されたDBパスに基づいてRetrieverのみを作成（Faiss修正版）

    Args:
        db_name: ベクトルDBの保存先ディレクトリ名（または定義名）

    Returns:
        LangChainのRetrieverオブジェクト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    else:
        folder_path = ct.DB_NAMES[db_name]
        add_docs(folder_path, docs_all)

    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)
    embeddings = OpenAIEmbeddings()

    # チャンク先頭にメタ情報を付加
    for doc in splitted_docs:
        file_name = doc.metadata.get("file_name", "不明")
        category = doc.metadata.get("category", "不明")
        heading = doc.metadata.get("first_heading", "")
        keywords_str = doc.metadata.get("top_keywords", "")

        # メタ情報を1行目に構造化（お好みで調整可能）
        prefix = f"【カテゴリ: {category}】【ファイル名: {file_name}】"
        if heading:
            prefix += f"【見出し: {heading}】"
        if keywords_str:  # ← 修正：文字列をそのまま使用
            prefix += f"【キーワード: {keywords_str}】"

        doc.page_content = prefix + "\n" + doc.page_content

    # Faissインデックスの作成（修正版）
    base_path = f"{db_name}_faiss"
    
    if should_rebuild_index(base_path, splitted_docs):
        db = FAISS.from_documents(splitted_docs, embeddings)
        if save_faiss_index(db, base_path):
            save_index_metadata(base_path, splitted_docs)
    else:
        db = load_faiss_index(base_path, embeddings)
        if db is None:
            db = FAISS.from_documents(splitted_docs, embeddings)
            if save_faiss_index(db, base_path):
                save_index_metadata(base_path, splitted_docs)

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    return retriever

# 以下、既存の関数群をそのまま維持...

def add_docs(folder_path, docs_all):
    """
    フォルダ内のファイル一覧を取得

    Args:
        folder_path: フォルダのパス
        docs_all: 各ファイルデータを格納するリスト
    """
    print(f"📂 読み込もうとしているフォルダ: {folder_path}")
    print(f"📂 フルパス: {os.path.abspath(folder_path)}")

    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        # ファイルの拡張子を取得
        file_extension = os.path.splitext(file)[1]
        # 想定していたファイル形式の場合のみ読み込む
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
        else:
            continue

        docs = loader.load()
        p = Path(file_path)

        for doc in docs:
            content = doc.page_content

            # 📌 基本メタ情報
            doc.metadata["file_name"] = p.name
            doc.metadata["file_stem"] = p.stem
            doc.metadata["file_ext"] = p.suffix
            doc.metadata["file_path"] = str(p)
            doc.metadata["category"] = p.parent.name
            doc.metadata["file_mtime"] = datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            doc.metadata["file_ctime"] = datetime.datetime.fromtimestamp(p.stat().st_ctime).isoformat()

            # 🧑‍💻 作成者（docxの場合のみ取得可能な場合あり）
            if p.suffix == ".docx":
                try:
                    from docx import Document as DocxDocument
                    core_props = DocxDocument(p).core_properties
                    doc.metadata["file_author"] = core_props.author
                except Exception:
                    doc.metadata["file_author"] = "不明"
            else:
                doc.metadata["file_author"] = "不明"

            # 🧩 セクション見出し
            section_titles = []
            for line in content.splitlines():
                if line.strip().startswith(("■", "●", "○", "【", "▶", "◇", "◆")):
                    section_titles.append(line.strip())
            doc.metadata["section_titles"] = " / ".join(section_titles)
            doc.metadata["first_heading"] = section_titles[0] if section_titles else ""
            doc.metadata["section_count"] = len(section_titles)

            # 🧠 頻出キーワード（名詞のみ）- 修正版
            try:
                # 形態素解析の実行
                tokenizer_obj = dictionary.Dictionary().create()
                mode = tokenizer.Tokenizer.SplitMode.C
                tokens = tokenizer_obj.tokenize(content, mode)
                
                # 名詞のみを抽出（クリーニング付き）
                nouns = []
                raw_nouns_sample = []  # デバッグ用サンプル
                
                for i, t in enumerate(tokens):
                    surface = t.surface()
                    if "名詞" in t.part_of_speech() and len(surface) > 1:
                        # デバッグ用サンプル収集（最初の10個まで）
                        if len(raw_nouns_sample) < 10:
                            raw_nouns_sample.append(surface)
                        
                        # キーワードをクリーニング
                        cleaned_surface = clean_keyword(surface)
                        if cleaned_surface and len(cleaned_surface) > 1:  # 空文字や1文字を除外
                            nouns.append(cleaned_surface)
                
                # デバッグ用出力
                if raw_nouns_sample:
                    print(f"📝 クリーニング前サンプル ({p.name}): {raw_nouns_sample}")
                    print(f"📝 クリーニング後サンプル ({p.name}): {nouns[:10]}")
                
                # 頻出キーワードを取得
                if nouns:
                    word_counts = Counter(nouns)
                    top_keywords = [word for word, count in word_counts.most_common(5)]
                    print(f"🔑 抽出キーワード ({p.name}): {top_keywords}")
                else:
                    top_keywords = []
                    print(f"⚠️ 名詞が抽出されませんでした: {p.name}")
                
                # メタデータに設定
                doc.metadata["top_keywords"] = " / ".join(top_keywords)
                
            except Exception as e:
                print(f"❌ キーワード抽出エラー ({p.name}): {e}")
                import traceback
                print(f"詳細エラー: {traceback.format_exc()}")
                doc.metadata["top_keywords"] = ""

            # ✏️ 文字数・行数など（追加で役立つ）
            doc.metadata["num_chars"] = len(content)
            doc.metadata["num_lines"] = len(content.splitlines())

        docs_all.extend(docs)

# 既存の関数群を維持（run_company_doc_chain, run_service_doc_chain など）

def run_company_doc_chain(param):
    """
    会社に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # 会社に関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_service_doc_chain(param):
    """
    サービスに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # サービスに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_customer_doc_chain(param):
    """
    顧客とのやり取りに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値
    
    Returns:
        LLMからの回答
    """
    # 顧客とのやり取りに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_manual_doc_chain(param):
    """
    操作マニュアルや手順書に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # RAG chainを使って回答を生成
    ai_msg = st.session_state.manual_doc_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })

    # 会話履歴に記録（オプション）
    st.session_state.chat_history.extend([
        HumanMessage(content=param),
        AIMessage(content=ai_msg["answer"])
    ])

    return ai_msg["answer"]

def run_policy_doc_chain(param):
    """
    利用規約・キャンセル・返品などの制度・ポリシーに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # RAG chainを使って回答を生成
    ai_msg = st.session_state.policy_doc_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })

    # 会話履歴に記録（オプション）
    st.session_state.chat_history.extend([
        HumanMessage(content=param),
        AIMessage(content=ai_msg["answer"])
    ])

    return ai_msg["answer"]


def run_sustainability_doc_chain(param):
    """
    環境・サステナビリティ・エシカル活動に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # RAG chainを使って回答を生成
    ai_msg = st.session_state.sustainability_doc_chain.invoke({
        "input": param,
        "chat_history": st.session_state.chat_history
    })

    # 会話履歴に記録（オプション）
    st.session_state.chat_history.extend([
        HumanMessage(content=param),
        AIMessage(content=ai_msg["answer"])
    ])

    return ai_msg["answer"]

def delete_old_conversation_log(result):
    """
    古い会話履歴の削除

    Args:
        result: LLMからの回答
    """
    # LLMからの回答テキストのトークン数を取得
    response_tokens = len(st.session_state.enc.encode(result))
    # 過去の会話履歴の合計トークン数に加算
    st.session_state.total_tokens += response_tokens

    # トークン数が上限値を下回るまで、順に古い会話履歴を削除
    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        # 最も古い会話履歴を削除
        removed_message = st.session_state.chat_history.pop(1)
        # 最も古い会話履歴のトークン数を取得
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        # 過去の会話履歴の合計トークン数から、最も古い会話履歴のトークン数を引く
        st.session_state.total_tokens -= removed_tokens

def notice_slack(chat_message):
    """
    問い合わせ内容のSlackへの通知

    Args:
        chat_message: ユーザーメッセージ

    Returns:
        問い合わせサンクスメッセージ
    """

    # Slack通知用のAgent Executorを作成
    toolkit = SlackToolkit()
    tools = toolkit.get_tools()
    agent_executor = initialize_agent(
        llm=st.session_state.llm,
        tools=tools,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
    )

    # 担当者割り振りに使う用の「従業員情報」と「問い合わせ対応履歴」の読み込み
    loader = CSVLoader(ct.EMPLOYEE_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs = loader.load()
    loader = CSVLoader(ct.INQUIRY_HISTORY_FILE_PATH, encoding=ct.CSV_ENCODING)
    docs_history = loader.load()

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    for doc in docs_history:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])

    # 問い合わせ内容と関連性が高い従業員情報を取得するために、参照先データを整形
    docs_all = adjust_reference_data(docs, docs_history)
    
    # 形態素解析による日本語の単語分割を行うため、参照先データからテキストのみを抽出
    docs_all_page_contents = []
    for doc in docs_all:
        docs_all_page_contents.append(doc.page_content)

    # Retrieverの作成（Faiss版）
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs_all, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})
    bm25_retriever = BM25Retriever.from_texts(
        docs_all_page_contents,
        preprocess_func=preprocess_func,
        k=ct.TOP_K
    )
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=ct.RETRIEVER_WEIGHTS
    )

    # 問い合わせ内容と関連性の高い従業員情報を取得
    employees = retriever.invoke(chat_message)
    
    # プロンプトに埋め込むための従業員情報テキストを取得
    context = get_context(employees)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", ct.SYSTEM_PROMPT_EMPLOYEE_SELECTION)
    ])
    # フォーマット文字列を生成
    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()

    # 問い合わせ内容と関連性が高い従業員のID一覧を取得
    messages = prompt_template.format_prompt(context=context, query=chat_message, format_instruction=format_instruction).to_messages()
    employee_id_response = st.session_state.llm(messages)
    employee_ids = output_parser.parse(employee_id_response.content)

    # 問い合わせ内容と関連性が高い従業員情報を、IDで照合して取得
    target_employees = get_target_employees(employees, employee_ids)
    
    # 問い合わせ内容と関連性が高い従業員情報の中から、SlackIDのみを抽出
    slack_ids = get_slack_ids(target_employees)
    
    # 抽出したSlackIDの連結テキストを生成
    slack_id_text = create_slack_id_text(slack_ids)
    
    # プロンプトに埋め込むための（問い合わせ内容と関連性が高い）従業員情報テキストを取得
    context = get_context(target_employees)

    # 現在日時を取得
    now_datetime = get_datetime()

    # Slack通知用のプロンプト生成
    prompt = PromptTemplate(
        input_variables=["slack_id_text", "query", "context", "now_datetime"],
        template=ct.SYSTEM_PROMPT_NOTICE_SLACK,
    )
    prompt_message = prompt.format(slack_id_text=slack_id_text, query=chat_message, context=context, now_datetime=now_datetime)

    # Slack通知の実行
    agent_executor.invoke({"input": prompt_message})

    return ct.CONTACT_THANKS_MESSAGE

def adjust_reference_data(docs, docs_history):
    """
    Slack通知用の参照先データの整形

    Args:
        docs: 従業員情報ファイルの読み込みデータ
        docs_history: 問い合わせ対応履歴ファイルの読み込みデータ

    Returns:
        従業員情報と問い合わせ対応履歴の結合テキスト
    """

    docs_all = []
    for row in docs:
        # 従業員IDの取得
        row_lines = row.page_content.split("\n")
        row_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_lines}
        employee_id = row_dict["従業員ID"]

        doc = ""

        # 取得した従業員IDに紐づく問い合わせ対応履歴を取得
        same_employee_inquiries = []
        for row_history in docs_history:
            row_history_lines = row_history.page_content.split("\n")
            row_history_dict = {item.split(": ")[0]: item.split(": ")[1] for item in row_history_lines}
            if row_history_dict["従業員ID"] == employee_id:
                same_employee_inquiries.append(row_history_dict)

        if same_employee_inquiries:
            doc += "【従業員情報】\n"
            row_data = "\n".join(row_lines)
            doc += row_data + "\n=================================\n"
            doc += "【この従業員の問い合わせ対応履歴】\n"
            for inquiry_dict in same_employee_inquiries:
                for key, value in inquiry_dict.items():
                    doc += f"{key}: {value}\n"
                doc += "---------------\n"

            new_doc = Document(page_content=doc, metadata={})
        else:
            new_doc = Document(page_content=row.page_content, metadata={})

        docs_all.append(new_doc)
    
    return docs_all

def get_target_employees(employees, employee_ids):
    """
    問い合わせ内容と関連性が高い従業員情報一覧の取得

    Args:
        employees: 問い合わせ内容と関連性が高い従業員情報一覧
        employee_ids: 問い合わせ内容と関連性が「特に」高い従業員のID一覧

    Returns:
        問い合わせ内容と関連性が「特に」高い従業員情報一覧
    """

    target_employees = []
    duplicate_check = []
    target_text = "従業員ID"
    for employee in employees:
        # 従業員IDの取得
        num = employee.page_content.find(target_text)
        employee_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        # 問い合わせ内容と関連性が高い従業員情報を、IDで照合して取得（重複除去）
        if employee_id in employee_ids:
            if employee_id in duplicate_check:
                continue
            duplicate_check.append(employee_id)
            target_employees.append(employee)
    
    return target_employees

def get_slack_ids(target_employees):
    """
    SlackIDの一覧を取得

    Args:
        target_employees: 問い合わせ内容と関連性が高い従業員情報一覧

    Returns:
        SlackIDの一覧
    """

    target_text = "SlackID"
    slack_ids = []
    for employee in target_employees:
        num = employee.page_content.find(target_text)
        slack_id = employee.page_content[num+len(target_text)+2:].split("\n")[0]
        slack_ids.append(slack_id)
    
    return slack_ids

def create_slack_id_text(slack_ids):
    """
    SlackIDの一覧を取得

    Args:
        slack_ids: SlackIDの一覧

    Returns:
        SlackIDを「と」で繋いだテキスト
    """
    slack_id_text = ""
    for i, id in enumerate(slack_ids):
        slack_id_text += f"「{id}」"
        # 最後のSlackID以外、連結後に「と」を追加
        if not i == len(slack_ids)-1:
            slack_id_text += "と"
    
    return slack_id_text

def get_context(docs):
    """
    プロンプトに埋め込むための従業員情報テキストの生成
    Args:
        docs: 従業員情報の一覧

    Returns:
        生成した従業員情報テキスト
    """

    context = ""
    for i, doc in enumerate(docs, start=1):
        context += "===========================================================\n"
        context += f"{i}人目の従業員情報\n"
        context += "===========================================================\n"
        context += doc.page_content + "\n\n"

    return context

def get_datetime():
    """
    現在日時を取得

    Returns:
        現在日時
    """

    dt_now = datetime.datetime.now()
    now_datetime = dt_now.strftime('%Y年%m月%d日 %H:%M:%S')

    return now_datetime

def preprocess_func(text):
    """
    形態素解析による日本語の単語分割
    Args:
        text: 単語分割対象のテキスト

    Returns:
        単語分割を実施後のテキスト
    """

    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))

    return words

def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s

def filter_chunks_by_top_keywords(docs, query):
    """
    top_keywords を使ってチャンクをフィルタリング

    Args:
        docs: 検索で得られたチャンクリスト（Documentオブジェクト）
        query: ユーザー入力

    Returns:
        フィルター後のチャンクリスト（条件に一致しない場合は元のdocsを返す）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    try:
        # クエリから名詞を抽出
        tokenizer_obj = dictionary.Dictionary().create()
        mode = tokenizer.Tokenizer.SplitMode.C
        tokens = tokenizer_obj.tokenize(query, mode)
        query_nouns = set([
            t.surface() 
            for t in tokens
            if "名詞" in t.part_of_speech() and len(t.surface()) > 1
        ])
        
        # デバッグログ出力
        logger.info(f"フィルター対象クエリ名詞: {query_nouns}")
        logger.info(f"フィルター前チャンク件数: {len(docs)}")
        
        # フィルタリング実行
        filtered_docs = []
        for doc in docs:
            top_keywords_str = doc.metadata.get("top_keywords", "")
            if top_keywords_str:
                # top_keywordsが " / " で区切られている場合の処理
                top_keywords = [kw.strip() for kw in top_keywords_str.split(" / ") if kw.strip()]
                
                # クエリの名詞とキーワードの一致チェック
                if any(kw in query_nouns for kw in top_keywords if kw):
                    filtered_docs.append(doc)
                    logger.info(f"マッチしたキーワード: {[kw for kw in top_keywords if kw in query_nouns]}")
        
        logger.info(f"フィルター後チャンク件数: {len(filtered_docs)}")
        
        # フィルター結果が空の場合は元のdocsを返す（fallback）
        if not filtered_docs:
            logger.info("フィルター結果が空のため、元のdocsを返します")
            return docs
        
        return filtered_docs
        
    except Exception as e:
        logger.error(f"フィルタリング処理でエラーが発生: {e}")
        return docs  # エラー時は元のdocsを返す

def execute_agent_or_chain(chat_message):
    """
    AIエージェントもしくはAIエージェントなしのRAGのChainを実行

    Args:
        chat_message: ユーザーメッセージ

    Returns:
        LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # === 追加: 実行モードの明確な記録 ===
    logger.info(f"🎯 実行モード: {st.session_state.agent_mode}")
    logger.info(f"📝 入力メッセージ: {chat_message}")

    # AIエージェント機能を利用する場合
    if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
        logger.info("🤖 AIエージェントモードで実行")
        st_callback = StreamlitCallbackHandler(st.container())
        result = st.session_state.agent_executor.invoke({"input": chat_message}, {"callbacks": [st_callback]})
        response = result["output"]
    else:
        logger.info("🔍 通常RAGモードで実行 - 柔軟キーワードマッチング適用")
        
        try:
            # 1. 通常のRetrieverで関連文書を取得
            retriever = create_retriever(ct.DB_ALL_PATH)
            original_docs = retriever.get_relevant_documents(chat_message)
            logger.info(f"📚 通常検索結果: {len(original_docs)}件")

            # 2. 柔軟なキーワードマッチングを適用
            logger.info("🧠 柔軟なキーワードマッチングを開始")
            filtered_docs = filter_chunks_by_flexible_keywords(original_docs, chat_message)
            logger.info(f"✅ フィルター後: {len(filtered_docs)}件")

            # 3. フィルター後の文書を使って手動でRAG処理を実行
            if filtered_docs:
                logger.info("📖 フィルター後文書でRAG実行")
                # フィルター後の文書からcontextを構築
                context = "\n\n".join([doc.page_content for doc in filtered_docs[:ct.TOP_K]])
                
                # プロンプトを手動で構築してLLMに送信
                question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
                messages = [
                    {"role": "system", "content": question_answer_template.format(context=context)},
                    {"role": "user", "content": chat_message}
                ]
                
                # LLMに送信
                response_obj = st.session_state.llm.invoke(messages)
                response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                logger.info("✅ カスタムRAG処理完了")
            else:
                logger.info("⚠️ フィルター結果が空 - 通常RAGチェーンにフォールバック")
                # フィルター結果が空の場合は通常のRAGを実行
                result = st.session_state.rag_chain.invoke({
                    "input": chat_message,
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]

            # 会話履歴への追加
            st.session_state.chat_history.extend([
                HumanMessage(content=chat_message),
                AIMessage(content=response)
            ])
            
        except Exception as e:
            logger.error(f"❌ RAG処理でエラー発生: {e}")
            import traceback
            logger.error(f"詳細エラー: {traceback.format_exc()}")
            
            # エラー時のフォールバック
            try:
                result = st.session_state.rag_chain.invoke({
                    "input": chat_message,
                    "chat_history": st.session_state.chat_history
                })
                response = result["answer"]
                logger.info("🔄 フォールバック処理完了")
            except Exception as e2:
                logger.error(f"❌ フォールバック処理もエラー: {e2}")
                response = "申し訳ございませんが、現在システムに問題が発生しています。"

    if response != ct.NO_DOC_MATCH_MESSAGE:
        st.session_state.answer_flg = True

    logger.info(f"📤 最終回答: {response[:100]}...")
    return response

def test_keyword_filter():
    """
    キーワードフィルターのテスト関数
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    test_queries = [
        "SNS投稿に関する特典はありますか？",
        "海外配送は対応していますか？", 
        "地域貢献活動はありますか？",
        "受賞歴を教えてください",
        "株主優待制度について教えて",
        "環境への取り組みを知りたい",
        "サブスクリプションプランの料金は？"
    ]
    
    retriever = create_retriever(ct.DB_ALL_PATH)
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"🧪 テストクエリ: {query}")
        print('='*100)
        
        # 通常検索
        original_docs = retriever.get_relevant_documents(query)
        
        # フィルター適用
        filtered_docs = filter_chunks_by_top_keywords(original_docs, query)
        
        print(f"📊 結果:")
        print(f"   - 通常検索: {len(original_docs)}件")
        print(f"   - フィルター後: {len(filtered_docs)}件")
        print(f"   - フィルター効果: {(1-len(filtered_docs)/len(original_docs))*100:.1f}%削減")

def test_flexible_keyword_filter():
    """
    柔軟なキーワードフィルターのテスト関数
    """
    logger = logging.getLogger(ct.LOGGER_NAME)
    
    # 問題のあったクエリを含むテストケース
    test_queries = [
        "受賞歴を教えてください",  # メインの問題クエリ
        "SNS投稿に関する特典はありますか？",
        "海外配送は対応していますか？", 
        "地域貢献活動はありますか？",
        "株主優待制度について教えて",
        "環境への取り組みを知りたい",
        "会社の実績を教えて",
        "アワードを受賞していますか？"
    ]
    
    retriever = create_retriever(ct.DB_ALL_PATH)
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"🧪 テストクエリ: {query}")
        print('='*100)
        
        # 通常検索
        original_docs = retriever.get_relevant_documents(query)
        
        # 従来のフィルター
        old_filtered_docs = filter_chunks_by_top_keywords(original_docs, query)
        
        # 新しい柔軟なフィルター
        new_filtered_docs = filter_chunks_by_flexible_keywords(original_docs, query)
        
        print(f"📊 結果比較:")
        print(f"   - 通常検索: {len(original_docs)}件")
        print(f"   - 従来フィルター: {len(old_filtered_docs)}件")
        print(f"   - 柔軟フィルター: {len(new_filtered_docs)}件")
        print(f"   - 改善効果: {len(new_filtered_docs) - len(old_filtered_docs):+d}件")