# 問い合わせ対応AIエージェント（LangChain × RAG）

本アプリは、LangChainベースのAIエージェントを活用し、社内ドキュメントに基づいた問い合わせ対応を自動化するシステムです。  
複数のドキュメント群（サービス案内、マニュアル、規約、環境方針など）をToolとして切り分け、ユーザーの質問に応じて最適な情報源を選定し、適切に回答します。

---

## 🚀 特徴

- LangChainの**ReAct型エージェント**をベースにしたTool選択機能
- ToolごとにベクトルDBを持ち、精度の高いRAG回答を実現
- ユーザーの曖昧な質問にも対応し、資料から根拠を提示
- `Streamlit`ベースのWebインターフェース

---

## 🧩 実装構成

customer-contact/
├── .streamlit
│ └── config.toml
├── images/
├── logs/
├── data/
│ └── rag/
│   ├── company/
│   ├── service/
│   ├── customer/
│   ├── manual/
│   ├── policy/
│   └── sustainability/
│ └── slack/  
├── .db/ ← ChromaベクトルDB格納先
├── src/
│ ├── main.py ← アプリ起動エントリポイント
│ ├── initialize.py ← RAGエージェント・Tool初期化
│ ├── utils.py ← 各Toolの実行関数
│ ├── constants.py ← パス定義・Tool一覧
│ └── components.py ← UI構築（Streamlit）
├── requirements.txt
└── README.md

---

## 🧠 使用技術

- [LangChain](https://www.langchain.com/)
- [OpenAI API (GPT-4)](https://platform.openai.com/)
- [ChromaDB](https://www.trychroma.com/)（ベクトルストア）
- [Streamlit](https://streamlit.io/)（UI）
- Python 3.11

---

## 🔧 実装済みのTool一覧

| Tool名 | 主な対応内容 |
|--------|-----------------------------|
| `search_company_info_tool` | 会社概要・株主優待情報の回答 |
| `search_service_info_tool` | サービス全般・製品概要の回答 |
| `search_customer_communication_tool` | 顧客属性・購入者情報に関する回答 |
| `search_manual_info_tool`  | EcoTee Creatorの操作ガイド・手順書など |
| `search_policy_info_tool`  | 利用規約・返品/キャンセルルールなど |
| `search_sustainability_info_tool`  | 環境・エシカル・サステナビリティ対応など |


---

## 🛠️ セットアップ手順

git clone https://github.com/your-repo/customer-contact.git
cd customer-contact
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt

# 初期ベクトルDB構築（必要に応じて以下ファイルを作成）
python build_vectorstore.py
streamlit run src/main.py