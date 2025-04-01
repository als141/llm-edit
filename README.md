# LLM-Edit

LLM-Editは、OpenAI APIを活用してテキストファイルを対話的に編集するためのコマンドラインツールです。AIがユーザーの編集指示を解析し、ファイルの特定の部分や全体を編集することができます。

## 特徴

- 自然言語による編集指示
- 部分編集と全体編集の両方に対応
- 複数箇所の一括編集機能
- 編集前の変更確認と確認プロセス
- 編集提案に対するフィードバック機能
- UTF-8およびShift-JISエンコーディングに対応

## 必要条件

- OpenAI API キー

## インストール

1. リポジトリをクローンします：

```bash
git clone https://github.com/als141/llm-edit.git
cd llm-edit
```

2. 依存パッケージをインストールします：

```bash
pip install -r requirements.txt
```

3. `.env`ファイルを作成し、OpenAI APIキーを設定します：

```bash
# echoコマンドを使用して.envファイルを作成する方法
echo "OPENAI_API_KEY=your_api_key_here" > .env

# または、以下のように追記することもできます
echo "OPENAI_API_KEY=your_api_key_here" >> .env

# 直接編集する場合
touch .env
# お好みのエディタで開いて編集
# nano .env
# vim .env
# code .env
```

APIキーは[OpenAIのダッシュボード](https://platform.openai.com/api-keys)から取得できます。

## 使い方

### 基本的な使用方法

1. スクリプトを実行します：

```bash
python llm-edit.py
```

2. 編集したいファイルのパスを入力します。
3. 編集指示を入力します（例：「このファイルの最初の関数にコメントを追加して」）。
4. AIが提案する編集内容を確認し、適用するかどうかを選択します。

### コマンド

編集セッション中に以下のコマンドが使用できます：

- `/show` - 現在のファイル内容を表示
- `/quit` または `/exit` - プログラムを終了

### 編集プロセスの操作

編集提案が表示されたら、以下の操作が可能です：

- `y` - 提案された編集を適用
- `n` - 提案を拒否
- フィードバックを入力 - AIに提案を改善してもらう（例：「もう少し詳しく説明を追加して」）

## 使用例

### 部分編集の例

```
ファイルパス> sample.txt
sample.txt> 全体的に絵文字を追加して

[AI が編集案を提示]

この編集を適用しますか？ (y/n/フィードバックを入力) y
```

### フィードバックの例

```
ファイルパス> config.json
config.json> デバッグモードをtrueに変更して

[AI が編集案を提示]

この編集を適用しますか？ (y/n/フィードバックを入力) ログレベルも "debug" に変更して
[AI が新しい編集案を提示]

この編集を適用しますか？ (y/n/フィードバックを入力) y
```

## 主な機能

### 編集タイプ

LLM-Editは以下の編集タイプをサポートしています：

1. **単一の部分編集** - ファイル内の特定の一箇所を変更
2. **複数の部分編集** - ファイル内の複数箇所を一度に変更
3. **全体置換** - ファイル全体を新しい内容に置き換え

### 会話機能

単なる編集だけでなく、AIとファイルの内容について会話することも可能です：

- ファイルの特定部分について質問する
- コードの改善案を相談する
- ファイルの構造や目的について説明を求める

## トラブルシューティング

### APIキーの問題

エラーメッセージ「OpenAI APIキーが設定されていません」が表示される場合：
- `.env`ファイルが正しい場所（スクリプトと同じディレクトリ）にあることを確認
- APIキーが正しい形式で設定されていることを確認

### 編集提案の問題

「変更元テキストが見つかりません」などのエラーが表示される場合：
- より具体的な指示を試す
- `/show`コマンドで現在のファイル内容を確認
- フィードバック機能を使って、AIに別のアプローチを提案

## 注意事項

- 重要なファイルを編集する前に、必ずバックアップを取ってください
- APIの使用には課金が発生する場合があります
- デフォルトのモデルは`gpt-4o-mini`ですが、スクリプト内の`MODEL`変数を変更することで別のモデルを使用できます