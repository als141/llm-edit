# -*- coding: utf-8 -*-
import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from typing import List, Dict, Union, Optional, Tuple, Any

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI APIキーが設定されていません。.envファイルを確認してください。")

client = OpenAI(api_key=API_KEY)
MODEL = "gpt-4o-mini" # Or "gpt-4", "o1" etc.

console = Console()

SYSTEM_PROMPT_TEMPLATE = """
あなたはテキストファイルを編集するアシスタントAIです。ユーザーは編集したいファイルの内容と編集指示を与えます。
あなたはこれまでのユーザーとの会話履歴も参照できます。履歴には、過去のあなたの提案（JSON形式の場合あり）やそれに対するユーザーのフィードバックも含まれます。これらを理解し、文脈を踏まえて応答・提案してください。

**特に重要な指示: フィードバックの処理**
ユーザー入力が「前回の提案に対するフィードバック」として与えられた場合、以下のルールに従ってください。
1.  **編集対象の特定:**
    * 前回の提案が `replace_all` または `success` であった場合： **現在のファイル内容ではなく、ユーザーメッセージ内に提示される【前回の提案内容 (編集対象)】を編集対象**としてください。
    * 前回の提案が `multiple_edits` であった場合： **ユーザーメッセージ内に提示される【現在のファイル内容】を編集対象**としますが、**会話履歴に含まれる直前のAI提案（JSON形式）の内容を十分に考慮**し、フィードバックを反映させてください。
2.  **フィードバックの反映:** ユーザーのフィードバックを最優先で考慮し、特定した編集対象に対して修正を加えた新しい提案（元の提案と同じ形式、または `clarification_needed` など状況に応じた適切な形式）を生成してください。

あなたの主なタスクは、ユーザーの指示（またはフィードバック）に基づいて、ファイル内のどの部分をどのように変更するかを特定し、編集を実行することです。

以下の手順に従ってください。

1. ユーザーの指示（またはフィードバック）、提供されたファイル内容（または前回の提案内容）、そして会話履歴を注意深く読み取ります。
2. 変更したい箇所や内容を特定します。
    * **曖昧な指示への対応:** 会話履歴や文脈に基づいた具体的な変更案を複数提示することを試みてください。
    * **編集範囲:** 部分編集を基本とします。全体書き換えはユーザーが明確に意図している場合のみ検討します。
3. **編集タイプの判断と応答形式の選択:** ユーザー指示/フィードバックと編集対象、会話履歴から、以下のいずれかの編集タイプに該当するか判断し、指定されたJSON形式で**必ず**応答してください。

    * **A) 単一の部分編集 (`success`):** 特定の一箇所のみを変更する場合。`old_string` (文脈付き) が編集対象内で**正確に1回だけ**出現することを確認してください。
        ```json
        {{{{
          "status": "success",
          "old_string": "{{実際の変更前文字列}}",
          "new_string": "{{実際の変更後文字列}}"
        }}}}
        ```
    * **B) 複数の部分編集 (`multiple_edits`):** 複数の箇所を一度に変更する場合。各編集箇所の `old_string` が編集対象内で**正確に1回だけ**出現し、かつ編集箇所同士が**重複しない**ことを確認してください。
        ```json
        {{{{
          "status": "multiple_edits",
          "edits": [
            {{{{ "old_string": "{{変更箇所1の前}}", "new_string": "{{変更箇所1の後}}" }}}},
            // ... 他の編集箇所
          ]
        }}}}
        ```
    * **C) 全体置換 (`replace_all`):** 文章量の大幅な拡張、全面的な書き換え、全体的なトーン変更など、編集対象全体の内容を新しいものに置き換える場合。**ユーザーが明確に全体変更を意図しており、部分置換が困難な場合、またはフィードバックの結果として全体を修正する場合のみ**使用してください。
        ```json
        {{{{
          "status": "replace_all",
          "content": "{{新しいファイル内容全体}}"
        }}}}
        ```
    * **D) 要確認/情報不足 (`clarification_needed`):** 指示/フィードバックが曖昧、`old_string` が見つからない/一意でない/重複する、どの編集タイプにも当てはまらない、フィードバックに対して更に質問が必要な場合。具体的な質問や提案を含めてください。
        ```json
        {{{{
          "status": "clarification_needed",
          "message": "{{具体的な質問メッセージ}}"
        }}}}
        ```
    * **E) 会話/質問応答 (`conversation`):** ユーザーの入力が編集指示やフィードバックではなく、単なる会話やファイル内容に関する質問の場合。
        ```json
        {{{{
          "status": "conversation",
          "message": "{{会話応答メッセージ}}"
        }}}}
        ```
    * **F) 拒否 (`rejected`):** 指示/フィードバックが危険、不適切、または実行不可能な場合。
        ```json
        {{{{
          "status": "rejected",
          "message": "{{拒否理由メッセージ}}"
        }}}}
        ```

**注意点:**
* `old_string` は、変更箇所を一意に特定できる十分な文脈を含めてください。
* `multiple_edits` の場合、各 `old_string` は一意であり、編集箇所が重ならないようにしてください。
* フィードバックに対しては、その内容を最優先で考慮し、提案を修正または具体化してください。

必ず上記のいずれかのJSON形式で応答してください。
"""

Message = Dict[str, str]
ConversationHistory = List[Message]
EditOperation = Dict[str, str]
AiResponse = Dict[str, Any]


def get_file_path() -> Path:
    """ユーザーから有効なファイルパスを取得する"""
    while True:
        try:
            filepath_str = Prompt.ask("[bold cyan]編集したいファイルのパスを入力してください[/]")
            filepath = Path(filepath_str).resolve()
            if filepath.is_file():
                return filepath
            else:
                console.print(f"[bold red]エラー: ファイルが見つかりません: {filepath}[/]")
        except Exception as e:
            console.print(f"[bold red]エラー: 無効なパスです: {e}[/]")

def read_file(filepath: Path) -> Optional[str]:
    """ファイルを読み込む (UTF-8 -> Shift-JIS フォールバック)"""
    try:
        return filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return filepath.read_text(encoding='shift_jis')
        except Exception as e:
            console.print(f"[bold red]エラー: ファイルの読み込みに失敗しました (Shift-JIS試行後) ({filepath}): {e}[/]")
            return None
    except Exception as e:
        console.print(f"[bold red]エラー: ファイルの読み込みに失敗しました ({filepath}): {e}[/]")
        return None

def determine_encoding(filepath: Path) -> str:
    """ファイルのエンコーディングを判別する (簡易版)"""
    try:
        filepath.read_text(encoding='utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        return 'shift_jis'
    except Exception:
        return 'utf-8'

def write_file(filepath: Path, content: str, encoding: str) -> bool:
    """ファイルに書き込む (指定されたエンコーディングで)"""
    try:
        filepath.write_text(content, encoding=encoding)
        return True
    except Exception as e:
        console.print(f"[bold red]エラー: ファイルへの書き込みに失敗しました ({filepath}, encoding={encoding}): {e}[/]")
        return False

def get_openai_response(
    current_file_content: str,
    latest_user_content: str,
    history: ConversationHistory,
    is_feedback: bool,
    previous_proposal: Optional[AiResponse]
) -> Optional[AiResponse]:
    """OpenAI APIを呼び出し、編集指示を解析する (フィードバック対応版)"""
    messages: ConversationHistory = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE}
    ]
    messages.extend(history)

    last_user_message_combined = ""
    if is_feedback and previous_proposal:
        prev_status = previous_proposal.get("status")
        editable_previous_content = None

        if prev_status == "replace_all":
            editable_previous_content = previous_proposal.get("content")
        elif prev_status == "success":
            editable_previous_content = previous_proposal.get("new_string")

        if editable_previous_content is not None and isinstance(editable_previous_content, str):
             last_user_message_combined = f"""
## 【前回の提案内容 (編集対象)】:
---
{editable_previous_content}
---
## 【上記提案に対するフィードバック】: {latest_user_content}

会話履歴と上記のフィードバックに基づき、**【前回の提案内容 (編集対象)】を修正**して、新しい提案をJSON形式で応答してください。(元の提案形式: {prev_status})
"""
        else:
             last_user_message_combined = f"""
## 【現在のファイル内容】:
---
{current_file_content}
---
## 【前回の提案 (履歴参照)】: {prev_status} 提案 (詳細は会話履歴の直前のassistantメッセージを確認してください)
## 【上記提案に対するフィードバック】: {latest_user_content}

会話履歴（特に直前のAI提案JSON）と上記のフィードバックに基づき、**【現在のファイル内容】を編集**して、新しい提案をJSON形式で応答してください。
"""
    else:
        last_user_message_combined = f"""
## 【現在のファイル内容】:
---
{current_file_content}
---
## 【ユーザー指示】: {latest_user_content}

会話履歴も踏まえ、上記の指示に基づいて、JSON形式で応答してください。
"""

    messages.append({"role": "user", "content": last_user_message_combined})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages, # type: ignore
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        if response_content:
            processed_content = response_content.strip()
            if processed_content.startswith("```json"):
                processed_content = processed_content[7:]
                if processed_content.endswith("```"):
                    processed_content = processed_content[:-3]
            elif processed_content.startswith("```"):
                 processed_content = processed_content[3:]
                 if processed_content.endswith("```"):
                     processed_content = processed_content[:-3]

            processed_content = processed_content.strip()
            try:
                parsed_json: AiResponse = json.loads(processed_content)
                if isinstance(parsed_json, dict):
                    status = parsed_json.get("status")
                    if not status:
                        console.print(f"[bold red]エラー: AI応答JSONに 'status' が含まれていません。[/]")
                        console.print(f"受信内容(加工後): {processed_content}")
                        return None
                    return parsed_json
                else:
                    console.print(f"[bold red]エラー: OpenAIからの応答が予期した辞書形式ではありません。[/]")
                    console.print(f"受信内容(加工後): {processed_content}")
                    return None
            except json.JSONDecodeError as e:
                 console.print(f"[bold red]エラー: OpenAIからの応答JSONの解析に失敗しました: {e}[/]")
                 console.print(f"受信内容(加工前):\n{response_content}")
                 console.print(f"受信内容(加工後):\n{processed_content}")
                 return None
        else:
            console.print("[bold red]エラー: OpenAIからの応答が空です。[/]")
            return None
    except Exception as e:
        console.print(f"[bold red]エラー: OpenAI APIの呼び出し中にエラーが発生しました: {e}[/]")
        return None


def ask_confirmation(prompt_message: str) -> str:
    """Asks 'y/n/feedback' and returns 'y', 'n', or feedback string."""
    while True:
        response = Prompt.ask(prompt_message, default="").strip()
        response_lower = response.lower()
        if response_lower == 'y':
            return 'y'
        elif response_lower == 'n':
            return 'n'
        elif response:
            return response

def verify_and_apply_edit(
    filepath: Path, file_content: str, old_string: str, new_string: str, encoding: str
) -> Tuple[str, Optional[str]]:
    try:
        count = file_content.count(old_string)
    except Exception as e:
        error_msg = f"変更元テキストの検索中にエラーが発生しました: {e}"
        console.print(f"[bold red]{error_msg}[/]")
        console.print(f"[dim]検索対象:\n---\n{old_string!r}\n---[/dim]")
        return "error", error_msg

    if count == 0:
        message = "エラー: AIが指定した変更元テキストが現在のファイルに見つかりませんでした。"
        console.print(f"[bold red]{message}[/]")
        console.print(f"[dim]検索対象:[/]\n---\n{old_string}\n---")
        return "error", message
    elif count > 1:
        console.print(f"[bold yellow]警告: 変更元テキストがファイル内に複数 ({count}箇所) 見つかりました。[/]")
        console.print(f"[dim]検索対象:[/]\n---\n{old_string}\n---")
        user_choice_multi = ask_confirmation("最初の1箇所に適用しますか？ (y/n/フィードバックを入力)")
        if user_choice_multi == 'y':
            pass
        elif user_choice_multi == 'n':
             return "cancelled", "複数箇所ヒットのため編集はキャンセルされました。"
        else:
             return "feedback", user_choice_multi

    console.print("\n[bold green]以下の編集案が見つかりました:[/]")
    console.print("[bold red]- 変更前:[/]")
    console.print(f"{old_string}")
    console.print("[bold blue]+ 変更後:[/]")
    console.print(f"{new_string}")

    user_choice = ask_confirmation("\nこの編集を適用しますか？ (y/n/フィードバックを入力)")

    if user_choice == 'y':
        try:
            new_content = file_content.replace(old_string, new_string, 1)
            if write_file(filepath, new_content, encoding):
                history_msg = f"単一編集を適用: '{old_string[:30].strip()}...' -> '{new_string[:30].strip()}...'"
                return "applied", history_msg
            else:
                return "error", "ファイル書き込みエラーが発生しました。"
        except Exception as e:
            error_msg = f"ファイル内容の置換中にエラーが発生しました: {e}"
            console.print(f"[bold red]{error_msg}[/]")
            return "error", error_msg
    elif user_choice == 'n':
        return "cancelled", "編集はユーザーによってキャンセルされました。"
    else:
        return "feedback", user_choice

def verify_and_apply_multiple_edits(
    filepath: Path, file_content: str, edits: List[EditOperation], encoding: str
) -> Tuple[str, Optional[str]]:
    validated_edits_with_indices = []
    problems = []
    overlapping_indices = set()
    temp_validation_content = file_content

    for i, edit in enumerate(edits):
        old = edit.get("old_string")
        new = edit.get("new_string")
        if not old or not isinstance(old, str) or new is None or not isinstance(new, str):
            problems.append(f"- 編集 {i+1}: 'old_string' または 'new_string' が無効です。 Edit: {edit}")
            continue

        try:
             indices = [idx for idx in range(len(temp_validation_content)) if temp_validation_content.startswith(old, idx)]
        except Exception as e:
             problems.append(f"- 編集 {i+1}: 変更元テキスト検索中にエラーが発生しました: {e}\n  テキスト: {old[:100]}...")
             continue

        if len(indices) == 0:
            problems.append(f"- 編集 {i+1}: 変更元テキストが見つかりません:\n  ---\n  {old}\n  ---")
        elif len(indices) > 1:
            problems.append(f"- 編集 {i+1}: 変更元テキストが複数 ({len(indices)}箇所) 見つかりました:\n  ---\n  {old}\n  ---")
        else:
            start_index = indices[0]
            end_index = start_index + len(old)
            current_range = set(range(start_index, end_index))
            if not current_range.isdisjoint(overlapping_indices):
                problems.append(f"- 編集 {i+1}: 他の編集箇所と重複しています:\n  ---\n  {old}\n  ---")
            else:
                validated_edits_with_indices.append({"start": start_index, "end": end_index, "old": old, "new": new})
                overlapping_indices.update(current_range)

    if problems:
        console.print("[bold red]エラー: 提案された複数の編集内容に問題が見つかりました:[/]")
        for problem in problems:
            console.print(problem)
        console.print("[yellow]編集は適用できません。AIに修正を依頼するか、指示をより具体的にしてください。[/]")
        return "error", "複数編集の検証エラーが発生しました。上記詳細を確認してください。"

    console.print("\n[bold green]以下の複数の編集案が見つかりました:[/]")
    validated_edits_with_indices.sort(key=lambda item: item["start"])
    for i, edit_info in enumerate(validated_edits_with_indices):
        console.print(f"\n--- 編集 {i+1} ---")
        console.print("[bold red]- 変更前:[/]")
        console.print(f"{edit_info['old']}")
        console.print("[bold blue]+ 変更後:[/]")
        console.print(f"{edit_info['new']}")

    user_choice = ask_confirmation("\nこれらの編集をすべて適用しますか？ (y/n/フィードバックを入力)")

    if user_choice == 'y':
        try:
            new_content_parts = []
            last_index = 0
            applied_count = 0
            for edit_info in validated_edits_with_indices:
                start_index = edit_info["start"]
                end_index = edit_info["end"]
                new_string = edit_info["new"]
                new_content_parts.append(file_content[last_index:start_index])
                new_content_parts.append(new_string)
                last_index = end_index
                applied_count += 1
            new_content_parts.append(file_content[last_index:])
            final_content = "".join(new_content_parts)

            if write_file(filepath, final_content, encoding):
                 history_msg = f"複数編集 ({applied_count}箇所) を適用しました。"
                 return "applied", history_msg
            else:
                 return "error", "ファイル書き込みエラーが発生しました。"
        except Exception as e:
            error_msg = f"ファイル内容の複数置換中にエラーが発生しました: {e}"
            console.print(f"[bold red]{error_msg}[/]")
            return "error", error_msg
    elif user_choice == 'n':
        return "cancelled", "複数箇所の編集はキャンセルされました。"
    else: # Feedback
        return "feedback", user_choice

def handle_replace_all_confirmation(
    filepath: Path, current_content: str, new_content_full: str, encoding: str
) -> Tuple[str, Optional[str]]:
    console.print("\n[bold green]AIによるファイル全体の書き換え案:[/]")
    console.print(f"元の内容 約{len(current_content)}文字 -> 新しい内容 約{len(new_content_full)}文字")

    console.print("[bold blue]--- 新しい内容 ---[/]")
    console.print(new_content_full)
    console.print("[bold blue]--- 新しい内容終 ---[/]")

    user_choice = ask_confirmation("\nファイル全体をこの新しい内容で上書きしますか？ (y/n/フィードバックを入力)")

    if user_choice == 'y':
        if write_file(filepath, new_content_full, encoding):
            history_msg = "ファイル全体をAI提案で書き換えました。"
            return "applied", history_msg
        else:
            return "error", "ファイル書き込みエラーが発生しました。"
    elif user_choice == 'n':
        return "cancelled", "ファイル全体の書き換えはキャンセルされました。"
    else: # Feedback
        return "feedback", user_choice

if __name__ == "__main__":
    console.print("[bold magenta]AIテキスト編集チャットへようこそ！[/]")
    filepath = get_file_path()
    file_encoding = determine_encoding(filepath)
    console.print(f"[green]編集対象ファイル: {filepath} (エンコーディング: {file_encoding})[/]")

    conversation_history: ConversationHistory = []
    state = "get_instruction"
    user_input = ""
    last_ai_proposal: Optional[AiResponse] = None

    while True:
        console.rule()
        latest_user_content_for_api = ""

        if state == "get_instruction":
            user_input = Prompt.ask(f"[bold yellow]{filepath.name}>[/]", default="")

            if not user_input.strip(): continue

            if user_input.lower() in ["/quit", "/exit"]:
                console.print("[bold magenta]終了します。[/]")
                break
            if user_input.lower() == "/show":
                current_content_display = read_file(filepath)
                if current_content_display is not None:
                    console.print("\n[bold cyan]--- ファイル内容 ---[/]")
                    try:
                        console.print(current_content_display)
                    except UnicodeEncodeError:
                         console.print("[red]コンソールへのファイル内容表示中にエンコーディングエラーが発生しました。[/red]")
                         console.print(f"[dim]{current_content_display!r}[/dim]")
                    console.print("[bold cyan]--- ファイル内容終 ---[/]")
                else:
                    console.print("[red]ファイル内容の表示に失敗しました。[/red]")
                continue

            latest_user_content_for_api = user_input
            conversation_history.append({"role": "user", "content": latest_user_content_for_api})
            console.print("[dim]AIが考えています...[/]")

        elif state == "process_feedback":
            console.print("[dim]フィードバックを元にAIが再考しています...[/]")

            if last_ai_proposal:
                try:
                    proposal_json = json.dumps(last_ai_proposal, ensure_ascii=False)
                    conversation_history.append({"role": "assistant", "content": proposal_json})
                except Exception as e:
                    console.print(f"[yellow]警告: 前回のAI提案を履歴に追加中にエラー: {e}[/yellow]")

            conversation_history.append({"role": "user", "content": user_input})
            latest_user_content_for_api = user_input

        MAX_HISTORY_PAIRS = 10
        if len(conversation_history) > MAX_HISTORY_PAIRS * 2:
            history_cutoff = len(conversation_history) - (MAX_HISTORY_PAIRS * 2)
            conversation_history = conversation_history[history_cutoff:]

        current_content = read_file(filepath)
        if current_content is None:
            if conversation_history and conversation_history[-1].get("role") == "user":
                 conversation_history.pop()
                 if state == "process_feedback" and conversation_history and conversation_history[-1].get("role") == "assistant":
                     conversation_history.pop()
            state = "get_instruction"
            continue

        is_feedback_call = (state == "process_feedback")

        ai_response = get_openai_response(
            current_file_content=current_content,
            latest_user_content=latest_user_content_for_api,
            history=conversation_history[:-1] if conversation_history else [],
            is_feedback=is_feedback_call,
            previous_proposal=last_ai_proposal if is_feedback_call else None
        )

        if ai_response:
            last_ai_proposal = ai_response
        else:
            pass

        if ai_response:
            status = ai_response.get("status")
            action_result = None
            feedback_or_history_msg = None

            if status == "success":
                old = ai_response.get("old_string")
                new = ai_response.get("new_string")
                if old is not None and new is not None and isinstance(old, str) and isinstance(new, str):
                    action_result, feedback_or_history_msg = verify_and_apply_edit(
                        filepath, current_content, old, new, file_encoding
                    )
                else:
                    action_result, feedback_or_history_msg = "error", "AI応答の形式不備 (success: old/new string invalid)"
                    console.print(f"[bold red]{feedback_or_history_msg}[/]")

            elif status == "multiple_edits":
                edits = ai_response.get("edits")
                if edits and isinstance(edits, list):
                    valid_edits = [e for e in edits if isinstance(e, dict) and isinstance(e.get("old_string"), str) and isinstance(e.get("new_string"), str)]
                    if len(valid_edits) == len(edits):
                         action_result, feedback_or_history_msg = verify_and_apply_multiple_edits(
                              filepath, current_content, valid_edits, file_encoding
                         )
                    else:
                         action_result, feedback_or_history_msg = "error", f"AI応答の形式不備 (multiple_edits: {len(edits) - len(valid_edits)}件の編集項目が無効)"
                         console.print(f"[bold red]{feedback_or_history_msg}[/]")
                else:
                    action_result, feedback_or_history_msg = "error", "AI応答の形式不備 (multiple_edits: editsキーがない、またはリストではない)"
                    console.print(f"[bold red]{feedback_or_history_msg}[/]")

            elif status == "replace_all":
                new_content_full = ai_response.get("content")
                if new_content_full is not None and isinstance(new_content_full, str):
                    action_result, feedback_or_history_msg = handle_replace_all_confirmation(
                        filepath, current_content, new_content_full, file_encoding
                    )
                else:
                    action_result, feedback_or_history_msg = "error", "AI応答の形式不備 (replace_all: contentキーがない、または文字列ではない)"
                    console.print(f"[bold red]{feedback_or_history_msg}[/]")

            elif status in ["clarification_needed", "conversation", "rejected"]:
                message = ai_response.get('message', f'AIからのメッセージ ({status})')
                color = "yellow" if status == "clarification_needed" else "blue" if status == "conversation" else "red"
                console.print(f"[bold {color}]AI:[/][{color}] {message}[/]")
                conversation_history.append({"role": "assistant", "content": message})
                state = "get_instruction"

            else:
                message = f"エラー: AIから不明なステータス ('{status}') が返されました。"
                console.print(f"[bold red]{message}[/]")
                console.print(f"応答全体: {ai_response}")
                if conversation_history and conversation_history[-1].get("role") == "user":
                     conversation_history.pop()
                state = "get_instruction"

            if action_result:
                if action_result == "applied":
                    if feedback_or_history_msg:
                         conversation_history.append({"role": "assistant", "content": feedback_or_history_msg})
                    state = "get_instruction"
                elif action_result == "cancelled":
                    if feedback_or_history_msg:
                         conversation_history.append({"role": "assistant", "content": feedback_or_history_msg})
                    state = "get_instruction"
                elif action_result == "error":
                    error_summary = f"エラー発生: {feedback_or_history_msg}" if feedback_or_history_msg else "提案処理中にエラーが発生しました。"
                    conversation_history.append({"role": "assistant", "content": error_summary})
                    state = "get_instruction"
                elif action_result == "feedback":
                    user_input = feedback_or_history_msg or ""
                    state = "process_feedback"
                else:
                    console.print(f"[bold red]内部エラー: 不明なアクション結果 '{action_result}'[/]")
                    state = "get_instruction"

        else:
            if conversation_history and conversation_history[-1].get("role") == "user":
                conversation_history.pop()
            state = "get_instruction"