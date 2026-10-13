"""3조 - MES 소스 변경 분석 에이전트 (MultiAgent)

같은 소스의 '수정 전 / 수정 후' 파일을 받아 무엇이 바뀌었는지, MES 업무에 어떤 영향이 있는지,
위험한 변경은 없는지 리뷰해 줍니다. 파일이 하나면 그 소스가 무슨 일을 하는지 설명합니다.

그래프 구조 (노드 7개: 파이썬 2개 + LLM 5개)
    analyze_request → load_sources ──┬─ (이어지는 질문) ─────→ answer_follow_up → END
                                     ├─ (파일 2개) ─→ compute_diff → review_changes → END
                                     ├─ (파일 1개 · 설명 요청) → explain_source ──→ END
                                     └─ (파일 부족) ──────────→ request_files ───→ END

핵심 설계: 파일 받기, diff 계산, 변경된 함수 찾기, 위험 패턴 점검은 LLM 이 아니라 파이썬이 합니다.
LLM 에게 두 파일을 통째로 주고 "뭐가 바뀌었어?" 라고 하면 바뀐 줄을 놓치거나
바뀌지 않은 줄을 바뀌었다고 말합니다. 그래서 '어디가 바뀌었는지' 는 코드가 정확히 계산하고,
LLM 은 '왜, 그래서 어떤 영향이 있는지' 를 리뷰하는 일만 맡습니다.

소스를 받는 방법 (README '0. 소스 파일 넣는 법' 참고)
  1) 채팅에 ``` 코드 블록 2개 붙여넣기 (수정 전 / 수정 후)
  2) teams/team3/inbox/ 폴더에 파일을 넣고 파일명으로 요청 ("LotTrackIn.cs 비교해줘")
  3) 메시지 두 번에 나눠 보내기 (이전 대화에서 코드를 다시 찾아 짝을 맞춥니다)

터미널에서 이 파일만 바로 실행해 볼 수 있습니다.

    python teams/team3/workflow.py "LotTrackIn.cs 수정 전후 비교해줘"
    python teams/team3/workflow.py --before a_old.cs --after a.cs "예외 처리 위주로 봐줘"
    python teams/team3/workflow.py --inbox
"""

from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# 이 파일을 직접 실행할 때도 app/ 을 찾을 수 있도록 최상위 폴더를 경로에 추가합니다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph

from app.contract import BaseGraphState
from app.llm import get_llm
from teams.team3.prompts import (
    ANALYZE_REQUEST_SYSTEM,
    EXPLAIN_SOURCE_SYSTEM,
    FOLLOW_UP_SYSTEM,
    REQUEST_FILES_SYSTEM,
    REVIEW_CHANGES_SYSTEM,
)

TEAM_INFO = {
    "name": "3조 MES 소스 리뷰 선배봇",
    "description": "수정 전/후 소스를 주면 무엇이 바뀌었는지, MES 업무 영향과 위험한 변경을 리뷰해 드립니다. 소스 하나만 주면 무슨 일을 하는 코드인지 설명합니다.",
    # inbox 폴더의 예제 파일(LotTrackIn_old.cs / LotTrackIn.cs)로 바로 돌려볼 수 있는 질문입니다.
    "examples": [
        "LotTrackIn.cs 수정 전후 비교해줘",
        "LotTrackIn_old.cs 이 소스가 무슨 일을 하는지 설명해줘",
        "소스 변경 리뷰 해줘",
    ],
}

# ── 설정값 (사내 LLM 의 입력 길이에 맞춰 조절하세요) ─────────────────
INBOX_DIR = Path(__file__).resolve().parent / "inbox"   # 파일을 넣어두는 폴더
MAX_FILE_BYTES = 1_000_000   # inbox 파일 최대 크기 (1MB)
MAX_DIFF_CHARS = 12_000      # LLM 에 넘길 diff 최대 글자 수
MAX_SOURCE_CHARS = 15_000    # 설명 모드에서 LLM 에 넘길 소스 최대 글자 수
CONTEXT_LINES = 3            # 변경 줄 앞뒤로 함께 보여줄 문맥 줄 수
MAX_HINTS = 15               # 자동 점검 결과 최대 개수
HISTORY_TURNS = 6            # 프롬프트에 함께 보낼 이전 대화 개수


class SourceState(BaseGraphState):
    """3조가 사용하는 상태입니다. BaseGraphState 를 상속해 필드를 더했습니다."""

    intent: str               # "new"(새 요청) / "followup"(방금 답변에 이어지는 질문)
    task: str                 # "compare"(수정 전/후 비교) / "explain"(소스 하나 설명)
    file_names: list[str]     # 메시지에 나온 파일 이름
    focus: str | None         # 특히 봐달라고 한 관점 (성능, 예외 처리 ...)
    before: dict | None       # 수정 전 소스 {"name", "code", "lang", "origin"}
    after: dict | None        # 수정 후 소스 (파일이 하나면 여기에 들어갑니다)
    source_status: str        # "pair"(2개) / "single"(1개) / "none"(없음)
    source_notes: list[str]   # 파일을 받으면서 생긴 문제 (inbox 에 없음 등)
    diff_text: str            # 행 번호가 붙은 diff (LLM 입력용)
    diff_stats: dict          # {"hunks", "added", "removed", "shown", "truncated"}
    changed_funcs: list[str]  # 변경이 일어난 함수 이름
    risk_hints: list[str]     # 패턴으로 찾은 위험 의심 지점


# ═════════════════════════════════════════════════════════════
# 1) 소스 받기 (LLM 없음) - 코드 블록 / 구분선 / inbox 파일 / 이전 대화
# ═════════════════════════════════════════════════════════════

CODE_EXTS = ("cs|java|py|sql|pls|pks|pkb|prc|fnc|trg|js|ts|cpp|cc|c|h|hpp|vb|"
             "xml|json|ya?ml|config|properties|txt|bak|orig|old")
# 한국어 조사가 바로 붙어도("LotTrackIn.cs랑") 파일명을 잡도록 끝을 영문/숫자로만 끊습니다.
FILE_NAME_RE = re.compile(rf"[A-Za-z0-9_\-.가-힣]+\.(?:{CODE_EXTS})(?![A-Za-z0-9_])", re.I)
FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)

# 수정 전/후를 알려주는 표시. 영어 단어는 앞뒤가 알파벳이 아닐 때만 인정합니다. ("Hold" 의 old 는 X)
_BEFORE_RE = re.compile(r"수정\s*전|변경\s*전|원본|기존|(?<![a-z])(?:before|as-?is|old|orig|original|prev|bak|backup)(?![a-z])", re.I)
_AFTER_RE = re.compile(r"수정\s*후|변경\s*후|신규|(?<![a-z])(?:after|to-?be|new|modified|mod)(?![a-z])", re.I)

# "--- 수정 전 ---", "[AS-IS]", "수정후:" 처럼 한 줄짜리 구분선
_MARK_WORDS = {
    "수정전": "before", "변경전": "before", "원본": "before", "기존": "before",
    "before": "before", "asis": "before", "old": "before",
    "수정후": "after", "변경후": "after", "신규": "after",
    "after": "after", "tobe": "after", "new": "after",
}


def _side(text: str | None) -> str | None:
    """텍스트(파일명이나 코드 블록 앞 문장)에서 수정 전/후 표시를 찾습니다. 가장 뒤에 나온 표시를 따릅니다."""
    if not text:
        return None
    marks = [(m.start(), "before") for m in _BEFORE_RE.finditer(text)]
    marks += [(m.start(), "after") for m in _AFTER_RE.finditer(text)]
    return max(marks)[1] if marks else None


def _read_text(path: Path) -> str:
    """사내 소스는 UTF-8 이 아닌 CP949(EUC-KR) 인 경우가 많아 순서대로 시도합니다."""
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _split_by_markers(text: str) -> tuple[list[dict], str]:
    """코드 블록 없이 구분선으로 나눠 붙여넣은 코드를 잘라냅니다. → (소스 목록, 코드를 줄인 텍스트)"""
    lines = text.splitlines()
    marks = []
    for i, line in enumerate(lines):
        core = re.sub(r"[\s\-=#\[\]*:<>()|/]+", "", line).lower()
        core = re.sub(r"(소스|코드|파일)$", "", core)
        if core in _MARK_WORDS:
            marks.append((i, _MARK_WORDS[core]))
    if len(marks) < 2:
        return [], text

    sources, compact = [], lines[: marks[0][0]]
    for n, (i, side) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)
        code = "\n".join(lines[i + 1: end]).strip("\n")
        compact.append(lines[i])
        if code.strip():
            sources.append({"name": None, "code": code, "side": side, "origin": "붙여넣기", "hint": ""})
            compact.append(f"[코드 {code.count(chr(10)) + 1}줄]")
    return sources, "\n".join(compact)


def _sources_from_text(text: str) -> list[dict]:
    """메시지에 들어있는 코드를 꺼냅니다. ``` 코드 블록 우선, 없으면 구분선."""
    blocks = list(FENCE_RE.finditer(text))
    if not blocks:
        return _split_by_markers(text)[0]

    found, prev_end = [], 0
    for m in blocks:
        # 코드 블록 바로 앞 문장("수정 전 LotTrackIn_old.cs:")으로 전/후와 파일명을 알아냅니다.
        label = text[prev_end: m.start()][-80:] + " " + m.group(1)
        prev_end = m.end()
        code = m.group(2).rstrip()
        if not code.strip():
            continue
        names = FILE_NAME_RE.findall(label)
        found.append({
            "name": names[-1] if names else None,
            "code": code,
            "side": _side(label),
            "origin": "코드 블록",
            "hint": m.group(1).strip(),
        })
    return found


def compact_text(text: str) -> str:
    """LLM 에 보낼 때 긴 코드를 "[코드 블록 N줄]" 로 줄입니다. (분석 노드는 코드를 따로 받습니다)"""
    if FENCE_RE.search(text):
        return FENCE_RE.sub(lambda m: f"[코드 블록 {m.group(2).count(chr(10)) + 1}줄]", text)
    return _split_by_markers(text)[1]


def inbox_files() -> list[str]:
    if not INBOX_DIR.is_dir():
        return []
    return sorted(p.name for p in INBOX_DIR.iterdir() if p.is_file() and not p.name.startswith("."))


def _load_inbox(name: str, notes: list[str] | None = None) -> dict | None:
    """inbox 에서 파일을 읽습니다. 경로 조작(../)을 막기 위해 파일 이름만 사용합니다."""
    name = Path(name).name.lower()
    for f in inbox_files():
        if f.lower() == name:
            path = INBOX_DIR / f
            if path.stat().st_size > MAX_FILE_BYTES:
                if notes is not None:
                    notes.append(f"'{f}' 파일이 {MAX_FILE_BYTES // 1000}KB 보다 커서 읽지 않음")
                return None
            return {"name": f, "code": _read_text(path), "side": _side(f), "origin": "inbox", "hint": ""}
    return None


_PAIR_MARK_RE = re.compile(
    r"(?:[_\-. ]?(?:수정전|수정후|원본)|[_\-.](?:old|bak|backup|before|asis|as-is|orig|original|prev|"
    r"new|after|tobe|to-be|mod|modified|v\d+|\d{6,8}))+$", re.I)


def _base_key(name: str) -> str:
    """'LotTrackIn_old.cs', 'LotTrackIn.cs.bak', 'LotTrackIn_v2.cs' → 'lottrackin.cs' (짝 찾기용)"""
    p = Path(name.lower())
    while p.suffix in (".bak", ".orig", ".old", ".backup"):
        p = Path(p.stem)
    return _PAIR_MARK_RE.sub("", p.stem) + p.suffix


def _inbox_pair(name: str) -> list[dict] | None:
    """파일 하나만 말해도 inbox 에서 같은 소스의 다른 버전을 찾아 [수정 전, 수정 후] 로 돌려줍니다."""
    key = _base_key(name)
    group = [f for f in inbox_files() if _base_key(f) == key]
    if len(group) < 2:
        return None

    def order(f: str):
        side = _side(f)
        rank = 0 if side == "before" else 2 if side == "after" else 1   # 표시 없는 파일은 가운데
        return (rank, (INBOX_DIR / f).stat().st_mtime)                  # 같으면 오래된 파일이 먼저

    group.sort(key=order)
    pair = [_load_inbox(group[0]), _load_inbox(group[-1])]
    return pair if all(pair) else None


def _from_attachments(state: dict) -> list[dict]:
    """공용 앱이 첨부 파일을 state 로 넘겨주는 경우를 대비한 입구입니다. (없으면 아무 일도 안 함)
    지원 형식: [{"name": ..., "content": ...}] 또는 서버의 파일 경로 문자열 목록"""
    found = []
    for key in ("files", "attachments", "uploaded_files"):
        for item in state.get(key) or []:
            if isinstance(item, dict) and item.get("content"):
                name = item.get("name")
                found.append({"name": name, "code": str(item["content"]), "side": _side(name),
                              "origin": "첨부", "hint": ""})
            elif isinstance(item, str) and Path(item).is_file():
                path = Path(item)
                found.append({"name": path.name, "code": _read_text(path), "side": _side(path.name),
                              "origin": "첨부", "hint": ""})
    return found


def _mentioned_names(text: str, extra: list[str] | None = None) -> list[str]:
    names = []
    for n in FILE_NAME_RE.findall(compact_text(text)) + list(extra or []):
        if n and "." in n and n.lower() not in [x.lower() for x in names]:
            names.append(n)
    return names


def _collect_turn(text: str, extra_names: list[str], task: str,
                  notes: list[str], quiet: bool = False) -> list[dict]:
    """메시지 하나에서 소스를 모읍니다. (코드 블록 → 파일명으로 inbox → inbox 에서 짝 찾기)"""
    sources = _sources_from_text(text)
    for name in _mentioned_names(text, extra_names):
        if any(s["name"] and s["name"].lower() == name.lower() for s in sources):
            continue    # 코드 블록으로 이미 받은 파일
        loaded = _load_inbox(name, None if quiet else notes)
        if loaded:
            sources.append(loaded)
        elif not quiet:
            notes.append(f"'{name}' 파일을 inbox 에서 찾지 못함")

    if task == "compare" and len(sources) == 1 and sources[0]["origin"] == "inbox":
        pair = _inbox_pair(sources[0]["name"])
        if pair:
            sources = pair
    return sources


def _from_history(messages: list[dict], task: str, need: int) -> list[dict]:
    """이전 사용자 메시지에서 소스를 찾습니다. 최근 메시지부터 거슬러 올라가며, 오래된 순서로 돌려줍니다."""
    found: list[dict] = []
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        found = _collect_turn(msg.get("content", ""), [], task, [], quiet=True) + found
        if len(found) >= need:
            break
    return found[-need:] if need else found


def _assign(sources: list[dict]) -> tuple[dict, dict]:
    """소스 여러 개 중 [수정 전, 수정 후] 를 고릅니다. 표시가 없으면 먼저 받은 쪽이 수정 전입니다."""
    befores = [s for s in sources if s.get("side") == "before"]
    afters = [s for s in sources if s.get("side") == "after"]
    if befores and afters:
        return befores[-1], afters[-1]
    a, b = sources[-2:]
    if a.get("side") == "after" or b.get("side") == "before":
        a, b = b, a
    return a, b


# ═════════════════════════════════════════════════════════════
# 2) 소스 분석 도구 (LLM 없음) - 언어 판별 / 함수 찾기 / diff / 위험 패턴
# ═════════════════════════════════════════════════════════════

EXT_LANG = {
    ".cs": "C#", ".java": "Java", ".py": "Python", ".sql": "SQL",
    ".pls": "PL/SQL", ".pks": "PL/SQL", ".pkb": "PL/SQL", ".prc": "PL/SQL", ".fnc": "PL/SQL", ".trg": "PL/SQL",
    ".js": "JavaScript", ".ts": "TypeScript", ".cpp": "C++", ".cc": "C++", ".c": "C", ".h": "C/C++",
    ".hpp": "C++", ".vb": "VB.NET", ".xml": "XML", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".config": "XML 설정", ".properties": "설정 파일",
}
FENCE_LANG = {
    "cs": "C#", "csharp": "C#", "c#": "C#", "java": "Java", "py": "Python", "python": "Python",
    "sql": "SQL", "plsql": "PL/SQL", "js": "JavaScript", "javascript": "JavaScript", "ts": "TypeScript",
    "cpp": "C++", "c++": "C++", "c": "C", "vb": "VB.NET", "xml": "XML", "json": "JSON",
}


def detect_lang(name: str | None, code: str, hint: str = "") -> str:
    if name:
        p = Path(name.lower())
        while p.suffix in (".bak", ".orig", ".old", ".backup"):
            p = Path(p.stem)
        if p.suffix in EXT_LANG:
            return EXT_LANG[p.suffix]
    if hint and hint.split()[0].lower() in FENCE_LANG:
        return FENCE_LANG[hint.split()[0].lower()]
    if re.search(r"^\s*using\s+System|^\s*namespace\s+[\w.]+", code, re.M):
        return "C#"
    if re.search(r"^\s*(?:package\s+[\w.]+;|import\s+java\.)", code, re.M):
        return "Java"
    if re.search(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:PROCEDURE|FUNCTION|PACKAGE|TRIGGER)\b", code, re.I):
        return "PL/SQL"
    if re.search(r"^\s*(?:SELECT|UPDATE|INSERT|DELETE|MERGE)\b", code, re.I | re.M):
        return "SQL"
    if re.search(r"^\s*(?:async\s+)?def\s+\w+\(.*\):", code, re.M):
        return "Python"
    if re.search(r"^\s*#include\b", code, re.M):
        return "C/C++"
    return "알 수 없음"


FUNC_PATTERNS = [
    re.compile(r"^\s*(?:(?:public|private|protected|internal|static|abstract|sealed|partial|final)\s+)*"
               r"(?:class|interface|struct|enum)\s+(\w+)"),                                   # 클래스
    re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\("),                                          # Python
    re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:PROCEDURE|FUNCTION|TRIGGER|PACKAGE(?:\s+BODY)?|VIEW)"
               r"\s+([\w.\"]+)", re.I),                                                        # SQL 객체
    re.compile(r"^\s*(?:PROCEDURE|FUNCTION)\s+(\w+)", re.I),                                   # 패키지 내부
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)"),                            # JavaScript
    re.compile(r"^\s*(?:(?:public|private|protected|internal|static|final|virtual|override|async|"
               r"abstract|synchronized|sealed|extern|unsafe|new)\s+)*"
               r"[\w<>\[\],.?]+\s+(\w+)\s*\([^;]*$"),                                          # C#/Java/C++ 메서드
]
_NOT_FUNC = {"if", "for", "foreach", "while", "switch", "catch", "using", "lock", "return",
             "new", "else", "throw", "sizeof", "typeof", "nameof", "await", "yield"}


def _func_name(line: str) -> str | None:
    for pattern in FUNC_PATTERNS:
        m = pattern.match(line)
        if m:
            name = next((g for g in m.groups() if g), None)
            if name and name.lower() not in _NOT_FUNC:
                return name
    return None


def _enclosing_map(lines: list[str]) -> list[str | None]:
    """각 줄을 감싸는 함수 이름 목록 (위로 올라가며 가장 가까운 함수 선언). 한 번만 계산해 둡니다."""
    result, current = [], None
    for line in lines:
        current = _func_name(line) or current
        result.append(current)
    return result


def outline(code: str) -> list[str]:
    return [f"{i}행 {name}" for i, line in enumerate(code.splitlines(), 1) if (name := _func_name(line))]


# 위험 패턴: (심각도, 정규식, 설명). 오탐이 있을 수 있어 LLM 이 코드로 다시 확인합니다.
ADDED_RULES = [
    ("높음", re.compile(r"catch\s*(?:\([^)]*\))?\s*\{\s*\}"), "빈 catch 블록 - 예외를 삼켜 장애 원인 추적이 어려워짐"),
    ("높음", re.compile(r"^\s*except\b[^:]*:\s*pass\b"), "except: pass - 예외를 무시함"),
    ("높음", re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]+", re.I), "비밀번호 하드코딩 의심"),
    ("중간", re.compile(r"\b(?:Thread\.Sleep|Task\.Delay|time\.sleep|DBMS_LOCK\.SLEEP)\s*\(", re.I),
     "대기(Sleep) 추가 - 처리 지연, 타임아웃 영향"),
    ("중간", re.compile(r"\bSELECT\s+\*", re.I), "SELECT * - 불필요한 컬럼 조회, 테이블 변경 시 영향"),
    ("중간", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), "IP 주소 하드코딩"),
    ("낮음", re.compile(r"\b(?:Console\.Write(?:Line)?|System\.out\.print(?:ln)?|printf)\s*\(|^\s*print\s*\(|"
                       r"DBMS_OUTPUT\.PUT_LINE", re.I), "디버그 출력이 남아 있음"),
    ("낮음", re.compile(r"\b(?:TODO|FIXME|HACK|XXX)\b"), "TODO/FIXME 주석이 남아 있음"),
]
REMOVED_RULES = [
    ("높음", re.compile(r"\bif\b.*(?:hold|status|state|check|valid|lock)", re.I),
     "상태·Hold·검증 조건 삭제 - 막아야 할 처리가 통과될 수 있음"),
    ("높음", re.compile(r"^\s*throw\s*;|^\s*raise\s*$"), "예외 재전달(throw) 삭제 - 실패가 호출한 쪽에 알려지지 않음"),
    ("중간", re.compile(r"\b(?:try|catch|finally|except)\b"), "예외 처리 코드 삭제"),
    ("중간", re.compile(r"[!=]=\s*null\b|\bis\s+(?:not\s+)?None\b|IsNullOrEmpty|\bIS\s+(?:NOT\s+)?NULL\b", re.I),
     "null 체크 삭제 - Null 오류 위험"),
    ("중간", re.compile(r"\block\s*\(|\bsynchronized\b|Monitor\.Enter|\bFOR\s+UPDATE\b", re.I),
     "동기화·잠금(lock / FOR UPDATE) 삭제 - 동시 처리 시 데이터 꼬임 위험"),
    ("낮음", re.compile(r"\b_?log(?:ger)?\s*\.\s*\w+\s*\(", re.I), "로그 삭제 - 이력 추적성 저하"),
]
BOTH_RULES = [
    ("높음", re.compile(r"\b(?:COMMIT|ROLLBACK)\b|\.(?:Commit|Rollback)\s*\(|BeginTransaction|TransactionScope|"
                       r"@Transactional", re.I), "트랜잭션 처리 코드 - 커밋·롤백 범위와 데이터 정합성 확인 필요"),
]
_WHERE_REMOVED = ("높음", re.compile(r"\bWHERE\b", re.I), "WHERE 조건 삭제/변경 - 조회·변경 범위가 넓어질 수 있음")
_DML_RE = re.compile(r"\b(?:UPDATE\s+[\w.]+\s+SET|DELETE\s+FROM\s+[\w.]+)", re.I)
_SEV_ORDER = {"높음": 0, "중간": 1, "낮음": 2}


def _hint(sev: str, where: str, msg: str, line: str) -> tuple[int, str]:
    return _SEV_ORDER[sev], f"[{sev}] {where}: {msg} → `{line.strip()[:80]}`"


def _dml_without_where(lines: list[str], idx: int) -> bool:
    """UPDATE ... SET / DELETE FROM 문이 끝날 때(;)까지 WHERE 가 없는지 확인합니다."""
    m = _DML_RE.search(lines[idx])
    if not m:
        return False
    stmt = " ".join([lines[idx][m.start():]] + lines[idx + 1: idx + 8]).split(";")[0]
    return not re.search(r"\bWHERE\b", stmt, re.I)


def scan_added(lines: list[str], idxs: list[int], where: str = "수정 후") -> list[tuple[int, str]]:
    hints = []
    for i in idxs:
        line = lines[i]
        for sev, rx, msg in ADDED_RULES + BOTH_RULES:
            if rx.search(line):
                hints.append(_hint(sev, f"{where} {i + 1}행", msg, line))
        if _dml_without_where(lines, i):
            hints.append(_hint("높음", f"{where} {i + 1}행", "WHERE 없는 UPDATE/DELETE 의심 - 전체 데이터 변경 위험", line))
    return hints


def scan_removed(lines: list[str], idxs: list[int], added_text: str) -> list[tuple[int, str]]:
    hints = []
    for i in idxs:
        line = lines[i]
        for sev, rx, msg in REMOVED_RULES + BOTH_RULES:
            if rx.search(line):
                hints.append(_hint(sev, f"수정 전 {i + 1}행(삭제)", msg, line))
        # WHERE 줄이 '수정' 된 경우(같은 블록에 WHERE 가 다시 추가됨)는 제외
        if _WHERE_REMOVED[1].search(line) and not _WHERE_REMOVED[1].search(added_text):
            hints.append(_hint(_WHERE_REMOVED[0], f"수정 전 {i + 1}행(삭제)", _WHERE_REMOVED[2], line))
    return hints


def build_diff(before_code: str, after_code: str) -> dict:
    """두 소스를 줄 단위로 비교해서 행 번호가 붙은 diff, 변경 함수, 위험 의심 지점을 만듭니다."""
    a = [line.rstrip() for line in before_code.replace("\r\n", "\n").split("\n")]
    b = [line.rstrip() for line in after_code.replace("\r\n", "\n").split("\n")]
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    enc_a, enc_b = _enclosing_map(a), _enclosing_map(b)

    hunks, total_added, total_removed = [], 0, 0
    for n, group in enumerate(matcher.get_grouped_opcodes(CONTEXT_LINES), 1):
        body, add_idx, del_idx = [], [], []
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                body += [f"  {i1 + k + 1:>5} {j1 + k + 1:>5} | {a[i1 + k]}" for k in range(i2 - i1)]
                continue
            body += [f"- {k + 1:>5} {'':>5} | {a[k]}" for k in range(i1, i2)]
            body += [f"+ {'':>5} {k + 1:>5} | {b[k]}" for k in range(j1, j2)]
            del_idx += range(i1, i2)
            add_idx += range(j1, j2)

        # 추가된 줄은 수정 후 파일 기준, 삭제된 줄은 수정 전 파일 기준으로 함수를 찾습니다.
        funcs = list(dict.fromkeys([enc_b[j] or "(함수 밖)" for j in add_idx]
                                   + [enc_a[i] or "(함수 밖)" for i in del_idx]))
        added_text = "\n".join(b[j] for j in add_idx)
        hints = scan_added(b, add_idx) + scan_removed(a, del_idx, added_text)
        i_from, i_to = group[0][1] + 1, group[-1][2]
        j_from, j_to = group[0][3] + 1, group[-1][4]
        header = f"@@ 변경 {n} · {', '.join(funcs)} · 수정 전 {i_from}~{i_to}행 → 수정 후 {j_from}~{j_to}행"
        hunks.append({"n": n, "funcs": funcs, "text": header + "\n" + "\n".join(body), "hints": hints})
        total_added += len(add_idx)
        total_removed += len(del_idx)

    # 변경이 많으면 위험 의심 지점이 있는 블록을 우선으로 넣고, 원래 순서로 다시 정렬합니다.
    chosen, size = [], 0
    for h in sorted(hunks, key=lambda h: (not h["hints"], h["n"])):
        if size + len(h["text"]) > MAX_DIFF_CHARS and chosen:
            continue
        chosen.append(h)
        size += len(h["text"])
    chosen.sort(key=lambda h: h["n"])

    hints = sorted({h for hunk in hunks for h in hunk["hints"]})
    funcs = list(dict.fromkeys(f for h in hunks for f in h["funcs"]))
    return {
        "diff_text": "\n\n".join(h["text"] for h in chosen),
        "diff_stats": {"hunks": len(hunks), "shown": len(chosen), "added": total_added,
                       "removed": total_removed, "truncated": len(chosen) < len(hunks),
                       "before_lines": len(a), "after_lines": len(b)},
        "changed_funcs": funcs,
        "risk_hints": [text for _, text in hints][:MAX_HINTS],
    }


def numbered(code: str, limit: int = MAX_SOURCE_CHARS) -> str:
    """행 번호를 붙이고, 너무 길면 뒷부분을 자릅니다."""
    lines = code.replace("\r\n", "\n").split("\n")
    out, size = [], 0
    for i, line in enumerate(lines, 1):
        row = f"{i:>5} | {line.rstrip()}"
        if size + len(row) > limit:
            out.append(f"... (소스가 길어 {i}행부터 생략, 전체 {len(lines)}행)")
            break
        out.append(row)
        size += len(row) + 1
    return "\n".join(out)


def _file_line(label: str, src: dict | None) -> str:
    if not src:
        return f"- {label}: (없음)"
    return (f"- {label}: {src['name']} ({src['lang']}, {src['code'].count(chr(10)) + 1}행, "
            f"받은 곳: {src['origin']})")


# ═════════════════════════════════════════════════════════════
# 3) LLM 호출 도우미
# ═════════════════════════════════════════════════════════════

def _ask(system: str, user: str, history: list[dict] | None = None,
         temperature: float = 0.3) -> str:
    """system + 이전 대화 + 이번 발화로 LLM 을 한 번 호출하고 텍스트만 뽑아옵니다.
    이전 대화의 긴 코드는 줄여서 보냅니다. (코드는 노드가 따로 정리해서 넘깁니다)"""
    llm = get_llm(temperature=temperature)
    messages = [{"role": "system", "content": system}]
    for past in (history or [])[-HISTORY_TURNS:]:
        content = past["content"]
        if past["role"] == "user":
            content = compact_text(content)
        messages.append({"role": past["role"], "content": content})
    messages.append({"role": "user", "content": user})
    result = llm.invoke(messages)
    return (result.content or "").strip()


def _request_text(state: SourceState) -> str:
    focus = state.get("focus")
    return (f"[사용자 요청]\n{compact_text(state['user_input'])}"
            + (f"\n\n[중점 관점] {focus}" if focus else ""))


# ═════════════════════════════════════════════════════════════
# 4) 노드 7개
# ═════════════════════════════════════════════════════════════

# -- 노드 1: 요청 분석 (LLM) ----------------------------------------
# 이어지는 질문인지, 비교인지 설명인지, 파일명과 중점 관점을 뽑습니다.
def analyze_request(state: SourceState) -> dict:
    text = state["user_input"]
    has_code = bool(_sources_from_text(text))
    names = _mentioned_names(text)
    info = []
    if has_code:
        info.append(f"코드 {len(_sources_from_text(text))}개")
    if names:
        info.append(f"파일명 {', '.join(names)}")
    user = f"[이번 메시지에 포함된 것: {' / '.join(info) or '코드·파일명 없음'}]\n{compact_text(text)}"
    raw = _ask(ANALYZE_REQUEST_SYSTEM, user, history=state["messages"], temperature=0.0)

    # LLM 이 JSON 앞뒤에 설명 문장을 붙일 수 있으니 방어적으로 파싱합니다.
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        parsed = {}

    task = parsed.get("task")
    if task not in ("compare", "explain"):   # 분석 실패 시 키워드로 판단
        task = "compare" if re.search(r"비교|차이|바뀐|변경|수정|diff|리뷰", text, re.I) else "explain"
    # 새 코드나 파일명을 줬으면 이어지는 질문이 아니라 새 요청입니다.
    intent = "followup" if parsed.get("intent") == "followup" and not has_code and not names else "new"
    llm_names = [str(n) for n in (parsed.get("file_names") or []) if n and "." in str(n)]
    return {
        "intent": intent,
        "task": task,
        "file_names": list(dict.fromkeys(names + llm_names)),
        "focus": parsed.get("focus") or None,
    }


# -- 노드 2: 소스 받기 (LLM 없음) ------------------------------------
def load_sources(state: SourceState) -> dict:
    task, notes = state["task"], []
    need = 2 if task == "compare" else 1

    current = _from_attachments(state) + _collect_turn(
        state["user_input"], state.get("file_names") or [], task, notes)

    if state["intent"] == "followup" and not current:
        # 이어지는 질문이면 앞 대화에서 분석했던 소스를 그대로 다시 씁니다.
        sources = _from_history(state["messages"], task, need)
    elif len(current) < need:
        # 이번에 1개만 줬으면 이전 메시지에서 나머지를 찾습니다. (이전 = 수정 전)
        sources = _from_history(state["messages"], task, need - len(current)) + current
    else:
        sources = current
    if len(sources) > 2:
        notes.append(f"소스가 {len(sources)}개라 2개만 비교함")

    for src in sources:
        src["lang"] = detect_lang(src.get("name"), src["code"], src.get("hint", ""))

    if len(sources) >= 2:
        before, after = _assign(sources)
        before["name"] = before.get("name") or "수정 전 코드"
        after["name"] = after.get("name") or "수정 후 코드"
        status = "pair"
    elif sources:
        before, after = None, sources[0]
        after["name"] = after.get("name") or "붙여넣은 코드"
        status = "single"
    else:
        before = after = None
        status = "none"
    return {"before": before, "after": after, "source_status": status, "source_notes": notes}


# --- 분기: 어느 노드로 보낼까? ----------------------------------
def route_after_load(state: SourceState) -> str:
    # 첫 턴에는 이어질 대화가 없으므로 followup 으로 새지 않게 messages 도 함께 확인합니다.
    if state["intent"] == "followup" and state["messages"]:
        return "answer_follow_up"
    if state["source_status"] == "pair":
        return "compute_diff"
    if state["source_status"] == "single" and state["task"] == "explain":
        return "explain_source"
    return "request_files"


# -- 노드 3: diff 계산 (LLM 없음) ------------------------------------
def compute_diff(state: SourceState) -> dict:
    return build_diff(state["before"]["code"], state["after"]["code"])


# -- 노드 4: 변경 리뷰 (LLM) ----------------------------------------
def review_changes(state: SourceState) -> dict:
    stats = state["diff_stats"]
    if stats["hunks"] == 0:
        return {"answer": f"**{state['before']['name']}** 와 **{state['after']['name']}** 는 내용이 같습니다. "
                          "(줄 끝 공백 차이는 무시했습니다) 수정 전/후 파일이 바뀌지 않았는지 확인해 주세요."}

    truncated = f" (일부 생략: {stats['hunks']}개 중 {stats['shown']}개만 표시)" if stats["truncated"] else ""
    user = (
        f"[파일 정보]\n{_file_line('수정 전', state['before'])}\n{_file_line('수정 후', state['after'])}\n\n"
        f"[변경 통계] 변경 블록 {stats['hunks']}개 / 추가 {stats['added']}줄 / 삭제 {stats['removed']}줄{truncated}\n\n"
        f"[변경된 함수] {', '.join(state['changed_funcs'])}\n\n"
        f"[자동 점검 결과] (패턴 기반, 오탐 가능)\n" + ("\n".join(state["risk_hints"]) or "(없음)") + "\n\n"
        f"[변경 내용]\n{state['diff_text']}\n\n"
        f"{_request_text(state)}"
    )
    return {"answer": _ask(REVIEW_CHANGES_SYSTEM, user, history=state["messages"], temperature=0.3)}


# -- 노드 5: 소스 설명 (LLM) ----------------------------------------
def explain_source(state: SourceState) -> dict:
    src = state["after"]
    lines = src["code"].replace("\r\n", "\n").split("\n")
    hints = [text for _, text in sorted(set(scan_added(lines, list(range(len(lines))), where="소스")))]
    user = (
        f"[파일 정보]\n{_file_line('소스', src)}\n\n"
        f"[구조]\n" + ("\n".join(outline(src["code"])) or "(자동으로 찾은 함수 없음)") + "\n\n"
        f"[자동 점검 결과] (패턴 기반, 오탐 가능)\n" + ("\n".join(hints[:MAX_HINTS]) or "(없음)") + "\n\n"
        f"[소스]\n{numbered(src['code'])}\n\n"
        f"{_request_text(state)}"
    )
    return {"answer": _ask(EXPLAIN_SOURCE_SYSTEM, user, history=state["messages"], temperature=0.3)}


# -- 노드 6: 파일 요청 (LLM) ----------------------------------------
def request_files(state: SourceState) -> dict:
    got = [s for s in (state.get("before"), state.get("after")) if s]
    files = inbox_files()
    user = (
        f"[요청 종류] {'수정 전/후 비교' if state['task'] == 'compare' else '소스 설명'}\n"
        f"[받은 파일] " + (", ".join(f"{s['name']}({s['origin']})" for s in got) or "없음") + "\n"
        f"[확인된 문제] " + ("; ".join(state["source_notes"]) or "없음") + "\n"
        f"[inbox 파일 목록] " + (", ".join(files[:20]) or "비어 있음")
    )
    return {"answer": _ask(REQUEST_FILES_SYSTEM, user, history=state["messages"], temperature=0.3)}


# -- 노드 7: 이어지는 질문 (LLM) ------------------------------------
def answer_follow_up(state: SourceState) -> dict:
    status = state["source_status"]
    if status == "pair":
        d = build_diff(state["before"]["code"], state["after"]["code"])
        ref = (f"{_file_line('수정 전', state['before'])}\n{_file_line('수정 후', state['after'])}\n"
               f"[변경 내용]\n{d['diff_text'][: MAX_DIFF_CHARS // 2]}")
    elif status == "single":
        ref = f"{_file_line('소스', state['after'])}\n[소스]\n{numbered(state['after']['code'], MAX_SOURCE_CHARS // 2)}"
    else:
        ref = "(이전 대화에서 소스를 찾지 못함 - 대화 내용만 참고)"
    user = f"[참고 자료]\n{ref}\n\n[추가 질문]\n{compact_text(state['user_input'])}"
    return {"answer": _ask(FOLLOW_UP_SYSTEM, user, history=state["messages"], temperature=0.4)}


# ═════════════════════════════════════════════════════════════
# 5) 그래프 조립
# ═════════════════════════════════════════════════════════════

def build_graph():
    """노드와 엣지를 연결해 그래프를 완성합니다."""
    builder = StateGraph(SourceState)

    builder.add_node("analyze_request", analyze_request)
    builder.add_node("load_sources", load_sources)
    builder.add_node("compute_diff", compute_diff)
    builder.add_node("review_changes", review_changes)
    builder.add_node("explain_source", explain_source)
    builder.add_node("request_files", request_files)
    builder.add_node("answer_follow_up", answer_follow_up)

    builder.add_edge(START, "analyze_request")
    builder.add_edge("analyze_request", "load_sources")
    builder.add_conditional_edges(
        "load_sources",
        route_after_load,
        {
            "answer_follow_up": "answer_follow_up",
            "compute_diff": "compute_diff",
            "explain_source": "explain_source",
            "request_files": "request_files",
        },
    )
    builder.add_edge("compute_diff", "review_changes")
    for name in ("review_changes", "explain_source", "request_files", "answer_follow_up"):
        builder.add_edge(name, END)

    return builder.compile()


# --- 터미널에서 바로 실행하기 ----
#
#   python teams/team3/workflow.py "LotTrackIn.cs 수정 전후 비교해줘"
#   python teams/team3/workflow.py --before 수정전.cs --after 수정후.cs "예외 처리 위주로 봐줘"
#   python teams/team3/workflow.py --inbox        ← inbox 파일 목록만 확인 (LLM 호출 없음)
#
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3조 MES 소스 변경 분석")
    parser.add_argument("message", nargs="*", help="질문")
    parser.add_argument("--before", help="수정 전 파일 경로")
    parser.add_argument("--after", help="수정 후 파일 경로")
    parser.add_argument("--inbox", action="store_true", help="inbox 파일 목록 보기")
    args = parser.parse_args()

    if args.inbox:
        print(f"inbox 폴더: {INBOX_DIR}")
        print("\n".join(f"- {f}" for f in inbox_files()) or "(비어 있음)")
        raise SystemExit(0)

    message = " ".join(args.message) or TEAM_INFO["examples"][0]
    # 로컬 파일을 넘기면 채팅에 코드 블록으로 붙여넣은 것과 똑같은 메시지를 만들어 실행합니다.
    for label, path in (("수정 전", args.before), ("수정 후", args.after)):
        if path:
            p = Path(path)
            message += f"\n\n{label} {p.name}:\n```\n{_read_text(p)}\n```"

    from app.contract import make_initial_state
    from app.llm import LLMConfigError

    print(f"입력: {compact_text(message)}\n")
    answer = ""
    try:
        graph = build_graph()
        for chunk in graph.stream(make_initial_state(message), stream_mode="updates"):
            for node, update in chunk.items():
                print(f"---- {node}")
                for key, value in (update or {}).items():
                    text = str(value).replace("\n", " ")
                    print(f"    {key} = {text[:160]}{'...' if len(text) > 160 else ''}")
                if isinstance(update, dict) and update.get("answer"):
                    answer = update["answer"]
                print()
    except (NotImplementedError, LLMConfigError) as exc:
        print(f"[안내] {exc}")
        raise SystemExit(1)

    print("=" * 60)
    print(answer)
