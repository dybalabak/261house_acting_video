"""3조 - MES 화면 안내 에이전트 (MultiAgent)

MES 화면명을 물어보면 사내에 등록된 화면 설명과 모델이 가진 반도체 공정·MES 지식을
섞어서 "어떤 화면인지, 언제 왜 쓰는지" 를 설명해 주고, 공정·용어 질문에도 답합니다.

그래프 구조 (노드 6개)
    analyze_query ──→ search_screens ──┬─ (이어지는 질문) ────────→ answer_follow_up → END
                                       ├─ (화면 질문 · 1개 특정) ──→ explain_screen ───→ END
                                       ├─ (화면 질문 · 못 찾음/여러 개) → clarify_screen → END
                                       └─ (공정·용어 질문) ────────→ explain_concept ──→ END

핵심 설계: 화면 '검색'은 LLM 이 아니라 파이썬(search_screens)이 합니다.
LLM 에게 화면 목록 전체를 주고 고르게 하면 비슷한 이름을 헷갈리거나 없는 화면을 지어냅니다.
그래서 화면명 매칭은 코드가 정확히 하고, LLM 은 찾은 정보를 '설명'하는 일만 맡습니다.

터미널에서 이 파일만 바로 실행해 볼 수 있습니다.

    python teams/team3/workflow.py "Hold Lot 관리 화면이 뭐야?"
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
    ANALYZE_QUERY_SYSTEM,
    CLARIFY_SCREEN_SYSTEM,
    EXPLAIN_CONCEPT_SYSTEM,
    EXPLAIN_SCREEN_SYSTEM,
    FOLLOW_UP_SYSTEM,
    MES_SCREENS,
)

TEAM_INFO = {
    "name": "3조 MES 화면 안내 선배봇",
    "description": "MES 화면명을 물어보면 어떤 화면인지, 언제 왜 쓰는지 공정 흐름과 함께 설명해 드립니다. 반도체 공정·MES 용어 질문도 받습니다.",
    # ※ 실제 MES_SCREENS 데이터를 넣은 뒤, 그 화면 이름으로 예시를 바꿔주세요.
    "examples": [
        "Hold Lot 관리 화면이 뭐야?",
        "OOC가 뭐고 MES에서 어디서 확인해?",
        "Lot 화면 뭐 있어?",
    ],
}

HISTORY_TURNS = 6       # 프롬프트에 함께 보낼 이전 대화 개수
MATCH_THRESHOLD = 0.6   # 이 점수 이상이어야 '찾았다' 로 봅니다 (0~1)
AMBIGUOUS_GAP = 0.1     # 1등과 2등 점수 차이가 이보다 작으면 '여러 개' 로 보고 되묻습니다
RELATED_LIMIT = 3       # 함께 보여줄 관련 화면 최대 개수


class MesState(BaseGraphState):
    """3조가 사용하는 상태입니다. BaseGraphState 를 상속해 필드를 더했습니다."""

    intent: str               # "new"(새 질문) / "followup"(방금 답변에 이어지는 질문)
    query_type: str           # "screen"(화면 질문) / "concept"(공정·용어·화면 찾기 질문)
    screen_name: str | None   # 사용자가 말한 화면 이름
    keywords: list[str]       # 질문 핵심 용어 (관련 화면 검색에 사용)
    match_status: str         # "found" / "ambiguous" / "not_found" / "none"(화면명 없음)
    matched: dict | None      # 특정된 화면 {"name", "desc", "aliases"}
    candidates: list[dict]    # 되물을 때 보여줄 후보 화면
    related: list[dict]       # 함께 보면 좋은 관련 화면


# ═════════════════════════════════════════════════════════════
# 1) 사내 화면 데이터 불러오기 & 검색 (LLM 없이 순수 파이썬)
# ═════════════════════════════════════════════════════════════

def load_screens(raw: str = MES_SCREENS) -> list[dict]:
    """prompts.py 의 MES_SCREENS 텍스트를 [{"name", "desc", "aliases"}] 로 바꿉니다.
    '|' 구분과 엑셀에서 복사한 탭 구분을 모두 받습니다."""
    screens = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "|" if "|" in line else "\t"
        parts = [p.strip() for p in line.split(sep)]
        if len(parts) < 2 or not parts[0]:
            continue
        aliases = [a.strip() for a in parts[2].split(",") if a.strip()] if len(parts) > 2 else []
        screens.append({"name": parts[0], "desc": parts[1], "aliases": aliases})
    return screens


SCREENS = load_screens()


def _norm(text: str) -> str:
    """비교용 정규화: 소문자, 공백·기호 제거, 끝의 '화면' 제거."""
    text = re.sub(r"[\s_\-·/()\[\]]", "", str(text).lower())
    return re.sub(r"(화면|메뉴)$", "", text)


def _name_score(query: str, screen: dict) -> float:
    """화면명(+별칭)과 사용자가 말한 이름이 얼마나 비슷한지 0~1 로 계산합니다."""
    q = _norm(query)
    if not q:
        return 0.0
    best = 0.0
    for name in [screen["name"], *screen["aliases"]]:
        n = _norm(name)
        if not n:
            continue
        if q == n:
            return 1.0
        if q in n or n in q:   # 부분 일치 ("Hold Lot" ↔ "Hold Lot 관리")
            best = max(best, 0.7 + 0.3 * min(len(q), len(n)) / max(len(q), len(n)))
        best = max(best, SequenceMatcher(None, q, n).ratio())
    return best


def find_screen(query: str | None) -> tuple[str, dict | None, list[dict]]:
    """화면명으로 사내 화면을 찾습니다. → (상태, 특정된 화면, 후보 목록)"""
    if not query:
        return "none", None, []
    ranked = sorted(((_name_score(query, s), s) for s in SCREENS), key=lambda x: -x[0])
    good = [(sc, s) for sc, s in ranked if sc >= MATCH_THRESHOLD]

    if good and good[0][0] == 1.0:               # 정확히 일치하면 바로 확정
        return "found", good[0][1], []
    if len(good) == 1 or (good and good[0][0] - good[1][0] >= AMBIGUOUS_GAP):
        return "found", good[0][1], []
    if len(good) >= 2:                            # 비슷한 점수가 여러 개 → 되묻기
        close = [s for sc, s in good if good[0][0] - sc < AMBIGUOUS_GAP]
        return "ambiguous", None, close[:5]
    # 못 찾았으면 그나마 비슷한 화면을 참고 후보로 넘깁니다.
    return "not_found", None, [s for sc, s in ranked[:3] if sc >= 0.35]


def related_screens(keywords: list[str], exclude: str | None = None) -> list[dict]:
    """키워드가 화면명·설명·별칭에 몇 개 들어 있는지로 관련 화면을 고릅니다."""
    words = [_norm(k) for k in keywords if len(_norm(k)) >= 2]
    if not words:
        return []
    scored = []
    for s in SCREENS:
        if s["name"] == exclude:
            continue
        text = _norm(" ".join([s["name"], s["desc"], *s["aliases"]]))
        hit = sum(1 for w in words if w in text)
        if hit:
            scored.append((hit, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:RELATED_LIMIT]]


def _screen_block(screens: list[dict]) -> str:
    """LLM 에 넘길 화면 정보 텍스트."""
    if not screens:
        return "(없음)"
    lines = []
    for s in screens:
        alias = f" (별칭: {', '.join(s['aliases'])})" if s["aliases"] else ""
        lines.append(f"- {s['name']}{alias}: {s['desc']}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
# 2) LLM 호출 도우미
# ═════════════════════════════════════════════════════════════

def _ask(system: str, user: str, history: list[dict] | None = None,
         temperature: float = 0.3) -> str:
    """system + 이전 대화 + 이번 발화로 LLM 을 한 번 호출하고 텍스트만 뽑아옵니다."""
    llm = get_llm(temperature=temperature)
    messages = [{"role": "system", "content": system}]
    for past in (history or [])[-HISTORY_TURNS:]:
        messages.append({"role": past["role"], "content": past["content"]})
    messages.append({"role": "user", "content": user})
    result = llm.invoke(messages)
    return (result.content or "").strip()


# ═════════════════════════════════════════════════════════════
# 3) 노드 6개
# ═════════════════════════════════════════════════════════════

# -- 노드 1: 질문 분석 (LLM) ---------------------------------------
# 이어지는 질문인지, 화면 질문인지 개념 질문인지, 화면 이름과 키워드를 뽑습니다.
def analyze_query(state: MesState) -> dict:
    raw = _ask(ANALYZE_QUERY_SYSTEM, state["user_input"], history=state["messages"], temperature=0.0)

    # LLM 이 JSON 앞뒤에 설명 문장을 붙일 수 있으니 방어적으로 파싱합니다.
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        parsed = {}

    keywords = [str(k) for k in (parsed.get("keywords") or []) if k]
    return {
        "intent": "followup" if parsed.get("intent") == "followup" else "new",
        "query_type": "screen" if parsed.get("type") == "screen" else "concept",
        # 분석이 실패해도 원문으로 검색은 해보도록 user_input 을 대신 씁니다.
        "screen_name": parsed.get("screen_name") or None,
        "keywords": keywords or [state["user_input"]],
    }


# -- 노드 2: 사내 화면 검색 (LLM 없음) -------------------------------
def search_screens(state: MesState) -> dict:
    status, matched, candidates = find_screen(state.get("screen_name"))

    # 개념 질문인데 문장 안에 등록된 화면 이름이 그대로 들어있으면 화면 질문으로 봅니다.
    if status == "none":
        q = _norm(state["user_input"])
        hits = [s for s in SCREENS if _norm(s["name"]) in q]
        if len(hits) == 1:
            status, matched = "found", hits[0]

    exclude = matched["name"] if matched else None
    return {
        "match_status": status,
        "matched": matched,
        "candidates": candidates,
        "related": related_screens(state["keywords"], exclude=exclude),
    }


# --- 분기: 어느 답변 노드로 보낼까? ------------------------------
def route_after_search(state: MesState) -> str:
    # 첫 턴에는 이어질 대화가 없으므로 followup 으로 새지 않게 messages 도 함께 확인합니다.
    if state["intent"] == "followup" and state["messages"]:
        return "answer_follow_up"
    if state["match_status"] == "found":
        return "explain_screen"
    if state["query_type"] == "screen" and state["match_status"] in ("ambiguous", "not_found"):
        return "clarify_screen"
    return "explain_concept"


# -- 노드 3: 화면 되묻기 (LLM) --------------------------------------
def clarify_screen(state: MesState) -> dict:
    label = "후보 화면" if state["match_status"] == "ambiguous" else "참고 후보"
    user = (
        f"[사용자가 말한 화면명] {state.get('screen_name')}\n"
        f"[검색 상태] {state['match_status']}\n"
        f"[{label}]\n{_screen_block(state['candidates'])}"
    )
    return {"answer": _ask(CLARIFY_SCREEN_SYSTEM, user, history=state["messages"], temperature=0.3)}


# -- 노드 4: 화면 설명 (LLM) ----------------------------------------
def explain_screen(state: MesState) -> dict:
    user = (
        f"[사내 MES 화면 정보]\n{_screen_block([state['matched']])}\n\n"
        f"[관련 화면]\n{_screen_block(state['related'])}\n\n"
        f"[사용자 질문]\n{state['user_input']}"
    )
    return {"answer": _ask(EXPLAIN_SCREEN_SYSTEM, user, history=state["messages"], temperature=0.4)}


# -- 노드 5: 공정·개념 설명 (LLM) -----------------------------------
def explain_concept(state: MesState) -> dict:
    user = (
        f"[관련 화면]\n{_screen_block(state['related'])}\n\n"
        f"[사용자 질문]\n{state['user_input']}"
    )
    return {"answer": _ask(EXPLAIN_CONCEPT_SYSTEM, user, history=state["messages"], temperature=0.4)}


# -- 노드 6: 이어지는 질문 (LLM) ------------------------------------
def answer_follow_up(state: MesState) -> dict:
    screens = [state["matched"]] if state.get("matched") else []
    user = (
        f"[사내 MES 화면 정보]\n{_screen_block(screens + state['related'])}\n\n"
        f"[추가 질문]\n{state['user_input']}"
    )
    return {"answer": _ask(FOLLOW_UP_SYSTEM, user, history=state["messages"], temperature=0.4)}


# ═════════════════════════════════════════════════════════════
# 4) 그래프 조립
# ═════════════════════════════════════════════════════════════

ANSWER_NODES = {
    "answer_follow_up": answer_follow_up,
    "clarify_screen": clarify_screen,
    "explain_screen": explain_screen,
    "explain_concept": explain_concept,
}


def build_graph():
    """노드와 엣지를 연결해 그래프를 완성합니다."""
    builder = StateGraph(MesState)

    builder.add_node("analyze_query", analyze_query)
    builder.add_node("search_screens", search_screens)
    for name, fn in ANSWER_NODES.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", "search_screens")
    builder.add_conditional_edges(
        "search_screens",
        route_after_search,
        {name: name for name in ANSWER_NODES},
    )
    for name in ANSWER_NODES:
        builder.add_edge(name, END)

    return builder.compile()


# --- 터미널에서 바로 실행하기 ----
#
#   python teams/team3/workflow.py "Hold Lot 관리 화면이 뭐야?"
#   python teams/team3/workflow.py --screens     ← 화면 데이터가 제대로 읽혔는지만 확인
#
if __name__ == "__main__":
    if sys.argv[1:] == ["--screens"]:
        print(f"등록된 화면 {len(SCREENS)}개")
        print(_screen_block(SCREENS))
        raise SystemExit(0)

    from app.contract import make_initial_state
    from app.llm import LLMConfigError

    message = " ".join(sys.argv[1:]) or TEAM_INFO["examples"][0]
    print(f"입력: {message}\n")

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
