"""3조 - 바이오리듬 MultiAgent

생년월일로 바이오리듬을 계산하고, 사용자가 원하는 것(리듬 해설 / 궁합 / 운세 / 조언 / 운동)에 맞는
전문 노드 하나로 보내 답변을 만듭니다.

그래프 구조
    analyze_intent ── (이어지는 질문) ──→ answer_follow_up ──→ END
                   ── (생년월일 부족) ──→ ask_clarify ──────→ END
                   ── (정보 충분) ──→ calc_biorhythm ──┬→ explain_biorhythm → END
                                                       ├→ compatibility ────→ END
                                                       ├→ fortune ──────────→ END
                                                       ├→ advice ───────────→ END
                                                       └→ exercise ─────────→ END

핵심 설계: 바이오리듬 '계산'은 LLM 이 아니라 파이썬(calc_biorhythm)이 합니다.
LLM 은 사인 함수 계산을 자주 틀리기 때문에, 숫자는 코드가 정확히 만들고
LLM 은 그 숫자를 '해석'하는 일만 맡습니다.

터미널에서 이 파일만 바로 실행해 볼 수 있습니다.

    python teams/team3/workflow.py "1995년 3월 15일생인데 오늘 바이오리듬 어때?"
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# 이 파일을 직접 실행할 때도 app/ 을 찾을 수 있도록 최상위 폴더를 경로에 추가합니다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langgraph.graph import END, START, StateGraph

from app.contract import BaseGraphState
from app.llm import get_llm
from teams.team3.prompts import (
    ADVICE_SYSTEM,
    ANALYZE_INTENT_SYSTEM,
    ASK_CLARIFY_SYSTEM,
    COMPATIBILITY_SYSTEM,
    EXERCISE_SYSTEM,
    EXPLAIN_SYSTEM,
    FOLLOW_UP_SYSTEM,
    FORTUNE_SYSTEM,
)

TEAM_INFO = {
    "name": "3조 바이오리듬 컨디션 코치",
    "description": "생년월일을 알려주면 바이오리듬을 계산해서 궁합, 오늘의 운세, 오늘의 조언, 운동까지 추천해 드립니다.",
    "examples": [
        "1995년 3월 15일생인데 오늘 내 바이오리듬 어때?",
        "나는 1993-07-02, 여자친구는 1995-11-20인데 바이오리듬 궁합 봐줘",
        "92년 5월 8일생이야. 퇴근하고 무슨 운동 하면 좋을까?",
        "88년 12월 1일생, 내일 운세 알려줘",
        "90년 6월 30일생인데 오늘 하루 조언 좀 해줘",
    ],
}

# 바이오리듬 3대 주기 (키: (한글 이름, 주기 일수))
CYCLES = {
    "physical": ("신체", 23),
    "emotional": ("감성", 28),
    "intellectual": ("지성", 33),
}

# task 값 → 실행할 해설 노드 이름
TASK_NODES = {
    "biorhythm": "explain_biorhythm",
    "compatibility": "compatibility",
    "fortune": "fortune",
    "advice": "advice",
    "exercise": "exercise",
}
TASK_LABELS = {
    "biorhythm": "바이오리듬 확인",
    "compatibility": "바이오리듬 궁합",
    "fortune": "오늘의 운세",
    "advice": "오늘의 조언",
    "exercise": "운동 추천",
}
MISSING_LABELS = {
    "birth_date": "본인 생년월일",
    "partner_birth_date": "궁합 상대의 생년월일",
}

WEEKDAYS = "월화수목금토일"
FORECAST_DAYS = 7   # 앞으로 며칠 흐름을 함께 계산할지
HISTORY_TURNS = 6   # 프롬프트에 함께 보낼 이전 대화 개수


class BioState(BaseGraphState):
    """3조가 사용하는 상태입니다. BaseGraphState 를 상속해 필드를 더했습니다."""

    intent: str                     # "new"(새 요청) 또는 "followup"(방금 답변에 이어지는 질문)
    task: str                       # biorhythm / compatibility / fortune / advice / exercise
    birth_date: str | None          # 본인 생년월일 "YYYY-MM-DD"
    partner_birth_date: str | None  # 궁합 상대 생년월일 "YYYY-MM-DD"
    target_date: str                # 보고 싶은 날짜 "YYYY-MM-DD" (기본: 오늘)
    missing: list[str]              # 아직 모르는 필수 항목
    bio_report: str                 # calc_biorhythm 이 만든 계산 결과 텍스트


# ═════════════════════════════════════════════════════════════
# 1) 바이오리듬 계산 (LLM 없이 순수 파이썬)
# ═════════════════════════════════════════════════════════════

def _parse_date(value) -> date | None:
    """LLM 이 준 날짜 문자열을 date 로 바꿉니다. 형식이 틀리거나 없는 날짜면 None."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value).strip(), fmt).date()
        except ValueError:
            continue
    return None


def _level(value: int) -> str:
    if value >= 50:
        return "높음"
    if value > 0:
        return "보통 이상"
    if value == 0:
        return "중간"
    if value > -50:
        return "보통 이하"
    return "낮음"


def _stars(total: int) -> str:
    """종합 지수(-100~100)를 별점 1~5개로 바꿉니다."""
    count = 5 if total >= 50 else 4 if total >= 20 else 3 if total > -20 else 2 if total > -50 else 1
    return "★" * count + "☆" * (5 - count)


def bio_values(birth: date, target: date) -> dict:
    """target 날짜의 신체/감성/지성 리듬을 계산합니다.

    value    : sin(2π × 살아온 날 / 주기) × 100  (-100 ~ +100)
    trend    : 내일 값과 비교한 방향
    critical : 그날 안에 0 을 지나가면(리듬이 +/- 로 바뀌면) '위험일'
    """
    days = (target - birth).days
    result = {}
    for key, (label, period) in CYCLES.items():
        w = 2 * math.pi / period
        now = math.sin(w * days)
        nxt = math.sin(w * (days + 1))
        value = round(now * 100)
        result[key] = {
            "label": label,
            "period": period,
            "value": value,
            "level": _level(value),
            "trend": "상승" if nxt > now else "하강",
            "critical": math.sin(w * (days - 0.5)) * math.sin(w * (days + 0.5)) <= 0,
        }
    return result


def composite(values: dict) -> int:
    """세 리듬의 평균 = 종합 지수."""
    return round(sum(v["value"] for v in values.values()) / len(values))


def compatibility_scores(a: date, b: date) -> dict:
    """두 사람의 궁합 점수 (0~100).

    두 사람 생일의 차이가 주기의 배수에 가까울수록 리듬이 같이 움직이므로 점수가 높습니다.
    점수 = (1 + cos(2π × 생일 차이 / 주기)) / 2 × 100
    """
    diff = abs((a - b).days)
    scores = {
        label: round((1 + math.cos(2 * math.pi * diff / period)) / 2 * 100)
        for label, period in CYCLES.values()
    }
    scores["종합"] = round(sum(scores.values()) / len(scores))
    return scores


def _person_lines(name: str, birth: date, target: date) -> list[str]:
    values = bio_values(birth, target)
    total = composite(values)
    lines = [f"[{name}] 생년월일 {birth.isoformat()} / 태어난 지 {(target - birth).days:,}일째"]
    for v in values.values():
        flag = " [위험일]" if v["critical"] else ""
        lines.append(
            f"- {v['label']}({v['period']}일 주기): {v['value']:+d} · {v['level']} · 내일 {v['trend']}{flag}"
        )
    lines.append(f"- 종합 지수: {total:+d} · 별점 {_stars(total)}")
    return lines


def _forecast_lines(birth: date, target: date) -> list[str]:
    rows = []
    for i in range(FORECAST_DAYS):
        day = target + timedelta(days=i)
        values = bio_values(birth, day)
        rows.append((day, values, composite(values)))

    lines = [f"[향후 {FORECAST_DAYS}일 흐름] 날짜: 신체/감성/지성 → 종합"]
    for day, v, total in rows:
        lines.append(
            f"- {day:%m-%d}({WEEKDAYS[day.weekday()]}): "
            f"{v['physical']['value']:+d}/{v['emotional']['value']:+d}/{v['intellectual']['value']:+d} → {total:+d}"
        )
    best = max(rows, key=lambda r: r[2])
    worst = min(rows, key=lambda r: r[2])
    lines.append(f"- 가장 좋은 날: {best[0]:%m-%d}({WEEKDAYS[best[0].weekday()]}) {best[2]:+d}")
    lines.append(f"- 가장 주의할 날: {worst[0]:%m-%d}({WEEKDAYS[worst[0].weekday()]}) {worst[2]:+d}")
    return lines


def build_report(state: dict) -> str:
    """state 의 날짜 정보로 LLM 에 넘길 [계산 결과] 텍스트를 만듭니다."""
    birth = _parse_date(state.get("birth_date"))
    if not birth:
        return ""
    target = _parse_date(state.get("target_date")) or date.today()

    lines = [f"기준일: {target.isoformat()} ({WEEKDAYS[target.weekday()]}요일)", ""]
    lines += _person_lines("본인", birth, target)

    partner = _parse_date(state.get("partner_birth_date"))
    if state.get("task") == "compatibility" and partner:
        lines += [""] + _person_lines("상대방", partner, target)
        lines += ["", "[궁합 점수] 태어난 날의 차이로 정해지는 고정값 (0~100점)"]
        lines += [f"- {label}: {score}점" for label, score in compatibility_scores(birth, partner).items()]
    else:
        lines += [""] + _forecast_lines(birth, target)
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


def _expert(system: str, state: BioState, temperature: float) -> dict:
    """해설 노드 5개가 공통으로 쓰는 호출. 계산 결과 + 사용자 요청을 함께 넘깁니다."""
    user = f"[계산 결과]\n{state['bio_report']}\n\n[사용자 요청]\n{state['user_input']}"
    return {"answer": _ask(system, user, history=state["messages"], temperature=temperature)}


# ═════════════════════════════════════════════════════════════
# 3) 노드
# ═════════════════════════════════════════════════════════════

# -- 노드 1: 의도 분석 (LLM) ---------------------------------------
# 무엇을 원하는지(task), 이어지는 질문인지(intent), 날짜 정보를 뽑습니다.
def analyze_intent(state: BioState) -> dict:
    today = date.today()
    # "내일", "이번 주 금요일" 을 계산할 수 있도록 오늘 날짜를 함께 넘깁니다.
    user = f"[오늘 날짜: {today.isoformat()} ({WEEKDAYS[today.weekday()]}요일)]\n{state['user_input']}"
    raw = _ask(ANALYZE_INTENT_SYSTEM, user, history=state["messages"], temperature=0.0)

    # LLM 이 JSON 앞뒤에 설명 문장을 붙일 수 있으니 방어적으로 파싱합니다.
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        parsed = {}

    intent = "followup" if parsed.get("intent") == "followup" else "new"
    task = parsed.get("task") if parsed.get("task") in TASK_NODES else "biorhythm"
    birth = _parse_date(parsed.get("birth_date"))
    partner = _parse_date(parsed.get("partner_birth_date"))
    target = _parse_date(parsed.get("target_date")) or today

    # LLM 이 준 날짜를 코드로 한 번 더 검증합니다. (없는 날짜, 미래 생일 → 다시 묻기)
    missing = []
    if not birth or birth > target:
        missing.append("birth_date")
    if task == "compatibility" and (not partner or partner > target):
        missing.append("partner_birth_date")

    return {
        "intent": intent,
        "task": task,
        "birth_date": birth.isoformat() if birth else None,
        "partner_birth_date": partner.isoformat() if partner else None,
        "target_date": target.isoformat(),
        "missing": missing,
    }


# --- 분기 1: 이어지는 질문인가? 정보가 충분한가? ----------
def route_after_intent(state: BioState) -> str:
    # 첫 턴에는 이어질 대화가 없으므로 followup 으로 새지 않게 messages 도 함께 확인합니다.
    if state["intent"] == "followup" and state["messages"]:
        return "answer_follow_up"
    return "ask_clarify" if state["missing"] else "calc_biorhythm"


# -- 노드 2: 이어지는 질문에 답하기 (LLM) ---------------------------
def answer_follow_up(state: BioState) -> dict:
    report = build_report(state)
    user = state["user_input"]
    if report:
        user = f"[참고용 계산 결과]\n{report}\n\n[추가 질문]\n{user}"
    return {"answer": _ask(FOLLOW_UP_SYSTEM, user, history=state["messages"], temperature=0.4)}


# -- 노드 3: 되묻기 (LLM) ------------------------------------------
def ask_clarify(state: BioState) -> dict:
    user = (
        f"사용자가 원하는 것: {TASK_LABELS[state['task']]}\n"
        f"지금까지 파악한 정보: 본인 생년월일={state.get('birth_date') or '모름'}, "
        f"상대 생년월일={state.get('partner_birth_date') or '모름'}\n"
        f"아직 모르는 항목: {', '.join(MISSING_LABELS[m] for m in state['missing'])}"
    )
    return {"answer": _ask(ASK_CLARIFY_SYSTEM, user, history=state["messages"], temperature=0.5)}


# -- 노드 4: 바이오리듬 계산 (LLM 없음) ------------------------------
def calc_biorhythm(state: BioState) -> dict:
    return {"bio_report": build_report(state)}


# --- 분기 2: 어떤 해설 노드로 보낼까? ---------------------------
def route_after_calc(state: BioState) -> str:
    return TASK_NODES.get(state["task"], "explain_biorhythm")


# -- 노드 5~9: 해설 노드 (LLM) --------------------------------------
def explain_biorhythm(state: BioState) -> dict:
    return _expert(EXPLAIN_SYSTEM, state, temperature=0.4)


def compatibility(state: BioState) -> dict:
    return _expert(COMPATIBILITY_SYSTEM, state, temperature=0.6)


def fortune(state: BioState) -> dict:
    return _expert(FORTUNE_SYSTEM, state, temperature=0.8)


def advice(state: BioState) -> dict:
    return _expert(ADVICE_SYSTEM, state, temperature=0.6)


def exercise(state: BioState) -> dict:
    return _expert(EXERCISE_SYSTEM, state, temperature=0.5)


# ═════════════════════════════════════════════════════════════
# 4) 그래프 조립
# ═════════════════════════════════════════════════════════════

EXPERT_NODES = {
    "explain_biorhythm": explain_biorhythm,
    "compatibility": compatibility,
    "fortune": fortune,
    "advice": advice,
    "exercise": exercise,
}


def build_graph():
    """노드와 엣지를 연결해 그래프를 완성합니다."""
    builder = StateGraph(BioState)

    builder.add_node("analyze_intent", analyze_intent)
    builder.add_node("answer_follow_up", answer_follow_up)
    builder.add_node("ask_clarify", ask_clarify)
    builder.add_node("calc_biorhythm", calc_biorhythm)
    for name, fn in EXPERT_NODES.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "analyze_intent")
    builder.add_conditional_edges(
        "analyze_intent",
        route_after_intent,
        {
            "answer_follow_up": "answer_follow_up",
            "ask_clarify": "ask_clarify",
            "calc_biorhythm": "calc_biorhythm",
        },
    )
    builder.add_conditional_edges(
        "calc_biorhythm",
        route_after_calc,
        {name: name for name in EXPERT_NODES},
    )

    builder.add_edge("answer_follow_up", END)
    builder.add_edge("ask_clarify", END)
    for name in EXPERT_NODES:
        builder.add_edge(name, END)

    return builder.compile()


# --- 터미널에서 바로 실행하기 ----
#
#   python teams/team3/workflow.py "1995년 3월 15일생인데 오늘 바이오리듬 어때?"
#
if __name__ == "__main__":
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
