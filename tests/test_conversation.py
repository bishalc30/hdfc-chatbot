"""Rule-based conversational routing (src/chat.py): only factual answers call the LLM.

Offline tests use a fake LLM that counts calls (retrieval is real: ONNX + exported index).
The live regression test calls the real LLM and only runs with LIVE_LLM=1:
    LIVE_LLM=1 pytest tests/test_conversation.py -s
"""
import os
import re

os.environ.setdefault("RETRIEVAL_BACKEND", "serving")       # ONNX + exported index, no chromadb needed

import pytest  # noqa: E402

from src import answerer as A  # noqa: E402
from src import chat as C  # noqa: E402

EM_DASH = "\N{EM DASH}"
MID = "HDFC Mid Cap Fund"
LIVE = os.getenv("LIVE_LLM") == "1"


class FakeLLM:
    """Answers from the first CONTEXT block and cites its source_url, like a well-behaved model."""
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def complete(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise TimeoutError("LLM down")
        url = re.search(r"source_url=(\S+?)\)", messages[-1]["content"]).group(1)
        return f"Fake fact for the scheme.\nSOURCE: {url}"


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(A, "get_llm", lambda: fake)
    C._CACHE.clear()
    yield fake
    C._CACHE.clear()


def turn(llm, message, fund=None, pill=None):
    before = llm.calls
    r = C.chat(message, {"fund": fund}, pill)
    return r, llm.calls - before


# ---------------------------------------------------------------- routing: LLM calls per message
@pytest.mark.parametrize("message,fund,route,calls", [
    ("Hi", None, "greeting", 0),
    ("how are you", None, "greeting", 0),
    ("What can you do?", None, "greeting", 0),
    ("thanks", None, "closing", 0),
    ("ok bye", None, "closing", 0),
    ("Should I invest in Mid Cap?", None, "advice", 0),
    ("compare large cap and mid cap", None, "fallback", 0),
    ("axis mid cap exit load", None, "fallback", 0),
    ("what's the weather", None, "fallback", 0),
    ("asdf", None, "fallback", 0),
    ("I'm not feeling good", None, "fallback", 0),
    ("my PAN is ABCDE1234F", None, "pii", 0),
    ("HDFC", None, "fund_step", 0),
    ("mid cap", None, "question_step", 0),
    ("what is the exit load?", None, "ask_fund", 0),
    ("what is the exit load?", MID, "factual", 1),
    ("What is the expense ratio of HDFC Large Cap Fund?", None, "factual", 1),
    ("who manages the flexi cap fund", None, "factual", 1),
])
def test_routes_and_llm_calls(llm, message, fund, route, calls):
    r, n = turn(llm, message, fund)
    assert (r["route"], n) == (route, calls)
    assert EM_DASH not in r["text"]
    if route != "factual" and route != "advice":
        assert r["url"] is None                                      # non-factual replies carry no link


@pytest.mark.parametrize("message,fund,fact", [
    ("midcap exit load", "HDFC Mid Cap Fund", "exit_load"),
    ("expence ratio of largecap", "HDFC Large Cap Fund", "expense_ratio"),
    ("elss lockin", "HDFC ELSS Tax Saver Fund", "lock_in"),
    ("min sip flexi cap", "HDFC Flexi Cap Fund", "min_sip"),
    ("risk level of tax saver", "HDFC ELSS Tax Saver Fund", "riskometer"),
    ("TER of mid-cap", "HDFC Mid Cap Fund", "expense_ratio"),
    ("flexicap exitload", "HDFC Flexi Cap Fund", "exit_load"),
    ("largecap exit charge", "HDFC Large Cap Fund", "exit_load"),
    ("sip amount midcap", "HDFC Mid Cap Fund", "min_sip"),
    ("benchmark index of flexi cap", "HDFC Flexi Cap Fund", "benchmark"),
    ("taxsaver lock in", "HDFC ELSS Tax Saver Fund", "lock_in"),
    ("risk of elss", "HDFC ELSS Tax Saver Fund", "riskometer"),
])
def test_aliases_and_typos_route_to_factual(llm, message, fund, fact):
    r, n = turn(llm, message)
    assert (r["route"], r["state"]["fund"], r["asked"], n) == ("factual", fund, C.question_for(fund, fact), 1)


@pytest.mark.parametrize("message", ["what is the expected return", "tell me a joke", "lock my account",
                                     "is it risky", "exit", "what is nav"])
def test_fuzzy_matching_has_no_false_positives(llm, message):
    r, n = turn(llm, message)
    assert r["route"] != "factual" and n == 0


def test_fallback_does_not_claim_missing_information():
    assert all("one HDFC fund" in t or "single HDFC fund" in t for t in C.REPLIES["fallback"])
    assert not any("don't have" in t or "outside what I know" in t for t in C.REPLIES["fallback"])


def test_pill_click_skips_classification(llm):
    pill = C.question_pills(MID)[1]                                  # Exit load
    assert (pill["action"], pill["fund"], pill["fact"]) == ("fact", MID, "exit_load")
    r, n = turn(llm, "anything at all", None, pill)                  # the text is not used for routing
    assert r["route"] == "factual" and n == 1 and r["state"]["fund"] == MID
    assert r["asked"] == "What is the exit load of HDFC Mid Cap Fund?"


def test_navigation_pills_make_no_llm_call(llm):
    r, n = turn(llm, "HDFC", None, C.house_pills()[0])
    assert r["route"] == "fund_step" and n == 0
    r, n = turn(llm, MID, None, C.fund_pills()[3])
    assert r["route"] == "question_step" and r["state"]["fund"] == MID and n == 0


def test_repeat_question_is_cached(llm):
    q = "What is the minimum SIP amount for HDFC Mid Cap Fund?"
    first, n1 = turn(llm, q)
    second, n2 = turn(llm, q)
    third, n3 = turn(llm, "min sip for mid cap?")                    # same (fund, fact), other wording
    assert (n1, n2, n3) == (1, 0, 0)
    assert not first["cached"] and second["cached"] and third["cached"]
    assert second["url"] == first["url"] and second["last_updated"] == first["last_updated"]


def test_fact_without_fund_asks_which_fund_then_answers(llm):
    r, n = turn(llm, "what is the benchmark?")
    assert r["route"] == "ask_fund" and n == 0
    assert all(p["action"] == "fact" and p["fact"] == "benchmark" for p in r["pills"])
    r, n = turn(llm, r["pills"][0]["message"], None, r["pills"][0])
    assert r["route"] == "factual" and n == 1


def test_advice_has_one_education_link(llm):
    r, _ = turn(llm, "Which fund is best?")
    assert r["route"] == "advice" and r["url"] == "https://www.amfiindia.com/investor"
    assert r["link_label"] == "Learn more"


def test_pii_is_never_logged(llm, monkeypatch):
    logged = []
    monkeypatch.setattr(C, "write_trace", lambda t: logged.append(str(t)))
    monkeypatch.setattr(A, "write_trace", lambda t: logged.append(str(t)))
    r, n = turn(llm, "my PAN is ABCDE1234F, phone 9876543210")
    assert r["route"] == "pii" and n == 0
    assert logged and not any("ABCDE1234F" in s or "9876543210" in s for s in logged)


def test_llm_failure_shows_generic_error_without_retry(monkeypatch):
    fake = FakeLLM(fail=True)
    monkeypatch.setattr(A, "get_llm", lambda: fake)
    C._CACHE.clear()
    r = C.chat("What is the exit load of HDFC Mid Cap Fund?", {"fund": None})
    assert fake.calls == 1                                           # no retry
    assert r["intent"] == "error" and r["text"] == A.MSG["error"]
    assert ("HDFC Mid Cap Fund", "exit_load") not in C._CACHE        # errors are never cached


def test_not_in_sources_has_no_link_without_a_named_fund():
    r = A.answer("What is the colour of the sky?")
    assert r["intent"] == "not_available" and r["url"] is None


def test_fixed_replies_have_no_em_dashes():
    assert not any(EM_DASH in t for variants in C.REPLIES.values() for t in variants)
    assert all(2 <= len(v) <= 3 for v in C.REPLIES.values())


# ---------------------------------------------------------------- live regression (real LLM)
REGRESSION = ["What is the expense ratio of HDFC Large Cap Fund?", "What is the exit load of HDFC Flexi Cap Fund?",
              "What is the minimum SIP amount for HDFC Mid Cap Fund?", "What is the riskometer level of HDFC ELSS Tax Saver?",
              "What is the lock-in period of HDFC ELSS Tax Saver?"]


@pytest.mark.skipif(not LIVE, reason="calls the real LLM; set LIVE_LLM=1")
@pytest.mark.parametrize("q", REGRESSION)
def test_live_factual_regression(q):
    import time
    C._CACHE.clear()
    r = C.chat(q, {"fund": None})
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", r["text"].strip()) if s]
    print(f"\n{q}\n  -> {r['text']}\n     {r['url']} | last updated {r['last_updated']} | {len(sentences)} sentence(s)")
    assert r["intent"] == "factual"
    assert r["url"].startswith("https://groww.in/mutual-funds/") and r["last_updated"]
    assert len(sentences) <= 3 and EM_DASH not in r["text"]
    time.sleep(3)
