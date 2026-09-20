"""Stage 1 node - turn a user request into a retrieved, verified paper."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from essay_agent.errors import LLMError, PaperSearchError
from essay_agent.nodes.base import Deps, NodeReturn, abort_state, fail_state
from essay_agent.prompts import fetch as prompts
from essay_agent.schemas.dialogue import MatchDecision
from essay_agent.schemas.paper import PaperSearchQuery, title_overlap
from essay_agent.state import RunState
from essay_agent.tools.paper_search import build_search_tool
from essay_agent.workspace import make_slug

MIN_TITLE_OVERLAP = 0.34


def make_fetch_node(deps: Deps) -> Callable[[RunState], NodeReturn]:
    """Build the stage-1 node."""

    def fetch(state: RunState) -> NodeReturn:
        ui = deps.ui
        workspace = deps.workspace(state)
        query = (state.get("query") or "").strip()
        ui.step("stage 1/5 - paper retrieval", "asking the search tool")
        if not query:
            return fail_state(state, "empty request: nothing to search for")

        # ---------------------------------------------------- 1. plan the search
        tool = build_search_tool(deps.searcher)
        try:
            search_args: PaperSearchQuery = deps.llm.tool_args(
                prompts.QUERY_SYSTEM,
                prompts.query_user_message(query),
                tool,
                PaperSearchQuery,
                label="paper_search_args",
            )
        except LLMError as exc:
            ui.error(f"could not build a search from your request: {exc}")
            return fail_state(state, f"search planning failed: {exc}")
        workspace.write_json("paper/search_args.json", search_args.model_dump(mode="json"))
        workspace.log(f"search args: {search_args.model_dump(mode='json')}")
        ui.dim(f"search: {search_args.model_dump(exclude_none=True)}")

        # ------------------------------------------------------ 2. run the tool
        try:
            candidates = deps.searcher.search(search_args)
        except PaperSearchError as exc:
            ui.error(f"search backends failed: {exc}")
            return fail_state(state, f"paper search failed: {exc}")
        workspace.write_json(
            "paper/candidates.json", [c.model_dump(mode="json") for c in candidates]
        )
        # A partial candidate list must never look like a complete one: without this the
        # model picks from a single backend and nobody learns the others were down.
        search_report = {
            "notes": list(deps.searcher.backend_notes),
            "failures": list(deps.searcher.backend_failures),
        }
        if search_report["notes"] or search_report["failures"]:
            workspace.write_json("paper/search_report.json", search_report)
            workspace.log(f"search report: {search_report}")
        if search_report["failures"]:
            ui.warn(
                "these search backends did not answer, so the candidate list may be "
                "incomplete: " + "; ".join(search_report["failures"])
            )
        if not candidates:
            ui.error("未匹配到对应论文 (no matching paper found)")
            return abort_state(state, "未匹配到对应论文 (no matching paper found)")
        ui.dim(f"{len(candidates)} candidate(s) retrieved")

        # ------------------------------------------------- 3. pick the right one
        decision: MatchDecision | None = None
        try:
            decision = deps.llm.json(
                prompts.MATCH_SYSTEM,
                prompts.match_user_message(query, candidates),
                MatchDecision,
                label="paper_match",
            )
        except LLMError as exc:
            ui.warn(f"could not verify the match automatically ({exc}); asking you instead")

        threshold = deps.settings.search.user_choice_confidence
        chosen_index: int | None = None
        reason = ""
        if (
            decision is not None
            and decision.index is not None
            and 0 <= decision.index < len(candidates)
            and not decision.ambiguous
            and decision.confidence >= threshold
        ):
            chosen_index = decision.index
            reason = decision.reason
        if chosen_index is None:
            why = decision.reason if decision else "the retrieval step could not be verified"
            ui.warn(f"the agent is not certain which paper you meant: {why}")
            ui.notice(
                "paper disambiguation",
                "Pick the paper to reproduce from the list below.\n"
                "Choosing 0 aborts the task (no files are published).",
            )
            chosen_index = ui.choose_candidate(candidates, "which paper should be reproduced?")
        if chosen_index is None:
            ui.error("未匹配到对应论文 (no matching paper found)")
            return abort_state(state, "未匹配到对应论文 (user did not select a paper)")

        candidate = candidates[chosen_index]
        ui.success(f"selected: {candidate.title} ({candidate.year or 'n.d.'})")
        if reason:
            ui.dim(f"match reason: {reason}")

        # ---------------------------------------------------- 4. get the source
        ui.step("stage 1/5 - fetching full text", candidate.title[:70])
        text, sections, info = deps.searcher.fetch_full_text(candidate)
        used_url = str(info.get("pdf_source_url") or "")
        if used_url and used_url != (candidate.pdf_url or ""):
            ui.dim(f"full text came from a fallback link: {used_url}")
        record = deps.searcher.build_record(candidate, text=text, sections=sections, info=info)
        workspace.write_json("paper/paper.json", record.model_dump(mode="json"))
        workspace.write_text("paper/full_text.md", text)
        workspace.write_json(
            "paper/sections_index.json",
            [
                {"index": s.index, "name": s.name, "page": s.page_hint, "chars": len(s.text)}
                for s in sections
            ],
        )

        slug = make_slug(record.title)
        extra: dict[str, Any] = {
            "slug": slug,
            "paper": record.model_dump(mode="json"),
            "paper_text": record.full_text,
            "sections": [s.model_dump(mode="json") for s in sections],
            "candidates": [c.model_dump(mode="json") for c in candidates],
            "chosen_candidate": candidate.model_dump(mode="json"),
            "match_decision": decision.model_dump(mode="json") if decision else {},
            "search_args": search_args.model_dump(mode="json"),
            "search_report": search_report,
        }

        # ------------------------------------------ 5. is the document plausible?
        if record.full_text_available:
            pdf_title = str(info.get("pdf_title") or "")
            if pdf_title and title_overlap(pdf_title, record.title) < MIN_TITLE_OVERLAP:
                ui.warn(
                    "the downloaded document does not obviously match the requested title:\n"
                    f"  requested: {record.title}\n  document : {pdf_title}"
                )
                if not deps.ui.confirm("continue with this document?", default=True):
                    return abort_state(state, "user rejected the downloaded document", extra=extra)
            ui.success(
                f"full text ready ({len(record.full_text):,} chars, "
                f"{len(record.sections) or 1} section block(s))"
            )
        else:
            ui.warn("only the abstract could be retrieved (no accessible PDF)")
            if deps.settings.search.require_full_text:
                ui.notice(
                    "full text unavailable",
                    "The reproduction cards would be written from the abstract alone.\n"
                    "That is usually not enough to reproduce an experiment honestly.",
                )
                if not deps.ui.confirm("continue with the abstract only?", default=True):
                    return abort_state(
                        state, "full text unavailable and the user declined", extra=extra
                    )
            else:
                ui.dim("search.require_full_text is disabled; continuing with the abstract")

        workspace.log(
            f"fetched '{record.title}' ({record.source}) full_text={record.full_text_available}"
        )
        return {
            "status": "running",
            "fetch_status": "ok",
            "fetch_message": f"retrieved {record.title}",
            "started_at": state.get("started_at")
            or datetime.now(UTC).isoformat(timespec="seconds"),
            **extra,
        }

    return fetch
