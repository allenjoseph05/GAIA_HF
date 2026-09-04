"""Checkpointed human approval interrupt for one frozen GAIA submission."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gaia_max.preflight import FrozenCandidateManifest
from gaia_max.submission import (
    GaiaSubmissionClient,
    SubmissionApproval,
    SubmissionBlockedError,
    SubmissionReceipt,
    SubmissionTransportError,
)

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false


class SubmissionWorkflowStatus(StrEnum):
    NEW = "new"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    BLOCKED = "blocked"
    FAILED = "failed"


class SubmissionApprovalRequest(BaseModel):
    """Answer-free interrupt value safe for an operator surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    username: str = Field(min_length=1)
    required_phrase: Literal["SUBMIT FROZEN GAIA CANDIDATE"] = (
        "SUBMIT FROZEN GAIA CANDIDATE"
    )


class SubmissionWorkflowState(TypedDict):
    manifest: FrozenCandidateManifest
    username: str
    agent_code: str
    approval: SubmissionApproval | None
    receipt: SubmissionReceipt | None
    status: SubmissionWorkflowStatus
    error_code: str | None


class SubmissionWorkflowInput(BaseModel):
    """Private input bound to a single frozen candidate and public Space identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: FrozenCandidateManifest
    username: str = Field(min_length=1)
    agent_code: str = Field(min_length=1)

    def initial_state(self) -> SubmissionWorkflowState:
        return {
            "manifest": self.manifest,
            "username": self.username,
            "agent_code": self.agent_code,
            "approval": None,
            "receipt": None,
            "status": SubmissionWorkflowStatus.NEW,
            "error_code": None,
        }


class SubmissionApprovalGraph:
    """Pause for hash-bound approval, then perform at most one graph submission call."""

    def __init__(self, client: GaiaSubmissionClient) -> None:
        self._client = client

    def compile(
        self,
        *,
        checkpointer: BaseCheckpointSaver[str],
    ) -> CompiledStateGraph[
        SubmissionWorkflowState,
        None,
        SubmissionWorkflowState,
        SubmissionWorkflowState,
    ]:
        builder = StateGraph(SubmissionWorkflowState)
        builder.add_node("preflight", self._preflight)
        builder.add_node("request_approval", self._request_approval)
        builder.add_node("submit", self._submit)
        builder.add_edge(START, "preflight")
        builder.add_conditional_edges(
            "preflight",
            self._after_preflight,
            {"approval": "request_approval", "end": END},
        )
        builder.add_conditional_edges(
            "request_approval",
            self._after_approval,
            {"submit": "submit", "end": END},
        )
        builder.add_edge("submit", END)
        return builder.compile(
            checkpointer=checkpointer,
            name="gaia_submission_approval_graph",
        )

    def _preflight(self, state: SubmissionWorkflowState) -> dict[str, object]:
        try:
            self._client.validate_candidate(state["manifest"])
        except SubmissionBlockedError:
            return {
                "status": SubmissionWorkflowStatus.BLOCKED,
                "error_code": "candidate_preflight_blocked",
            }
        return {"status": SubmissionWorkflowStatus.AWAITING_APPROVAL}

    @staticmethod
    def _request_approval(state: SubmissionWorkflowState) -> dict[str, object]:
        request = SubmissionApprovalRequest(
            candidate_sha256=state["manifest"].candidate_sha256,
            snapshot_sha256=state["manifest"].snapshot_sha256,
            task_count=len(state["manifest"].answers),
            username=state["username"],
        )
        raw_approval = interrupt(request.model_dump(mode="json"))
        try:
            approval = SubmissionApproval.model_validate(raw_approval)
        except ValidationError:
            return {
                "status": SubmissionWorkflowStatus.BLOCKED,
                "error_code": "approval_invalid",
            }
        return {
            "approval": approval,
            "status": SubmissionWorkflowStatus.APPROVED,
        }

    async def _submit(self, state: SubmissionWorkflowState) -> dict[str, object]:
        approval = state["approval"]
        if approval is None:
            return {
                "status": SubmissionWorkflowStatus.BLOCKED,
                "error_code": "approval_missing",
            }
        try:
            receipt = await self._client.submit_once(
                manifest=state["manifest"],
                approval=approval,
                username=state["username"],
                agent_code=state["agent_code"],
            )
        except SubmissionBlockedError:
            return {
                "status": SubmissionWorkflowStatus.BLOCKED,
                "error_code": "submission_gate_blocked",
            }
        except SubmissionTransportError:
            return {
                "status": SubmissionWorkflowStatus.FAILED,
                "error_code": "submission_outcome_unknown",
            }
        return {
            "receipt": receipt,
            "status": SubmissionWorkflowStatus.SUBMITTED,
        }

    @staticmethod
    def _after_preflight(
        state: SubmissionWorkflowState,
    ) -> Literal["approval", "end"]:
        if state["status"] is SubmissionWorkflowStatus.AWAITING_APPROVAL:
            return "approval"
        return "end"

    @staticmethod
    def _after_approval(
        state: SubmissionWorkflowState,
    ) -> Literal["submit", "end"]:
        if state["status"] is SubmissionWorkflowStatus.APPROVED:
            return "submit"
        return "end"


def build_submission_graph(
    client: GaiaSubmissionClient,
    *,
    checkpointer: BaseCheckpointSaver[str],
) -> CompiledStateGraph[
    SubmissionWorkflowState,
    None,
    SubmissionWorkflowState,
    SubmissionWorkflowState,
]:
    return SubmissionApprovalGraph(client).compile(checkpointer=checkpointer)


__all__ = [
    "SubmissionApprovalGraph",
    "SubmissionApprovalRequest",
    "SubmissionWorkflowInput",
    "SubmissionWorkflowState",
    "SubmissionWorkflowStatus",
    "build_submission_graph",
]
