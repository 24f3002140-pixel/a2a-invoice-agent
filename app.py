import asyncio
import copy
import hashlib
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai import AIError, decide_packages
from database import database_healthcheck, initialize_database
from storage import (
    IdempotencyConflictError,
    TaskConflictError,
    TaskNotFoundError,
    cancel_task,
    canonical_json,
    complete_task,
    create_initial_task,
    get_many_cached_decisions,
    get_task,
    get_task_with_proposals,
    hash_json,
    hash_principal,
    list_tasks,
    resolve_continuation_replay,
    resolve_initial_replay,
    save_many_cached_decisions,
)


# ============================================================
# Constants
# ============================================================

A2A_VERSION = "1.0"
A2A_MEDIA_TYPE = "application/a2a+json"

INPUT_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-claim-batch+json"
)

PROPOSAL_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-proposals+json"
)

RESULT_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-results+json"
)

RECEIPT_MEDIA_TYPE = (
    "application/vnd.ga5.invoice-action-receipts+json"
)

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_CANCELED",
}

MAX_RESPONSE_BYTES = 512 * 1024


# ============================================================
# Configuration
# ============================================================

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://placeholder.onrender.com/a2a/",
).strip()

if not PUBLIC_BASE_URL.endswith("/"):
    PUBLIC_BASE_URL += "/"


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="A2A Invoice Action Agent",
    description="Evidence-backed invoice reconciliation agent.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.on_event("startup")
def startup_event() -> None:
    initialize_database()


# ============================================================
# In-process locks
# ============================================================

_lock_registry_guard = asyncio.Lock()
_request_locks: Dict[str, asyncio.Lock] = {}


async def get_request_lock(key: str) -> asyncio.Lock:
    async with _lock_registry_guard:
        lock = _request_locks.get(key)

        if lock is None:
            lock = asyncio.Lock()
            _request_locks[key] = lock

        return lock


# ============================================================
# General helpers
# ============================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def make_action_id(
    task_id: str,
    package_id: str,
    action: str,
) -> str:
    source = f"{task_id}|{package_id}|{action}"

    digest = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    return f"act_{digest[:28]}"


def request_content_type(request: Request) -> str:
    content_type = request.headers.get(
        "content-type",
        "",
    )

    return content_type.split(
        ";",
        1,
    )[0].strip().lower()


def a2a_response(
    content: Any,
    status_code: int = 200,
) -> JSONResponse:
    encoded = canonical_json(content).encode("utf-8")

    if len(encoded) > MAX_RESPONSE_BYTES:
        return JSONResponse(
            status_code=500,
            media_type=A2A_MEDIA_TYPE,
            content={
                "error": {
                    "code": "RESPONSE_TOO_LARGE",
                    "message": "The response exceeds the allowed size.",
                }
            },
        )

    return JSONResponse(
        status_code=status_code,
        media_type=A2A_MEDIA_TYPE,
        content=content,
    )


def error_response(
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return a2a_response(
        {
            "error": {
                "code": code,
                "message": message,
            }
        },
        status_code=status_code,
    )


async def read_json_body(
    request: Request,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[JSONResponse],
]:
    try:
        body = await request.json()
    except Exception:
        return None, error_response(
            400,
            "INVALID_JSON",
            "The request body must contain valid JSON.",
        )

    if not isinstance(body, dict):
        return None, error_response(
            400,
            "INVALID_REQUEST",
            "The request body must be a JSON object.",
        )

    return body, None


# ============================================================
# Authentication and protocol validation
# ============================================================

def get_bearer_token(
    request: Request,
) -> Tuple[
    Optional[str],
    Optional[JSONResponse],
]:
    authorization = request.headers.get(
        "authorization"
    )

    if not authorization:
        return None, error_response(
            401,
            "UNAUTHENTICATED",
            "Authentication is required.",
        )

    match = re.fullmatch(
        r"Bearer ([^\s]+)",
        authorization,
    )

    if match is None:
        return None, error_response(
            403,
            "INVALID_AUTHENTICATION",
            "Authentication is invalid.",
        )

    return match.group(1), None


def validate_a2a_version(
    request: Request,
) -> Optional[JSONResponse]:
    version = request.headers.get("a2a-version")

    if version != A2A_VERSION:
        return error_response(
            400,
            "UNSUPPORTED_A2A_VERSION",
            "A2A-Version must be 1.0.",
        )

    return None


def validate_a2a_content_type(
    request: Request,
) -> Optional[JSONResponse]:
    if request_content_type(request) != A2A_MEDIA_TYPE:
        return error_response(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            f"Content-Type must be {A2A_MEDIA_TYPE}.",
        )

    return None


def validate_protected_request(
    request: Request,
    require_content_type: bool = False,
) -> Tuple[
    Optional[str],
    Optional[JSONResponse],
]:
    token, authentication_error = get_bearer_token(
        request
    )

    if authentication_error is not None:
        return None, authentication_error

    version_error = validate_a2a_version(request)

    if version_error is not None:
        return None, version_error

    if require_content_type:
        content_type_error = (
            validate_a2a_content_type(request)
        )

        if content_type_error is not None:
            return None, content_type_error

    return hash_principal(token), None


# ============================================================
# Message validation
# ============================================================

def find_message_part(
    message: Dict[str, Any],
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[str],
]:
    parts = message.get("parts")

    if not isinstance(parts, list):
        return None, "parts must be an array."

    if len(parts) != 1:
        return None, (
            "The message must contain exactly one Part."
        )

    part = parts[0]

    if not isinstance(part, dict):
        return None, "The Part must be an object."

    media_type = part.get("mediaType")

    if (
        not isinstance(media_type, str)
        or not media_type
    ):
        return None, "The Part requires mediaType."

    data = part.get("data")

    if not isinstance(data, dict):
        return None, "The Part requires object-valued data."

    return part, None


def validate_message(
    body: Dict[str, Any],
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Optional[JSONResponse],
]:
    message = body.get("message")

    if not isinstance(message, dict):
        return None, None, error_response(
            400,
            "INVALID_MESSAGE",
            "The request requires a message object.",
        )

    message_id = message.get("messageId")

    if (
        not isinstance(message_id, str)
        or not message_id.strip()
    ):
        return None, None, error_response(
            400,
            "INVALID_MESSAGE_ID",
            "messageId must be a nonempty string.",
        )

    if message.get("role") != "ROLE_USER":
        return None, None, error_response(
            400,
            "INVALID_ROLE",
            "The message role must be ROLE_USER.",
        )

    part, part_error = find_message_part(message)

    if part_error is not None:
        return None, None, error_response(
            400,
            "INVALID_PART",
            part_error,
        )

    return message, part, None


def validate_output_modes(
    configuration: Any,
) -> Optional[JSONResponse]:
    if not isinstance(configuration, dict):
        return error_response(
            400,
            "INVALID_CONFIGURATION",
            "configuration must be an object.",
        )

    accepted_modes = configuration.get(
        "acceptedOutputModes"
    )

    if not isinstance(accepted_modes, list):
        return error_response(
            400,
            "INVALID_OUTPUT_MODES",
            "acceptedOutputModes must be an array.",
        )

    accepted_set = {
        mode
        for mode in accepted_modes
        if isinstance(mode, str)
    }

    required_modes = {
        PROPOSAL_MEDIA_TYPE,
        RECEIPT_MEDIA_TYPE,
    }

    if not required_modes.issubset(accepted_set):
        return error_response(
            400,
            "UNSUPPORTED_OUTPUT_MODE",
            "Both required invoice output modes must be accepted.",
        )

    return None


# ============================================================
# Task helpers
# ============================================================

def get_task_state(
    task: Dict[str, Any],
) -> Optional[str]:
    status = task.get("status")

    if not isinstance(status, dict):
        return None

    state = status.get("state")

    if not isinstance(state, str):
        return None

    return state


def make_artifact(
    media_type: str,
    data: Dict[str, Any],
    name: str,
) -> Dict[str, Any]:
    return {
        "artifactId": new_id("artifact"),
        "name": name,
        "parts": [
            {
                "mediaType": media_type,
                "data": data,
            }
        ],
    }


def create_task_status(
    state: str,
) -> Dict[str, str]:
    return {
        "state": state,
        "timestamp": utc_now(),
    }


def index_stored_proposals(
    proposal_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    proposals = proposal_data.get("proposals")

    if not isinstance(proposals, list):
        raise TaskConflictError(
            "The stored proposals are invalid."
        )

    proposal_index: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise TaskConflictError(
                "A stored proposal is invalid."
            )

        package_id = proposal.get("packageId")

        if (
            not isinstance(package_id, str)
            or not package_id
        ):
            raise TaskConflictError(
                "A stored proposal is missing packageId."
            )

        if package_id in proposal_index:
            raise TaskConflictError(
                "Stored proposal package IDs are duplicated."
            )

        proposal_index[package_id] = proposal

    return proposal_index


# ============================================================
# Public routes
# ============================================================

@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "A2A Invoice Action Agent",
        "healthy": database_healthcheck(),
    }


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": (
            "ok"
            if database_healthcheck()
            else "error"
        )
    }


@app.get("/.well-known/agent-card.json")
async def agent_card() -> JSONResponse:
    card = {
        "name": "Invoice Action Agent",
        "description": (
            "An A2A invoice reconciliation agent that "
            "returns evidence-backed business-action proposals "
            "and executes only accepted proposals."
        ),
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "supportedInterfaces": [
            {
                "url": PUBLIC_BASE_URL,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [
            INPUT_MEDIA_TYPE,
        ],
        "defaultOutputModes": [
            PROPOSAL_MEDIA_TYPE,
            RECEIPT_MEDIA_TYPE,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": (
                    "Analyzes invoice packages and selects one "
                    "safe typed action for each package."
                ),
                "tags": [
                    "invoice",
                    "reconciliation",
                    "payment",
                    "approval",
                    "evidence",
                ],
            }
        ],
    }

    return JSONResponse(
        status_code=200,
        media_type="application/json",
        content=card,
    )


# ============================================================
# Initial invoice batch
# ============================================================

async def process_initial_message(
    body: Dict[str, Any],
    message: Dict[str, Any],
    part: Dict[str, Any],
    principal_hash: str,
) -> JSONResponse:
    if part.get("mediaType") != INPUT_MEDIA_TYPE:
        return error_response(
            400,
            "INVALID_INPUT_MODE",
            "The initial message uses an unsupported media type.",
        )

    output_error = validate_output_modes(
        body.get("configuration")
    )

    if output_error is not None:
        return output_error

    data = part["data"]

    batch_id = data.get("batchId")
    policy_revision = data.get("policyRevision")
    packages = data.get("packages")

    if (
        not isinstance(batch_id, str)
        or not batch_id.strip()
    ):
        return error_response(
            400,
            "INVALID_BATCH_ID",
            "batchId must be a nonempty string.",
        )

    if policy_revision is None:
        return error_response(
            400,
            "MISSING_POLICY_REVISION",
            "policyRevision is required.",
        )

    if not isinstance(packages, list) or not packages:
        return error_response(
            400,
            "INVALID_PACKAGES",
            "packages must be a nonempty array.",
        )

    package_ids: List[str] = []

    for package in packages:
        if not isinstance(package, dict):
            return error_response(
                400,
                "INVALID_PACKAGE",
                "Every package must be an object.",
            )

        package_id = package.get("packageId")

        if (
            not isinstance(package_id, str)
            or not package_id.strip()
        ):
            return error_response(
                400,
                "INVALID_PACKAGE_ID",
                "Every package requires a nonempty packageId.",
            )

        package_ids.append(package_id)

    if len(package_ids) != len(set(package_ids)):
        return error_response(
            400,
            "DUPLICATE_PACKAGE_ID",
            "Package IDs must be unique.",
        )

    message_id = message["messageId"]
    message_hash = hash_json(message)

    lock = await get_request_lock(
        f"initial:{principal_hash}:{message_id}"
    )

    async with lock:
        try:
            replay_task = resolve_initial_replay(
                principal_hash=principal_hash,
                message_id=message_id,
                message_hash=message_hash,
            )
        except IdempotencyConflictError:
            return error_response(
                409,
                "IDEMPOTENCY_CONFLICT",
                "The message ID was reused with different content.",
            )

        if replay_task is not None:
            return a2a_response(
                {
                    "task": replay_task,
                }
            )

        task_id = new_id("task")
        context_id = new_id("context")

        cached_decisions, uncached_items = (
            get_many_cached_decisions(packages)
        )

        uncached_packages = [
            package
            for _, package in uncached_items
        ]

        generated_decisions: List[
            Dict[str, Any]
        ] = []

        if uncached_packages:
            try:
                generated_decisions = (
                    await asyncio.to_thread(
                        decide_packages,
                        uncached_packages,
                        policy_revision,
                    )
                )
            except AIError as exc:
                return error_response(
                    502,
                    "AI_DECISION_FAILED",
                    str(exc),
                )
            except Exception:
                return error_response(
                    502,
                    "AI_DECISION_FAILED",
                    "The invoice decision provider failed.",
                )

            generated_by_package = {
                decision["packageId"]: decision
                for decision in generated_decisions
            }

            cache_entries = []

            for package_hash, package in uncached_items:
                package_id = package["packageId"]

                decision = generated_by_package.get(
                    package_id
                )

                if decision is None:
                    return error_response(
                        502,
                        "INCOMPLETE_AI_RESPONSE",
                        "The AI did not return every package decision.",
                    )

                cache_entries.append(
                    (
                        package_hash,
                        package,
                        decision,
                    )
                )

            save_many_cached_decisions(
                cache_entries
            )

        generated_by_package = {
            decision["packageId"]: decision
            for decision in generated_decisions
        }

        proposals: List[Dict[str, Any]] = []
        action_ids = set()

        for package in packages:
            package_id = package["packageId"]
            package_hash = hash_json(package)

            decision = cached_decisions.get(
                package_hash
            )

            if decision is None:
                decision = generated_by_package.get(
                    package_id
                )

            if not isinstance(decision, dict):
                return error_response(
                    500,
                    "DECISION_NOT_AVAILABLE",
                    "A package decision was unavailable.",
                )

            action = decision.get("action")

            if action not in ALLOWED_ACTIONS:
                return error_response(
                    500,
                    "INVALID_STORED_DECISION",
                    "A package decision contains an invalid action.",
                )

            action_id = make_action_id(
                task_id,
                package_id,
                action,
            )

            if action_id in action_ids:
                return error_response(
                    500,
                    "DUPLICATE_ACTION_ID",
                    "Generated action IDs were duplicated.",
                )

            action_ids.add(action_id)

            proposal = {
                "packageId": package_id,
                "actionId": action_id,
                "action": action,
                "facts": copy.deepcopy(
                    decision.get("facts")
                ),
                "evidenceRefs": copy.deepcopy(
                    decision.get("evidenceRefs")
                ),
                "rationale": decision.get("rationale"),
            }

            proposals.append(proposal)

        proposal_data = {
            "batchId": batch_id,
            "proposals": proposals,
        }

        proposal_artifact = make_artifact(
            PROPOSAL_MEDIA_TYPE,
            proposal_data,
            "Invoice action proposals",
        )

        task = {
            "id": task_id,
            "contextId": context_id,
            "status": create_task_status(
                "TASK_STATE_INPUT_REQUIRED"
            ),
            "history": [
                copy.deepcopy(message),
            ],
            "artifacts": [
                proposal_artifact,
            ],
        }

        try:
            stored_task, _created = (
                create_initial_task(
                    task_id=task_id,
                    context_id=context_id,
                    principal_hash=principal_hash,
                    batch_id=batch_id,
                    message_id=message_id,
                    message_hash=message_hash,
                    task=task,
                    proposals=proposal_data,
                )
            )
        except IdempotencyConflictError:
            return error_response(
                409,
                "IDEMPOTENCY_CONFLICT",
                "The message ID was reused with different content.",
            )

        return a2a_response(
            {
                "task": stored_task,
            }
        )


# ============================================================
# Result continuation
# ============================================================

async def process_result_message(
    message: Dict[str, Any],
    part: Dict[str, Any],
    principal_hash: str,
) -> JSONResponse:
    if part.get("mediaType") != RESULT_MEDIA_TYPE:
        return error_response(
            400,
            "INVALID_CONTINUATION_MODE",
            "The continuation uses an unsupported media type.",
        )

    task_id = message.get("taskId")
    context_id = message.get("contextId")

    if (
        not isinstance(task_id, str)
        or not task_id.strip()
    ):
        return error_response(
            400,
            "MISSING_TASK_ID",
            "A continuation requires taskId.",
        )

    if (
        not isinstance(context_id, str)
        or not context_id.strip()
    ):
        return error_response(
            400,
            "MISSING_CONTEXT_ID",
            "A continuation requires contextId.",
        )

    message_id = message["messageId"]
    message_hash = hash_json(message)

    lock = await get_request_lock(
        f"task:{principal_hash}:{task_id}"
    )

    async with lock:
        try:
            replay_task = resolve_continuation_replay(
                principal_hash=principal_hash,
                message_id=message_id,
                message_hash=message_hash,
            )
        except IdempotencyConflictError:
            return error_response(
                409,
                "IDEMPOTENCY_CONFLICT",
                "The continuation message ID was reused with different content.",
            )

        if replay_task is not None:
            return a2a_response(
                {
                    "task": replay_task,
                }
            )

        try:
            stored = get_task_with_proposals(
                task_id,
                principal_hash,
            )
        except TaskNotFoundError:
            return error_response(
                404,
                "TASK_NOT_FOUND",
                "The requested task was not found.",
            )

        if stored["context_id"] != context_id:
            return error_response(
                409,
                "CONTINUATION_MISMATCH",
                "The continuation does not match the task.",
            )

        current_task = stored["task"]
        current_state = get_task_state(
            current_task
        )

        if current_state in TERMINAL_STATES:
            return error_response(
                409,
                "TASK_ALREADY_TERMINAL",
                "The task is already terminal.",
            )

        if (
            current_state
            != "TASK_STATE_INPUT_REQUIRED"
        ):
            return error_response(
                409,
                "TASK_NOT_WAITING_FOR_INPUT",
                "The task is not waiting for results.",
            )

        result_data = part["data"]

        if (
            result_data.get("batchId")
            != stored["batch_id"]
        ):
            return error_response(
                409,
                "CONTINUATION_MISMATCH",
                "The result batch does not match the task.",
            )

        results = result_data.get("results")

        if not isinstance(results, list):
            return error_response(
                400,
                "INVALID_RESULTS",
                "results must be an array.",
            )

        proposal_data = stored.get("proposals")

        if not isinstance(proposal_data, dict):
            return error_response(
                409,
                "INVALID_TASK_STATE",
                "The stored proposals are unavailable.",
            )

        try:
            proposal_index = index_stored_proposals(
                proposal_data
            )
        except TaskConflictError:
            return error_response(
                409,
                "INVALID_TASK_STATE",
                "The stored proposal data is invalid.",
            )

        if len(results) != len(proposal_index):
            return error_response(
                409,
                "RESULT_SET_MISMATCH",
                "The results must match all stored proposals.",
            )

        seen_package_ids = set()
        seen_action_ids = set()
        executions: List[Dict[str, Any]] = []

        for result in results:
            if not isinstance(result, dict):
                return error_response(
                    400,
                    "INVALID_RESULT",
                    "Every result must be an object.",
                )

            package_id = result.get("packageId")
            action_id = result.get("actionId")
            action = result.get("action")
            outcome = result.get("outcome")
            receipt_nonce = result.get(
                "receiptNonce"
            )

            if (
                not isinstance(package_id, str)
                or package_id in seen_package_ids
            ):
                return error_response(
                    409,
                    "RESULT_SET_MISMATCH",
                    "Result package IDs are invalid.",
                )

            if (
                not isinstance(action_id, str)
                or action_id in seen_action_ids
            ):
                return error_response(
                    409,
                    "RESULT_SET_MISMATCH",
                    "Result action IDs are invalid.",
                )

            if outcome not in {
                "ACCEPTED",
                "REJECTED",
            }:
                return error_response(
                    400,
                    "INVALID_OUTCOME",
                    "outcome must be ACCEPTED or REJECTED.",
                )

            if (
                not isinstance(receipt_nonce, str)
                or not receipt_nonce
            ):
                return error_response(
                    400,
                    "INVALID_RECEIPT_NONCE",
                    "Every result requires a receiptNonce.",
                )

            proposal = proposal_index.get(
                package_id
            )

            if proposal is None:
                return error_response(
                    409,
                    "RESULT_SET_MISMATCH",
                    "A result does not match a stored proposal.",
                )

            if (
                proposal.get("actionId")
                != action_id
                or proposal.get("action")
                != action
            ):
                return error_response(
                    409,
                    "ACTION_IDENTITY_MISMATCH",
                    "A result action does not match its proposal.",
                )

            seen_package_ids.add(package_id)
            seen_action_ids.add(action_id)

            if outcome == "ACCEPTED":
                executions.append(
                    {
                        "packageId": package_id,
                        "actionId": action_id,
                        "action": action,
                        "receiptNonce": receipt_nonce,
                        "facts": copy.deepcopy(
                            proposal.get("facts")
                        ),
                        "evidenceRefs": copy.deepcopy(
                            proposal.get(
                                "evidenceRefs"
                            )
                        ),
                    }
                )

        if seen_package_ids != set(
            proposal_index.keys()
        ):
            return error_response(
                409,
                "RESULT_SET_MISMATCH",
                "The results do not cover all proposals.",
            )

        receipt_data = {
            "batchId": stored["batch_id"],
            "executions": executions,
        }

        receipt_artifact = make_artifact(
            RECEIPT_MEDIA_TYPE,
            receipt_data,
            "Invoice action receipts",
        )

        completed_task = copy.deepcopy(
            current_task
        )

        completed_task["status"] = (
            create_task_status(
                "TASK_STATE_COMPLETED"
            )
        )

        history = completed_task.get(
            "history",
            [],
        )

        if not isinstance(history, list):
            history = []

        history.append(copy.deepcopy(message))
        completed_task["history"] = history

        artifacts = completed_task.get(
            "artifacts",
            [],
        )

        if not isinstance(artifacts, list):
            artifacts = []

        artifacts.append(receipt_artifact)
        completed_task["artifacts"] = artifacts

        try:
            final_task, _changed = complete_task(
                task_id=task_id,
                principal_hash=principal_hash,
                continuation_message_id=message_id,
                continuation_message_hash=message_hash,
                completed_task=completed_task,
                receipts=receipt_data,
            )
        except IdempotencyConflictError:
            return error_response(
                409,
                "IDEMPOTENCY_CONFLICT",
                "The continuation message ID was reused with different content.",
            )
        except TaskNotFoundError:
            return error_response(
                404,
                "TASK_NOT_FOUND",
                "The requested task was not found.",
            )
        except TaskConflictError:
            return error_response(
                409,
                "TASK_STATE_CONFLICT",
                "The task changed concurrently.",
            )

        return a2a_response(
            {
                "task": final_task,
            }
        )


# ============================================================
# POST /a2a/message:send
# ============================================================

@app.post("/a2a/message:send")
@app.post("/a2a/message:send/")
async def message_send(
    request: Request,
) -> JSONResponse:
    principal_hash, protection_error = (
        validate_protected_request(
            request,
            require_content_type=True,
        )
    )

    if protection_error is not None:
        return protection_error

    body, body_error = await read_json_body(
        request
    )

    if body_error is not None:
        return body_error

    message, part, message_error = (
        validate_message(body)
    )

    if message_error is not None:
        return message_error

    is_continuation = (
        message.get("taskId") is not None
        or message.get("contextId") is not None
        or part.get("mediaType")
        == RESULT_MEDIA_TYPE
    )

    if is_continuation:
        return await process_result_message(
            message=message,
            part=part,
            principal_hash=principal_hash,
        )

    return await process_initial_message(
        body=body,
        message=message,
        part=part,
        principal_hash=principal_hash,
    )


# ============================================================
# GET /a2a/tasks
# ============================================================

@app.get("/a2a/tasks")
@app.get("/a2a/tasks/")
async def get_tasks(
    request: Request,
) -> JSONResponse:
    principal_hash, protection_error = (
        validate_protected_request(request)
    )

    if protection_error is not None:
        return protection_error

    tasks = list_tasks(principal_hash)

    return a2a_response(
        {
            "tasks": tasks,
        }
    )


# ============================================================
# GET /a2a/tasks/{task_id}
# ============================================================

@app.get("/a2a/tasks/{task_id}")
@app.get("/a2a/tasks/{task_id}/")
async def get_single_task(
    task_id: str,
    request: Request,
) -> JSONResponse:
    principal_hash, protection_error = (
        validate_protected_request(request)
    )

    if protection_error is not None:
        return protection_error

    try:
        task = get_task(
            task_id,
            principal_hash,
        )
    except TaskNotFoundError:
        return error_response(
            404,
            "TASK_NOT_FOUND",
            "The requested task was not found.",
        )

    return a2a_response(task)


# ============================================================
# POST /a2a/tasks/{task_id}:cancel
# ============================================================

@app.post("/a2a/tasks/{task_id}:cancel")
@app.post("/a2a/tasks/{task_id}:cancel/")
async def cancel_existing_task(
    task_id: str,
    request: Request,
) -> JSONResponse:
    principal_hash, protection_error = (
        validate_protected_request(
            request,
            require_content_type=True,
        )
    )

    if protection_error is not None:
        return protection_error

    lock = await get_request_lock(
        f"task:{principal_hash}:{task_id}"
    )

    async with lock:
        try:
            current_task = get_task(
                task_id,
                principal_hash,
            )
        except TaskNotFoundError:
            return error_response(
                404,
                "TASK_NOT_FOUND",
                "The requested task was not found.",
            )

        current_state = get_task_state(
            current_task
        )

        if current_state in TERMINAL_STATES:
            return error_response(
                409,
                "TASK_ALREADY_TERMINAL",
                "The task is already terminal.",
            )

        canceled_task = copy.deepcopy(
            current_task
        )

        canceled_task["status"] = (
            create_task_status(
                "TASK_STATE_CANCELED"
            )
        )

        existing_artifacts = (
            canceled_task.get(
                "artifacts",
                [],
            )
        )

        retained_artifacts = []

        if isinstance(existing_artifacts, list):
            for artifact in existing_artifacts:
                if not isinstance(artifact, dict):
                    continue

                parts = artifact.get(
                    "parts",
                    [],
                )

                contains_receipt = False

                if isinstance(parts, list):
                    for artifact_part in parts:
                        if (
                            isinstance(
                                artifact_part,
                                dict,
                            )
                            and artifact_part.get(
                                "mediaType"
                            )
                            == RECEIPT_MEDIA_TYPE
                        ):
                            contains_receipt = True
                            break

                if not contains_receipt:
                    retained_artifacts.append(
                        artifact
                    )

        canceled_task["artifacts"] = (
            retained_artifacts
        )

        try:
            final_task = cancel_task(
                task_id=task_id,
                principal_hash=principal_hash,
                canceled_task=canceled_task,
            )
        except TaskNotFoundError:
            return error_response(
                404,
                "TASK_NOT_FOUND",
                "The requested task was not found.",
            )
        except TaskConflictError:
            return error_response(
                409,
                "TASK_STATE_CONFLICT",
                "The task changed concurrently.",
            )

        return a2a_response(final_task)
