from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ----------------------------------------------------
# Generic A2A Models
# ----------------------------------------------------

Role = Literal["ROLE_USER", "ROLE_AGENT"]

TaskState = Literal[
    "TASK_STATE_SUBMITTED",
    "TASK_STATE_WORKING",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_COMPLETED",
    "TASK_STATE_CANCELED",
]


class Part(BaseModel):
    mediaType: str
    data: Dict[str, Any]


class Message(BaseModel):
    messageId: str
    role: Role
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    parts: List[Part]


class Configuration(BaseModel):
    returnImmediately: bool = False
    historyLength: int = 20
    acceptedOutputModes: List[str]


class SendRequest(BaseModel):
    message: Message
    configuration: Configuration


# ----------------------------------------------------
# Invoice Proposal Models
# ----------------------------------------------------

InvoiceAction = Literal[
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
]


class InvoiceFacts(BaseModel):
    vendorName: str
    invoiceNumber: str
    amountMinor: int
    currency: str


class InvoiceProposal(BaseModel):
    packageId: str
    actionId: str
    action: InvoiceAction
    facts: InvoiceFacts
    evidenceRefs: List[str]
    rationale: str


class ProposalArtifact(BaseModel):
    batchId: str
    proposals: List[InvoiceProposal]


# ----------------------------------------------------
# Receipt Models
# ----------------------------------------------------

Outcome = Literal["ACCEPTED", "REJECTED"]


class InvoiceResult(BaseModel):
    packageId: str
    actionId: str
    action: InvoiceAction
    outcome: Outcome
    receiptNonce: str


class ResultArtifact(BaseModel):
    batchId: str
    results: List[InvoiceResult]


class InvoiceExecution(BaseModel):
    packageId: str
    actionId: str
    action: InvoiceAction
    receiptNonce: str
    facts: InvoiceFacts
    evidenceRefs: List[str]


class ReceiptArtifact(BaseModel):
    batchId: str
    executions: List[InvoiceExecution]


# ----------------------------------------------------
# Task Models
# ----------------------------------------------------

class HistoryEntry(BaseModel):
    role: Role
    parts: List[Part]


class Task(BaseModel):
    id: str
    contextId: str
    state: TaskState
    history: List[HistoryEntry] = Field(default_factory=list)
    artifacts: List[Part] = Field(default_factory=list)


class TaskList(BaseModel):
    tasks: List[Task]


# ----------------------------------------------------
# Agent Card
# ----------------------------------------------------

class AgentSkill(BaseModel):
    name: str
    description: str
    tags: List[str]


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    capabilities: Dict[str, Any]
    supportedInterfaces: List[Dict[str, Any]]
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[AgentSkill]