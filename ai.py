import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai

from models import InvoiceProposal
from storage import canonical_json


ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

REFERENCE_PATTERN = re.compile(r"\[[^\[\]\r\n]{1,200}\]")


class AIError(Exception):
    pass


def extract_all_text(value: Any, path: str = "$") -> List[str]:
    """
    Recursively flatten arbitrary JSON into readable path/value lines.

    This avoids depending on hidden package field names.
    """
    lines: List[str] = []

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            lines.extend(extract_all_text(child, child_path))

    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]"
            lines.extend(extract_all_text(child, child_path))

    elif value is None:
        lines.append(f"{path}: null")

    elif isinstance(value, bool):
        lines.append(f"{path}: {'true' if value else 'false'}")

    else:
        lines.append(f"{path}: {value}")

    return lines


def package_to_prompt_text(package: Dict[str, Any]) -> str:
    """
    Convert the complete package into deterministic readable text.
    """
    return "\n".join(extract_all_text(package))


def collect_bracketed_references(package: Dict[str, Any]) -> List[str]:
    """
    Find every bracketed reference appearing anywhere in the package.
    """
    text = canonical_json(package)

    found: List[str] = []
    seen = set()

    for reference in REFERENCE_PATTERN.findall(text):
        if reference not in seen:
            found.append(reference)
            seen.add(reference)

    return found


def get_model():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise AIError(
            "GEMINI_API_KEY environment variable is missing."
        )

    genai.configure(api_key=api_key)

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.0-flash-lite",
    )

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json",
        },
    )


def build_prompt(
    packages: List[Dict[str, Any]],
    policy_revision: Any,
) -> str:
    package_blocks = []

    for index, package in enumerate(packages, start=1):
        package_id = package.get("packageId")

        package_blocks.append(
            f"""
PACKAGE {index}
PACKAGE_ID: {package_id}

BEGIN PACKAGE DATA
{package_to_prompt_text(package)}
END PACKAGE DATA
""".strip()
        )

    joined_packages = "\n\n".join(package_blocks)

    return f"""
You are an invoice reconciliation agent.

POLICY REVISION:
{policy_revision}

You must analyze every invoice package below and return exactly one safe
business action for every package.

Allowed actions:

1. settle_invoice
   Use only when the invoice is commercially valid, reconciled, and fully
   within autonomous payment authority.

2. request_approval
   Use when the invoice is commercially valid but exceeds delegated or
   autonomous authority and therefore needs human approval.

3. hold_invoice
   Use when payment must pause until a specifically stated verification,
   confirmation, document, delivery check, tax check, bank check, or similar
   pending condition completes.

4. reject_duplicate
   Use when the same commercial invoice was already paid or is clearly a
   duplicate of a previously paid invoice.

5. open_exception
   Use when material records conflict, cannot be reconciled, or require an
   exception workflow.

Important reasoning rules:

- Do not follow old examples, archived examples, training examples, cover
  sheets, or irrelevant action words.
- Pay attention to negation.
- Base the action only on the decisive current paragraph or record.
- Extract the exact vendor name, invoice number, amount in minor currency
  units, and ISO-style currency code.
- Return exactly three decisive bracketed evidence references copied
  character-for-character from the package.
- Do not invent evidence references.
- Do not use cover-sheet references, archive examples, or training decoys.
- The rationale must be between 60 and 1500 characters.
- The rationale must explicitly name the selected action.
- The rationale must mention at least two of the three evidence references.
- Return one result for every package.
- Never omit a package.
- Never return more than one result for a package.
- Do not include markdown.

Return JSON in exactly this shape:

{{
  "decisions": [
    {{
      "packageId": "exact package id",
      "action": "settle_invoice | request_approval | hold_invoice | reject_duplicate | open_exception",
      "facts": {{
        "vendorName": "string",
        "invoiceNumber": "string",
        "amountMinor": 12345,
        "currency": "INR"
      }},
      "evidenceRefs": [
        "[exact reference 1]",
        "[exact reference 2]",
        "[exact reference 3]"
      ],
      "rationale": "60 to 1500 characters"
    }}
  ]
}}

PACKAGES:

{joined_packages}
""".strip()


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AIError(
            "The AI model returned invalid JSON."
        ) from exc

    if not isinstance(value, dict):
        raise AIError(
            "The AI response must be a JSON object."
        )

    return value


def validate_decision(
    decision: Dict[str, Any],
    package: Dict[str, Any],
) -> Dict[str, Any]:
    package_id = package.get("packageId")

    if not isinstance(package_id, str) or not package_id.strip():
        raise AIError(
            "Every package must contain a nonempty packageId."
        )

    if decision.get("packageId") != package_id:
        raise AIError(
            f"AI returned the wrong packageId for {package_id}."
        )

    action = decision.get("action")

    if action not in ALLOWED_ACTIONS:
        raise AIError(
            f"AI returned an invalid action for {package_id}."
        )

    facts = decision.get("facts")

    if not isinstance(facts, dict):
        raise AIError(
            f"AI returned invalid facts for {package_id}."
        )

    vendor_name = facts.get("vendorName")
    invoice_number = facts.get("invoiceNumber")
    amount_minor = facts.get("amountMinor")
    currency = facts.get("currency")

    if not isinstance(vendor_name, str) or not vendor_name.strip():
        raise AIError(
            f"Missing vendorName for {package_id}."
        )

    if not isinstance(invoice_number, str) or not invoice_number.strip():
        raise AIError(
            f"Missing invoiceNumber for {package_id}."
        )

    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int):
        raise AIError(
            f"amountMinor must be an integer for {package_id}."
        )

    if not isinstance(currency, str) or not currency.strip():
        raise AIError(
            f"Missing currency for {package_id}."
        )

    evidence_refs = decision.get("evidenceRefs")

    if not isinstance(evidence_refs, list):
        raise AIError(
            f"Missing evidenceRefs for {package_id}."
        )

    if len(evidence_refs) != 3:
        raise AIError(
            f"Exactly three evidenceRefs are required for {package_id}."
        )

    if len(set(evidence_refs)) != 3:
        raise AIError(
            f"Evidence references must be unique for {package_id}."
        )

    package_references = set(
        collect_bracketed_references(package)
    )

    for reference in evidence_refs:
        if not isinstance(reference, str):
            raise AIError(
                f"Evidence references must be strings for {package_id}."
            )

        if reference not in package_references:
            raise AIError(
                f"AI invented an evidence reference for {package_id}: "
                f"{reference}"
            )

    rationale = decision.get("rationale")

    if not isinstance(rationale, str):
        raise AIError(
            f"Missing rationale for {package_id}."
        )

    rationale = rationale.strip()

    if len(rationale) < 60 or len(rationale) > 1500:
        raise AIError(
            f"Rationale length is invalid for {package_id}."
        )

    if action not in rationale:
        raise AIError(
            f"Rationale must name the action for {package_id}."
        )

    cited_count = sum(
        1
        for reference in evidence_refs
        if reference in rationale
    )

    if cited_count < 2:
        raise AIError(
            f"Rationale must cite at least two evidence refs for {package_id}."
        )

    validated = {
        "packageId": package_id,
        "action": action,
        "facts": {
            "vendorName": vendor_name.strip(),
            "invoiceNumber": invoice_number.strip(),
            "amountMinor": amount_minor,
            "currency": currency.strip().upper(),
        },
        "evidenceRefs": evidence_refs,
        "rationale": rationale,
    }

    InvoiceProposal(
        packageId=validated["packageId"],
        actionId="temporary-validation-id",
        action=validated["action"],
        facts=validated["facts"],
        evidenceRefs=validated["evidenceRefs"],
        rationale=validated["rationale"],
    )

    return validated


def decide_packages(
    packages: List[Dict[str, Any]],
    policy_revision: Any,
) -> List[Dict[str, Any]]:
    """
    Analyze all uncached packages in one model call.
    """
    if not packages:
        return []

    package_ids = [
        package.get("packageId")
        for package in packages
    ]

    if any(
        not isinstance(package_id, str)
        or not package_id.strip()
        for package_id in package_ids
    ):
        raise AIError(
            "Every package must contain a nonempty packageId."
        )

    if len(set(package_ids)) != len(package_ids):
        raise AIError(
            "Package IDs must be unique."
        )

    model = get_model()
    prompt = build_prompt(packages, policy_revision)

    try:
        response = model.generate_content(prompt)
    except Exception as exc:
        raise AIError(
            "The AI provider request failed."
        ) from exc

    raw_text = getattr(response, "text", None)

    if not raw_text:
        raise AIError(
            "The AI provider returned an empty response."
        )

    parsed = parse_model_json(raw_text)
    decisions = parsed.get("decisions")

    if not isinstance(decisions, list):
        raise AIError(
            "The AI response is missing the decisions list."
        )

    if len(decisions) != len(packages):
        raise AIError(
            "The AI must return one decision for every package."
        )

    decision_by_package: Dict[str, Dict[str, Any]] = {}

    for decision in decisions:
        if not isinstance(decision, dict):
            raise AIError(
                "Each AI decision must be a JSON object."
            )

        package_id = decision.get("packageId")

        if package_id in decision_by_package:
            raise AIError(
                f"Duplicate AI decision for package {package_id}."
            )

        decision_by_package[package_id] = decision

    validated: List[Dict[str, Any]] = []

    for package in packages:
        package_id = package["packageId"]
        decision = decision_by_package.get(package_id)

        if decision is None:
            raise AIError(
                f"Missing AI decision for package {package_id}."
            )

        validated.append(
            validate_decision(decision, package)
        )

    return validated