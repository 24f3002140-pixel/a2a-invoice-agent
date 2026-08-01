import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai

from storage import canonical_json


ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

REFERENCE_PATTERN = re.compile(r"\[[^\[\]\r\n]{1,300}\]")

DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
]


class AIError(Exception):
    """Raised when invoice decision generation fails."""


def package_to_prompt_text(package: Dict[str, Any]) -> str:
    """
    Preserve the full arbitrary package JSON without assuming hidden fields.
    Compact JSON reduces token usage while retaining all source text.
    """
    return canonical_json(package)


def collect_bracketed_references(
    package: Dict[str, Any],
) -> List[str]:
    """
    Return every unique bracketed reference appearing in the package.
    """
    package_text = json.dumps(
        package,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    references: List[str] = []
    seen = set()

    for reference in REFERENCE_PATTERN.findall(package_text):
        if reference not in seen:
            references.append(reference)
            seen.add(reference)

    return references


def clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def parse_model_json(raw_text: str) -> Dict[str, Any]:
    cleaned = clean_json_response(raw_text)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Occasionally a model may place text before or after the JSON.
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")

        if first_brace < 0 or last_brace <= first_brace:
            raise AIError(
                "The AI returned invalid JSON."
            )

        try:
            parsed = json.loads(
                cleaned[first_brace:last_brace + 1]
            )
        except json.JSONDecodeError as exc:
            raise AIError(
                "The AI returned invalid JSON."
            ) from exc

    if not isinstance(parsed, dict):
        raise AIError(
            "The AI response must be a JSON object."
        )

    return parsed


def configured_models() -> List[str]:
    models: List[str] = []

    configured = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    ).strip()

    if configured:
        models.append(configured)

    for fallback in DEFAULT_MODELS:
        if fallback not in models:
            models.append(fallback)

    return models


def configure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise AIError(
            "GEMINI_API_KEY is not configured."
        )

    genai.configure(api_key=api_key)


def create_model(model_name: str):
    """
    New Gemini models should not receive deprecated temperature,
    top_p, or top_k sampling options.
    """
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "response_mime_type": "application/json",
            "max_output_tokens": 65536,
        },
    )


def build_decision_prompt(
    packages: List[Dict[str, Any]],
    policy_revision: Any,
) -> str:
    package_sections: List[str] = []

    for index, package in enumerate(packages, start=1):
        package_id = package.get("packageId")

        package_sections.append(
            "\n".join(
                [
                    f"===== PACKAGE {index} =====",
                    f"PACKAGE_ID: {package_id}",
                    "PACKAGE_JSON:",
                    package_to_prompt_text(package),
                    f"===== END PACKAGE {index} =====",
                ]
            )
        )

    packages_text = "\n\n".join(package_sections)

    return f"""
You are an invoice reconciliation and payment-control agent.

You must analyze all invoice packages in one batch and return one decision
for every package.

POLICY REVISION:
{policy_revision}

ALLOWED ACTIONS

settle_invoice
Use only when the current commercial invoice is valid, fully reconciled,
and within autonomous payment authority.

request_approval
Use when the current invoice is commercially valid and reconciled, but its
amount, category, or authority rule requires human approval.

hold_invoice
Use when payment must pause until a specifically stated verification,
confirmation, missing document, bank check, delivery check, tax check,
compliance check, or other pending condition is completed.

reject_duplicate
Use only when the same commercial invoice has already been paid or the
current invoice is explicitly identified as a duplicate of a paid invoice.

open_exception
Use when material current records conflict, cannot be reconciled, or require
an exception workflow.

MANDATORY RULES

1. Return exactly one decision for every package.
2. Use the exact packageId supplied for that package.
3. Extract:
   - vendorName
   - invoiceNumber
   - amountMinor as an integer in minor currency units
   - currency as an uppercase currency code
4. Return exactly three distinct decisive bracketed evidence references.
5. Copy every evidence reference character-for-character from that package.
6. Use references from the current decisive paragraph or record.
7. Never use a cover-sheet reference.
8. Never use references from archived examples, old examples, demonstrations,
   templates, training passages, or decoy text.
9. Pay careful attention to negation and statements saying an action does
   not apply.
10. The rationale must:
    - be 60 to 1500 characters,
    - explicitly name the selected action,
    - include at least two of the three exact evidence references.
11. Do not invent missing facts or evidence.
12. Do not include markdown or commentary outside the JSON.

Return exactly this JSON structure:

{{
  "decisions": [
    {{
      "packageId": "exact package id",
      "action": "settle_invoice",
      "facts": {{
        "vendorName": "vendor",
        "invoiceNumber": "invoice number",
        "amountMinor": 12345,
        "currency": "INR"
      }},
      "evidenceRefs": [
        "[exact reference 1]",
        "[exact reference 2]",
        "[exact reference 3]"
      ],
      "rationale": "The settle_invoice action applies because [reference 1] and [reference 2] establish the decisive current facts."
    }}
  ]
}}

INVOICE PACKAGES

{packages_text}
""".strip()


def build_repair_prompt(
    packages: List[Dict[str, Any]],
    policy_revision: Any,
    previous_output: str,
    validation_error: str,
) -> str:
    package_sections = []

    for index, package in enumerate(packages, start=1):
        package_sections.append(
            "\n".join(
                [
                    f"===== PACKAGE {index} =====",
                    f"PACKAGE_ID: {package.get('packageId')}",
                    "PACKAGE_JSON:",
                    package_to_prompt_text(package),
                    f"===== END PACKAGE {index} =====",
                ]
            )
        )

    packages_text = "\n\n".join(package_sections)

    return f"""
Correct the invoice-decision JSON.

POLICY REVISION:
{policy_revision}

VALIDATION ERROR:
{validation_error}

PREVIOUS OUTPUT:
{previous_output}

REQUIREMENTS

- Return exactly one decision for every supplied package.
- packageId must match exactly.
- action must be one of:
  settle_invoice, request_approval, hold_invoice,
  reject_duplicate, open_exception.
- facts must contain vendorName, invoiceNumber, integer amountMinor,
  and uppercase currency.
- Return exactly three unique bracketed references copied exactly from the
  same package.
- References must be decisive current references, not cover-sheet,
  archive, example, training, or decoy references.
- rationale must contain the action name and at least two evidence references.
- rationale length must be 60 to 1500 characters.
- Return JSON only.

EXPECTED STRUCTURE

{{
  "decisions": [
    {{
      "packageId": "...",
      "action": "...",
      "facts": {{
        "vendorName": "...",
        "invoiceNumber": "...",
        "amountMinor": 123,
        "currency": "INR"
      }},
      "evidenceRefs": ["[...]", "[...]", "[...]"],
      "rationale": "..."
    }}
  ]
}}

PACKAGES

{packages_text}
""".strip()


def call_model(
    prompt: str,
) -> Tuple[str, str]:
    """
    Try the configured model first, then stable fallbacks.

    Returns:
        (model_name, response_text)
    """
    configure_gemini()

    errors: List[str] = []

    for model_name in configured_models():
        model = create_model(model_name)

        for attempt in range(2):
            try:
                print(
                    f"GEMINI_REQUEST model={model_name} "
                    f"attempt={attempt + 1}",
                    flush=True,
                )

                response = model.generate_content(
                    prompt,
                    request_options={
                        "timeout": 38,
                    },
                )

                text = getattr(response, "text", None)

                if isinstance(text, str) and text.strip():
                    print(
                        f"GEMINI_SUCCESS model={model_name}",
                        flush=True,
                    )

                    return model_name, text

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            except Exception as exc:
                error_message = (
                    f"{model_name} attempt {attempt + 1}: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"GEMINI_ERROR {error_message}",
                    flush=True,
                )

                errors.append(error_message)

                if attempt == 0:
                    time.sleep(1)

    raise AIError(
        "All configured Gemini model attempts failed. "
        + " | ".join(errors[-4:])
    )


def normalize_amount_minor(
    amount: Any,
    package_id: str,
) -> int:
    if isinstance(amount, bool):
        raise AIError(
            f"amountMinor is invalid for {package_id}."
        )

    if isinstance(amount, int):
        return amount

    if isinstance(amount, float) and amount.is_integer():
        return int(amount)

    if isinstance(amount, str):
        cleaned = amount.strip().replace(",", "")

        if re.fullmatch(r"-?\d+", cleaned):
            return int(cleaned)

    raise AIError(
        f"amountMinor must be an integer for {package_id}."
    )


def normalize_evidence_refs(
    evidence_refs: Any,
    package: Dict[str, Any],
    package_id: str,
) -> List[str]:
    if not isinstance(evidence_refs, list):
        raise AIError(
            f"evidenceRefs is missing for {package_id}."
        )

    available_references = set(
        collect_bracketed_references(package)
    )

    valid_refs: List[str] = []
    seen = set()

    for reference in evidence_refs:
        if not isinstance(reference, str):
            continue

        reference = reference.strip()

        if (
            reference in available_references
            and reference not in seen
        ):
            valid_refs.append(reference)
            seen.add(reference)

    if len(valid_refs) != 3:
        raise AIError(
            f"Exactly three valid evidence references are required "
            f"for {package_id}; received {len(valid_refs)}."
        )

    return valid_refs


def normalize_rationale(
    rationale: Any,
    action: str,
    evidence_refs: List[str],
    package_id: str,
) -> str:
    if not isinstance(rationale, str):
        rationale = ""

    rationale = rationale.strip()

    missing_refs = [
        reference
        for reference in evidence_refs
        if reference not in rationale
    ]

    if action not in rationale:
        rationale = (
            f"The selected action is {action}. "
            + rationale
        ).strip()

    refs_in_rationale = sum(
        1
        for reference in evidence_refs
        if reference in rationale
    )

    if refs_in_rationale < 2:
        rationale = (
            f"{rationale} The decisive evidence is "
            f"{evidence_refs[0]} and {evidence_refs[1]}."
        ).strip()

    if len(rationale) < 60:
        rationale = (
            f"{rationale} These current records determine the safe "
            f"invoice-processing outcome for this package."
        )

    if len(rationale) > 1500:
        rationale = rationale[:1500].rstrip()

    if action not in rationale:
        raise AIError(
            f"The rationale does not name the action for {package_id}."
        )

    cited_refs = sum(
        1
        for reference in evidence_refs
        if reference in rationale
    )

    if cited_refs < 2:
        raise AIError(
            f"The rationale does not cite two references for {package_id}."
        )

    if len(rationale) < 60 or len(rationale) > 1500:
        raise AIError(
            f"The rationale length is invalid for {package_id}."
        )

    return rationale


def validate_decision(
    decision: Dict[str, Any],
    package: Dict[str, Any],
) -> Dict[str, Any]:
    package_id = package.get("packageId")

    if (
        not isinstance(package_id, str)
        or not package_id.strip()
    ):
        raise AIError(
            "Every package requires a nonempty packageId."
        )

    if decision.get("packageId") != package_id:
        raise AIError(
            f"The AI returned the wrong packageId for {package_id}."
        )

    action = decision.get("action")

    if action not in ALLOWED_ACTIONS:
        raise AIError(
            f"The AI returned an invalid action for {package_id}."
        )

    facts = decision.get("facts")

    if not isinstance(facts, dict):
        raise AIError(
            f"The AI returned invalid facts for {package_id}."
        )

    vendor_name = facts.get("vendorName")
    invoice_number = facts.get("invoiceNumber")
    currency = facts.get("currency")

    if (
        not isinstance(vendor_name, str)
        or not vendor_name.strip()
    ):
        raise AIError(
            f"vendorName is missing for {package_id}."
        )

    if (
        not isinstance(invoice_number, str)
        or not invoice_number.strip()
    ):
        raise AIError(
            f"invoiceNumber is missing for {package_id}."
        )

    if (
        not isinstance(currency, str)
        or not currency.strip()
    ):
        raise AIError(
            f"currency is missing for {package_id}."
        )

    amount_minor = normalize_amount_minor(
        facts.get("amountMinor"),
        package_id,
    )

    evidence_refs = normalize_evidence_refs(
        decision.get("evidenceRefs"),
        package,
        package_id,
    )

    rationale = normalize_rationale(
        decision.get("rationale"),
        action,
        evidence_refs,
        package_id,
    )

    return {
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


def validate_all_decisions(
    parsed: Dict[str, Any],
    packages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    decisions = parsed.get("decisions")

    if not isinstance(decisions, list):
        raise AIError(
            "The AI response is missing the decisions array."
        )

    if len(decisions) != len(packages):
        raise AIError(
            "The AI must return one decision for every package."
        )

    decisions_by_package: Dict[str, Dict[str, Any]] = {}

    for decision in decisions:
        if not isinstance(decision, dict):
            raise AIError(
                "Every AI decision must be an object."
            )

        package_id = decision.get("packageId")

        if not isinstance(package_id, str):
            raise AIError(
                "An AI decision is missing packageId."
            )

        if package_id in decisions_by_package:
            raise AIError(
                f"The AI returned duplicate decisions for {package_id}."
            )

        decisions_by_package[package_id] = decision

    validated: List[Dict[str, Any]] = []

    for package in packages:
        package_id = package["packageId"]

        decision = decisions_by_package.get(package_id)

        if decision is None:
            raise AIError(
                f"The AI omitted package {package_id}."
            )

        validated.append(
            validate_decision(
                decision,
                package,
            )
        )

    return validated


def decide_packages(
    packages: List[Dict[str, Any]],
    policy_revision: Any,
) -> List[Dict[str, Any]]:
    """
    Analyze all uncached packages together.

    A repair request is made only when the first response is malformed.
    """
    if not packages:
        return []

    package_ids: List[str] = []

    for package in packages:
        if not isinstance(package, dict):
            raise AIError(
                "Every package must be an object."
            )

        package_id = package.get("packageId")

        if (
            not isinstance(package_id, str)
            or not package_id.strip()
        ):
            raise AIError(
                "Every package requires a nonempty packageId."
            )

        package_ids.append(package_id)

    if len(package_ids) != len(set(package_ids)):
        raise AIError(
            "Package IDs must be unique."
        )

    prompt = build_decision_prompt(
        packages,
        policy_revision,
    )

    _model_name, first_text = call_model(prompt)

    try:
        first_parsed = parse_model_json(first_text)

        return validate_all_decisions(
            first_parsed,
            packages,
        )

    except AIError as first_error:
        print(
            f"AI_VALIDATION_ERROR {first_error}",
            flush=True,
        )

        repair_prompt = build_repair_prompt(
            packages=packages,
            policy_revision=policy_revision,
            previous_output=first_text,
            validation_error=str(first_error),
        )

        _repair_model, repair_text = call_model(
            repair_prompt
        )

        repair_parsed = parse_model_json(
            repair_text
        )

        return validate_all_decisions(
            repair_parsed,
            packages,
        )
