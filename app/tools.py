"""
TOOLS — Things the LLM Can Call
=================================
A tool is a function the LLM can choose to invoke to get information
it does not already have. The LLM does not run the code itself; it
emits a request to call the tool, our node runs the function, and the
result goes back to the LLM.

This file holds one tool: vendor_lookup. It is a deterministic fake
(no network) so the behavior is reproducible for course work. In a
real system this would hit a vendor database or an external API.

Used by the enrich_extraction node (Unit 2). The LLM decides whether
a given document warrants a vendor lookup; invoices usually do,
resumes do not.
"""

from langchain_core.tools import tool


# A tiny fake "vendor database." Keys are lowercased vendor names.
_FAKE_VENDOR_DB = {
    "acme corp": {
        "address": "100 Industrial Way, Springfield, IL 62701",
        "tax_id": "12-3456789",
        "payment_history": "on_time",
    },
    "globex llc": {
        "address": "42 Galaxy Blvd, Austin, TX 78701",
        "tax_id": "98-7654321",
        "payment_history": "late_2_invoices",
    },
    "initech": {
        "address": "7 Cubicle Lane, Dallas, TX 75201",
        "tax_id": "55-5555555",
        "payment_history": "on_time",
    },
}


@tool
def vendor_lookup(vendor_name: str) -> dict:
    """Look up a vendor's address, tax ID, and payment history by name.

    Use this when processing an invoice or similar document where a
    vendor or supplier is named and additional vendor details would
    enrich the extracted data. Do not use it for documents that have
    no vendor, such as resumes.

    Args:
        vendor_name: The vendor or company name to look up.

    Returns:
        A dict with address, tax_id, and payment_history. If the vendor
        is unknown, returns a record marked as not found.
    """
    key = vendor_name.strip().lower()
    record = _FAKE_VENDOR_DB.get(key)
    if record is None:
        return {
            "address": None,
            "tax_id": None,
            "payment_history": "unknown",
            "found": False,
        }
    return {**record, "found": True}
