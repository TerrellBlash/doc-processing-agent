"""
Extraction schemas, one per document type.

The model's output is constrained to these schemas (structured output), so
extraction no longer depends on parsing free-form JSON out of a reply.

Every field is nullable and has no default, so the model has to state each
one, and must use null when the document doesn't contain it. The confidence
gate decides what a human reviews by counting missing fields, so a guessed
value is worse than a null.
"""

from pydantic import BaseModel, Field

ISO_DATE = "Full date in ISO 8601 (YYYY-MM-DD)"


class LineItem(BaseModel):
    description: str | None = Field(description="Item description as written")
    quantity: float | None = Field(description="Quantity as a number")
    unit_price: float | None = Field(description="Price per unit as a plain number")
    total: float | None = Field(description="Line total as a plain number")


class InvoiceFields(BaseModel):
    vendor_name: str | None = Field(description="Company that issued the invoice")
    invoice_number: str | None = Field(description="Invoice number exactly as printed")
    invoice_date: str | None = Field(description=f"Issue date. {ISO_DATE}")
    due_date: str | None = Field(description=f"Payment due date. {ISO_DATE}")
    line_items: list[LineItem] = Field(description="Every billed line; empty list if none")
    subtotal: float | None = Field(description="Amount before tax as a plain number")
    tax: float | None = Field(description="Total tax as a plain number")
    total_amount: float | None = Field(description="Amount due including tax as a plain number")
    currency: str | None = Field(description="ISO 4217 code, e.g. USD. A bare $ means USD")


class Experience(BaseModel):
    company: str | None = Field(description="Employer name")
    title: str | None = Field(description="Job title")
    start_date: str | None = Field(description="Start date as written, e.g. 2020 or Mar 2020")
    end_date: str | None = Field(description="End date as written, or Present")
    description: str | None = Field(description="What the role involved")


class Education(BaseModel):
    institution: str | None = Field(description="School or university name")
    degree: str | None = Field(description="Degree or certificate, e.g. B.S.")
    field: str | None = Field(description="Field of study")
    graduation_date: str | None = Field(description="Graduation date as written")


class ResumeFields(BaseModel):
    full_name: str | None = Field(description="Candidate's full name")
    email: str | None = Field(description="Email address")
    phone: str | None = Field(description="Phone number as written")
    location: str | None = Field(description="City and state or country")
    summary: str | None = Field(description="Professional summary as written")
    skills: list[str] = Field(description="Listed skills; empty list if none")
    experience: list[Experience] = Field(description="Every role listed; empty list if none")
    education: list[Education] = Field(description="Every degree or program; empty list if none")


class Party(BaseModel):
    name: str | None = Field(description="Legal name of the party")
    role: str | None = Field(description="Role in the agreement, e.g. Disclosing Party")


class ContractFields(BaseModel):
    contract_type: str | None = Field(description="e.g. NDA, Master Services Agreement")
    parties: list[Party] = Field(description="Every party to the agreement")
    effective_date: str | None = Field(description=f"Date the agreement takes effect. {ISO_DATE}")
    expiration_date: str | None = Field(description=f"Date the agreement ends. {ISO_DATE}")
    key_terms: list[str] = Field(description="Main obligations, one per item")
    payment_terms: str | None = Field(description="Payment amounts, schedule and terms")
    termination_clause: str | None = Field(description="How either party can end the agreement")
    governing_law: str | None = Field(description="Jurisdiction whose law governs, e.g. State of Delaware")


EXTRACTION_SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceFields,
    "resume": ResumeFields,
    "contract": ContractFields,
}

# The confidence gate checks exactly the fields the model is asked to fill.
REQUIRED_FIELDS: dict[str, list[str]] = {
    doc_type: list(schema.model_fields) for doc_type, schema in EXTRACTION_SCHEMAS.items()
}
