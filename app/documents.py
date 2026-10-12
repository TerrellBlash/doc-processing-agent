"""
Turning uploaded files into text the pipeline can read.

PDF pages that carry a text layer are read directly: that text is exact and
costs nothing. Pages without one (scans, photos saved as PDF) are rendered
to an image and transcribed by the vision model. PNG and JPEG uploads go
straight to the vision model.

Every page records which of the two produced its text, so a reviewer can
tell machine-transcribed pages from text the PDF itself contained.
"""

import base64
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image

from app.llm import get_llm

MIN_TEXT_LAYER_CHARS = 20  # less embedded text than this and the page is treated as a scan
MAX_PAGES = 20  # vision calls cost money; longer documents are refused up front
RENDER_SCALE = 2.0  # 144 DPI, enough for 8 pt print
MAX_IMAGE_SIDE = 2048  # larger photos are downscaled; more pixels only add image tokens

TRANSCRIBE_PROMPT = (
    "Transcribe all text on this page exactly as written, top to bottom. Keep "
    "line breaks and the column order of tables. Do not summarize, correct, "
    "translate or add anything. Return plain text only, no markdown."
)

TextSource = Literal["text_layer", "vision"]

# (image bytes, MIME type) -> transcribed text
Transcriber = Callable[[bytes, str], str]


class DocumentError(ValueError):
    """The upload can't be turned into text."""


class UnsupportedDocumentError(DocumentError):
    """The upload isn't a PDF, PNG or JPEG."""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    source: TextSource


@dataclass(frozen=True)
class DocumentText:
    pages: tuple[PageText, ...]

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages if page.text)

    def page_summary(self) -> list[dict]:
        """Per-page provenance for the graph state: page, source, chars."""
        return [
            {"page": page.page_number, "source": page.source, "chars": len(page.text)}
            for page in self.pages
        ]


def document_to_text(data: bytes, transcribe: Transcriber) -> DocumentText:
    """
    Read an uploaded file as text, page by page.

    - Parameter data: the file's bytes. The type comes from its leading
        bytes, not from the filename or the client's Content-Type.
    - Parameter transcribe: reads an image; vision_transcriber() in production.
    - Returns: the document's pages with the text and source of each.
    - Throws: UnsupportedDocumentError for anything but PDF, PNG or JPEG;
        DocumentError for a file that can't be opened or is too long.
    """
    if data.startswith(b"%PDF-"):
        return _pdf_to_text(data, transcribe)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _image_to_text(data, "PNG", transcribe)
    if data.startswith(b"\xff\xd8\xff"):
        return _image_to_text(data, "JPEG", transcribe)
    raise UnsupportedDocumentError("Unsupported file type. Send a PDF, PNG or JPEG.")


def vision_transcriber(config: RunnableConfig | None = None) -> Transcriber:
    """
    Build a transcriber backed by the run's model.

    - Parameter config: the run config; selects the model like every node.
    - Returns: a function that sends one image to the model and returns its
        transcription.
    """
    llm = get_llm(config)

    def transcribe(image: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": TRANSCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ]
        )
        return llm.invoke([message]).text

    return transcribe


def _pdf_to_text(data: bytes, transcribe: Transcriber) -> DocumentText:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as error:
        raise DocumentError(f"Could not open the PDF: {error}") from error
    try:
        if len(pdf) == 0:
            raise DocumentError("The PDF has no pages.")
        if len(pdf) > MAX_PAGES:
            raise DocumentError(f"The PDF has {len(pdf)} pages; the limit is {MAX_PAGES}.")
        return DocumentText(
            pages=tuple(_read_page(pdf[index], index + 1, transcribe) for index in range(len(pdf)))
        )
    finally:
        pdf.close()


def _read_page(page: pdfium.PdfPage, page_number: int, transcribe: Transcriber) -> PageText:
    try:
        textpage = page.get_textpage()
        try:
            text = _clean(textpage.get_text_bounded())
        finally:
            textpage.close()
        if len(text) >= MIN_TEXT_LAYER_CHARS:
            return PageText(page_number, text, "text_layer")

        buffer = io.BytesIO()
        # pypdfium2 takes a float scale; pyright infers int from its default.
        page.render(scale=RENDER_SCALE).to_pil().save(buffer, format="PNG")  # pyright: ignore[reportArgumentType]
        return PageText(page_number, _clean(transcribe(buffer.getvalue(), "image/png")), "vision")
    finally:
        page.close()


def _image_to_text(data: bytes, fmt: str, transcribe: Transcriber) -> DocumentText:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except OSError as error:
        raise DocumentError(f"Could not open the image: {error}") from error
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        data = buffer.getvalue()
    text = _clean(transcribe(data, f"image/{fmt.lower()}"))
    return DocumentText(pages=(PageText(1, text, "vision"),))


def _clean(text: str) -> str:
    """Normalize line endings and trailing spaces; PDF text layers use CRLF."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()
