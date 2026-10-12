"""
Turning uploaded files into text the pipeline can read.

PDF pages that carry a text layer are read directly: that text is exact and
costs nothing. Pages without one (scans, photos saved as PDF) are rendered
to an image and transcribed by the vision model. PNG and JPEG uploads go
straight to the vision model.

Every page records which of the two produced its text, so a reviewer can
tell machine-transcribed pages from text the PDF itself contained.

Uploads are untrusted, so every size is bounded before work is done: page
count before any page is read, render size before any page is drawn, and
pixel count before any image is decoded.
"""

import base64
import io
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image, ImageOps

from app.llm import get_llm

MIN_TEXT_LAYER_CHARS = 20  # less embedded text than this and the page is treated as a scan
MAX_PAGES = 20  # vision calls cost money; longer documents are refused up front
RENDER_SCALE = 2.0  # 144 DPI, enough for 8 pt print
MAX_IMAGE_SIDE = 2048  # larger pages and photos are scaled down; more pixels only add image tokens
MAX_IMAGE_PIXELS = 50_000_000  # covers 48 MP phone cameras; larger images are refused undecoded

_EXIF_ORIENTATION = 0x0112

# PDFium is not thread-safe, and uploads are read on FastAPI's worker
# threads. Every PDFium call happens under this lock. The vision calls,
# which take seconds, happen after it is released.
_PDFIUM_LOCK = threading.Lock()

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


@dataclass(frozen=True)
class _PdfPage:
    """What PDFium produced for one page: its text layer, or a rendered image."""

    page_number: int
    text: str  # empty when the page was rendered instead
    png: bytes | None


def document_to_text(data: bytes, transcribe: Transcriber) -> DocumentText:
    """
    Read an uploaded file as text, page by page.

    - Parameter data: the file's bytes. The type comes from its leading
        bytes, not from the filename or the client's Content-Type.
    - Parameter transcribe: reads an image; vision_transcriber() in production.
    - Returns: the document's pages with the text and source of each.
    - Throws: UnsupportedDocumentError for anything but PDF, PNG or JPEG;
        DocumentError for a file that can't be read or exceeds a limit.
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
        transcription. The client is built on first use, so a document that
        never needs vision never needs an API client.
    """

    def transcribe(image: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(image).decode()
        message = HumanMessage(
            content=[
                {"type": "text", "text": TRANSCRIBE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ]
        )
        return get_llm(config).invoke([message]).text

    return transcribe


def _pdf_to_text(data: bytes, transcribe: Transcriber) -> DocumentText:
    pages = tuple(
        PageText(page.page_number, page.text, "text_layer")
        if page.png is None
        else PageText(page.page_number, _clean(transcribe(page.png, "image/png")), "vision")
        for page in _read_pdf(data)
    )
    return DocumentText(pages=pages)


def _read_pdf(data: bytes) -> list[_PdfPage]:
    """All PDFium work for one document, under the lock."""
    with _PDFIUM_LOCK:
        try:
            pdf = pdfium.PdfDocument(data)
        except pdfium.PdfiumError as error:
            raise DocumentError("Could not open the PDF.") from error
        try:
            page_count = len(pdf)
            if page_count == 0:
                raise DocumentError("The PDF has no pages.")
            if page_count > MAX_PAGES:
                raise DocumentError(f"The PDF has {page_count} pages; the limit is {MAX_PAGES}.")
            return [_read_page(pdf, index) for index in range(page_count)]
        except pdfium.PdfiumError as error:
            raise DocumentError("Could not read the PDF.") from error
        finally:
            pdf.close()


def _read_page(pdf: pdfium.PdfDocument, index: int) -> _PdfPage:
    page = pdf[index]
    try:
        textpage = page.get_textpage()
        try:
            text = _clean(textpage.get_text_bounded())
        finally:
            textpage.close()
        if len(text) >= MIN_TEXT_LAYER_CHARS:
            return _PdfPage(index + 1, text, None)

        # A page can declare any size, so cap the render instead of trusting it.
        width, height = page.get_size()
        scale = min(RENDER_SCALE, MAX_IMAGE_SIDE / max(width, height, 1))
        bitmap = page.render(scale=scale)  # pyright: ignore[reportArgumentType]
        try:
            buffer = io.BytesIO()
            bitmap.to_pil().save(buffer, format="PNG")
        finally:
            bitmap.close()
        return _PdfPage(index + 1, "", buffer.getvalue())
    finally:
        page.close()


def _image_to_text(data: bytes, fmt: str, transcribe: Transcriber) -> DocumentText:
    try:
        image = Image.open(io.BytesIO(data))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise DocumentError("The image is larger than 50 megapixels.")
        if fmt == "JPEG":
            image.draft(image.mode, (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))  # decode at reduced size
        image.load()
    except (OSError, Image.DecompressionBombError) as error:
        raise DocumentError("Could not open the image.") from error

    # Phone photos are often stored sideways with an EXIF tag saying how to
    # turn them. Re-encoding drops the tag, so apply it first.
    rotated = image.getexif().get(_EXIF_ORIENTATION, 1) != 1
    if rotated:
        image = ImageOps.exif_transpose(image)
    if rotated or max(image.size) > MAX_IMAGE_SIDE:
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
