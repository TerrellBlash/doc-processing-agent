"""
Turning uploaded files into text the pipeline can read.

PDF pages that carry a text layer are read directly: that text is exact and
costs nothing. Pages without one (scans, photos saved as PDF) are rendered
to an image and transcribed by the vision model. PNG and JPEG uploads go
straight to the vision model.

Every page records which of the two produced its text, so a reviewer can
tell machine-transcribed pages from text the PDF itself contained.

A vision page can also be read a second time by a different model, at the
same time as the first. The eval showed models misread poor scans (a 2026
read as 2025) and repeat the same misread when asked again, while a second
model disagrees. The page records whether the two reads agree and which
words differ, and the confidence gate sends a disagreement to review.

Uploads are untrusted, so every size is bounded before work is done: page
count before any page is read, render size before any page is drawn, and
pixel count before any image is decoded.
"""

import base64
import contextvars
import difflib
import io
import logging
import re
import threading
import unicodedata
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image, ImageOps

from app.llm import get_check_llm, get_llm, resolve_check_model

logger = logging.getLogger(__name__)

MIN_TEXT_LAYER_CHARS = 20  # less embedded text than this and the page is treated as a scan
MAX_PAGES = 20  # vision calls cost money; longer documents are refused up front
RENDER_SCALE = 2.0  # 144 DPI, enough for 8 pt print
MAX_IMAGE_SIDE = 2048  # larger pages and photos are scaled down; more pixels only add image tokens
MAX_IMAGE_PIXELS = 50_000_000  # covers 48 MP phone cameras; larger images are refused undecoded

MAX_PARALLEL_READS = 8  # vision calls in flight at once for one document
MAX_LISTED_DIFFERENCES = 20  # words kept per side when two reads disagree

_EXIF_ORIENTATION = 0x0112
# A model declining a page replies with a short, first-person apology about
# the request instead of the page's text. All three must hold, so a short
# letter ("I'm sorry for the delay", "Unfortunately, we are unable to help
# with your claim") still reads as a document.
_REFUSAL_OPENING = re.compile(
    r"^\W*(i['’]?m sorry|i am sorry|sorry, (i|but)|i apologi[sz]e"
    r"|i(['’]?m| am) (unable|not able)|i (can['’]?t|cannot|won['’]?t))\b",
    re.IGNORECASE,
)
_REFUSAL_SUBJECT = re.compile(
    r"\b(transcrib\w*|(assist|help) with (that|this)|comply|(this|that|the) (image|page|request|document))\b",
    re.IGNORECASE,
)
_REFUSAL_MAX_CHARS = 300
# Numbers keep their sign, currency and percent; words are any letters or
# digits in any script. Underscore rules and other punctuation are layout.
_TOKEN = re.compile(r"-?[$€£¥]?\d+(?:\.\d+)*%?|[^\W_]+")

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
class PageCheck:
    """How a second, independent read of a vision page compared with the first."""

    agrees: bool
    first_read_only: tuple[str, ...] = ()  # words and numbers only the first read has
    second_read_only: tuple[str, ...] = ()
    error: str | None = None  # the second read failed, so the page is unchecked

    def summary(self) -> dict:
        return {
            "agrees": self.agrees,
            "first_read_only": list(self.first_read_only),
            "second_read_only": list(self.second_read_only),
            "error": self.error,
        }


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    source: TextSource
    check: PageCheck | None = None  # vision pages only, when a check model is set


@dataclass(frozen=True)
class DocumentText:
    pages: tuple[PageText, ...]

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages if page.text)

    def page_summary(self) -> list[dict]:
        """Per-page provenance for the graph state: page, source, chars, and
        for vision pages that were read twice, the check."""
        summary = []
        for page in self.pages:
            entry: dict = {"page": page.page_number, "source": page.source, "chars": len(page.text)}
            if page.check is not None:
                entry["check"] = page.check.summary()
            summary.append(entry)
        return summary


@dataclass(frozen=True)
class _PdfPage:
    """What PDFium produced for one page: its text layer, or a rendered image."""

    page_number: int
    text: str  # empty when the page was rendered instead
    png: bytes | None


def document_to_text(
    data: bytes, transcribe: Transcriber, check: Transcriber | None = None
) -> DocumentText:
    """
    Read an uploaded file as text, page by page.

    - Parameter data: the file's bytes. The type comes from its leading
        bytes, not from the filename or the client's Content-Type.
    - Parameter transcribe: reads an image; vision_transcriber() in production.
    - Parameter check: reads each vision page a second time, at the same
        time as `transcribe`; check_transcriber() in production. None skips it.
    - Returns: the document's pages with the text and source of each, and
        the check for vision pages.
    - Throws: UnsupportedDocumentError for anything but PDF, PNG or JPEG;
        DocumentError for a file that can't be read, exceeds a limit, or
        has a page the vision model declined to transcribe.
    """
    if data.startswith(b"%PDF-"):
        return _pdf_to_text(data, transcribe, check)
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _image_to_text(data, "PNG", transcribe, check)
    if data.startswith(b"\xff\xd8\xff"):
        return _image_to_text(data, "JPEG", transcribe, check)
    raise UnsupportedDocumentError("Unsupported file type. Send a PDF, PNG or JPEG.")


def vision_transcriber(config: RunnableConfig | None = None) -> Transcriber:
    """
    Build a transcriber backed by the run's model.

    - Parameter config: the run config; selects the model like every node.
    - Returns: a function that sends one image to the model and returns its
        transcription. The client is built on first use, so a document that
        never needs vision never needs an API client.
    """
    return lambda image, mime_type: _transcribe(get_llm(config), image, mime_type)


def check_transcriber(config: RunnableConfig | None = None) -> Transcriber | None:
    """
    Build the second reader for vision pages.

    - Parameter config: the run config; picks the check model (see
        app.llm.resolve_check_model).
    - Returns: a transcriber backed by the check model, or None when the
        check is off or would use the run's own model. The client is built
        on first use.
    """
    if resolve_check_model(config) is None:
        return None

    def transcribe(image: bytes, mime_type: str) -> str:
        client = get_check_llm(config)
        if client is None:  # configuration changed since this was built
            raise RuntimeError("no check model configured")
        return _transcribe(client, image, mime_type)

    return transcribe


def _transcribe(client, image: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(image).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": TRANSCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ]
    )
    return client.invoke([message]).text


def _pdf_to_text(data: bytes, transcribe: Transcriber, check: Transcriber | None) -> DocumentText:
    pdf_pages = _read_pdf(data)
    images = [(page.page_number, page.png, "image/png") for page in pdf_pages if page.png is not None]
    read = _read_images(images, transcribe, check)
    pages = tuple(
        PageText(page.page_number, page.text, "text_layer")
        if page.png is None
        else PageText(page.page_number, read[page.page_number][0], "vision", read[page.page_number][1])
        for page in pdf_pages
    )
    return DocumentText(pages=pages)


def _read_images(
    images: list[tuple[int, bytes, str]], transcribe: Transcriber, check: Transcriber | None
) -> dict[int, tuple[str, PageCheck | None]]:
    """
    Read every image, and with a check model read each one again, all at once.

    - Returns: page number to (text, check).
    - Throws: DocumentError for a page the main reader declined. A failed
        first read propagates right away, cancelling reads not yet started;
        a failed second read only marks the page unchecked.
    """
    if not images:
        return {}
    pool = ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_READS, len(images) * (2 if check else 1)))

    def submit(read: Transcriber, image: bytes, mime: str) -> Future:
        # Each read runs in a copy of the caller's context, so callbacks
        # held in context variables (token usage, tracing) see it.
        return pool.submit(contextvars.copy_context().run, read, image, mime)

    try:
        firsts = {number: submit(transcribe, image, mime) for number, image, mime in images}
        seconds = {number: submit(check, image, mime) for number, image, mime in images} if check else {}
        results = {}
        for number, first in firsts.items():
            text = _clean(first.result())
            if _is_refusal(text):
                raise DocumentError(
                    f"Page {number} couldn't be read: the vision model declined to transcribe it."
                )
            results[number] = (text, _compare(number, text, seconds[number]) if check else None)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results


def _compare(number: int, first: str, second_read: Future) -> PageCheck:
    try:
        second = _clean(second_read.result())
    except Exception as error:  # an unchecked page goes to review, never through
        logger.warning("Second read of page %d failed: %r", number, error)
        return PageCheck(agrees=False, error=type(error).__name__)
    if _is_refusal(second):
        return PageCheck(agrees=False, error="the check model declined to transcribe the page")
    ours, theirs = _tokens(first), _tokens(second)
    first_only: list[str] = []
    second_only: list[str] = []
    # In order, so swapped values (day and month, two amounts) count too.
    matcher = difflib.SequenceMatcher(a=ours, b=theirs, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            first_only += ours[i1:i2]
            second_only += theirs[j1:j2]
    agrees = not first_only and not second_only
    return PageCheck(agrees=agrees, first_read_only=tuple(first_only[:MAX_LISTED_DIFFERENCES]),
                     second_read_only=tuple(second_only[:MAX_LISTED_DIFFERENCES]))


def _tokens(text: str) -> list[str]:
    """Words and numbers in reading order, ignoring case, layout, punctuation
    and thousands commas."""
    text = unicodedata.normalize("NFKC", text).casefold().replace("−", "-")
    return _TOKEN.findall(re.sub(r"(?<=\d),(?=\d)", "", text))


def _is_refusal(text: str) -> bool:
    return (len(text) <= _REFUSAL_MAX_CHARS and _REFUSAL_OPENING.search(text) is not None
            and _REFUSAL_SUBJECT.search(text) is not None)


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


def _image_to_text(
    data: bytes, fmt: str, transcribe: Transcriber, check: Transcriber | None
) -> DocumentText:
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
    text, page_check = _read_images([(1, data, f"image/{fmt.lower()}")], transcribe, check)[1]
    return DocumentText(pages=(PageText(1, text, "vision", page_check),))


def _clean(text: str) -> str:
    """Normalize line endings and trailing spaces; PDF text layers use CRLF."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()
