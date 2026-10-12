"""
Build PDFs for tests and the evaluation dataset, without a PDF library.

text_pdf() writes a real text-layer PDF (Helvetica, US Letter), like the
ones billing and HR systems export. scanned_pdf() renders that PDF to
grayscale images, roughens them the way a desk scanner does (slight skew,
blur, JPEG artifacts) and wraps them in a PDF with no text layer, so the
only way to read it is a vision model.
"""

import io
import random
import textwrap

import pypdfium2 as pdfium
from PIL import Image, ImageFilter

PAGE_WIDTH, PAGE_HEIGHT = 612, 792  # US Letter, in points
MARGIN = 72
FONT_SIZE = 10
LEADING = 13
LINES_PER_PAGE = (PAGE_HEIGHT - 2 * MARGIN) // LEADING
WRAP_COLUMNS = 90  # fits the text width at 10 pt Helvetica

SCAN_DPI = 115
JPEG_QUALITY = 50


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _layout(text: str) -> list[list[str]]:
    """Wrap long lines, keep blank ones, and split into pages."""
    lines: list[str] = []
    for line in text.splitlines():
        # The standard Helvetica font only covers WinAnsi, so keep to ASCII.
        line = line.encode("ascii", "replace").decode()
        lines.extend(textwrap.wrap(line, WRAP_COLUMNS) or [""])
    lines = lines or [""]
    return [lines[i : i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]


def text_pdf(text: str) -> bytes:
    """
    Write `text` into a PDF with a real text layer.

    - Parameter text: plain text; long lines wrap, and pages break every
        LINES_PER_PAGE lines.
    - Returns: the PDF file as bytes.
    """
    pages = _layout(text)
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    }
    kids = []
    for index, page_lines in enumerate(pages):
        page_id, content_id = 4 + 2 * index, 5 + 2 * index
        kids.append(f"{page_id} 0 R")
        ops = [f"BT /F1 {FONT_SIZE} Tf {LEADING} TL {MARGIN} {PAGE_HEIGHT - MARGIN} Td"]
        ops += [f"({_escape(line)}) Tj T*" for line in page_lines]
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode()
        objects[content_id] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = {}
    for number in sorted(objects):
        offsets[number] = out.tell()
        out.write(b"%d 0 obj\n%s\nendobj\n" % (number, objects[number]))
    xref_at = out.tell()
    size = max(objects) + 1
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % size)
    for number in range(1, size):
        out.write(b"%010d 00000 n \n" % offsets[number])
    out.write(b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref_at))
    return out.getvalue()


def _scan_page(page: pdfium.PdfPage, rng: random.Random) -> Image.Image:
    # pypdfium2 takes a float scale; pyright infers int from its default.
    image = page.render(scale=SCAN_DPI / 72).to_pil().convert("L")  # pyright: ignore[reportArgumentType]
    image = image.rotate(rng.uniform(-1.0, 1.0), resample=Image.Resampling.BICUBIC, fillcolor=255)
    image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))
    # Lift the blacks a little, like toner on a worn drum.
    image = image.point(lambda value: 25 + value * 0.9)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Image.open(io.BytesIO(buffer.getvalue()))


def scanned_pdf(text: str, seed: int = 0) -> bytes:
    """
    Render `text` as a scanned document: images only, no text layer.

    - Parameter text: plain text, laid out the same way as text_pdf().
    - Parameter seed: makes the skew and blur reproducible.
    - Returns: the PDF file as bytes.
    """
    rng = random.Random(seed)
    source = pdfium.PdfDocument(text_pdf(text))
    try:
        images = [_scan_page(source[index], rng) for index in range(len(source))]
    finally:
        source.close()
    out = io.BytesIO()
    images[0].save(
        out, format="PDF", save_all=True, append_images=images[1:], resolution=SCAN_DPI
    )
    return out.getvalue()
