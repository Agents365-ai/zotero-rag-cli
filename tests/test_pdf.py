from pathlib import Path
from unittest.mock import patch, MagicMock

from rak.pdf import chunk_text, extract_pdf_text, extract_file_text, find_attachments, _split_paragraphs

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_pdf_text():
    text = extract_pdf_text(FIXTURES / "sample.pdf")
    assert "test PDF document" in text
    assert len(text) > 10


def test_extract_pdf_text_missing_file():
    text = extract_pdf_text(Path("/nonexistent/file.pdf"))
    assert text == ""


def test_find_attachments_pdf_and_md(tmp_path: Path):
    key_dir = tmp_path / "ABC12345"
    key_dir.mkdir()
    pdf_file = key_dir / "Smith 2024 - Paper.pdf"
    pdf_file.touch()
    md_file = key_dir / "notes.md"
    md_file.write_text("some notes")
    result = find_attachments(tmp_path, "ABC12345")
    assert len(result) == 2
    assert pdf_file in result
    assert md_file in result


def test_find_attachments_not_found(tmp_path: Path):
    result = find_attachments(tmp_path, "NONEXIST")
    assert result == []


def test_find_attachments_ignores_other_files(tmp_path: Path):
    key_dir = tmp_path / "ABC12345"
    key_dir.mkdir()
    (key_dir / "snapshot.html").touch()
    (key_dir / "image.png").touch()
    result = find_attachments(tmp_path, "ABC12345")
    assert result == []


def test_extract_file_text_md(tmp_path: Path):
    md = tmp_path / "note.md"
    md.write_text("# Title\nSome content here.")
    text = extract_file_text(md)
    assert "# Title" in text
    assert "Some content here." in text


def test_extract_file_text_unsupported(tmp_path: Path):
    txt = tmp_path / "readme.txt"
    txt.write_text("hello")
    assert extract_file_text(txt) == ""


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_short():
    text = "hello world this is a short text"
    chunks = chunk_text(text, chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits():
    words = [f"word{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    # Each chunk (except possibly last) should have ~30 words
    first_words = chunks[0].split()
    assert len(first_words) == 30
    # Overlap: second chunk should start 20 words in
    second_words = chunks[1].split()
    assert second_words[0] == "word20"


def test_chunk_text_overlap_content():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    # Last words of chunk 0 should appear at start of chunk 1
    c0_words = chunks[0].split()
    c1_words = chunks[1].split()
    assert c0_words[-5:] == c1_words[:5]


def test_split_paragraphs_double_newline():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paras = _split_paragraphs(text)
    assert len(paras) == 3
    assert paras[0] == "First paragraph."
    assert paras[1] == "Second paragraph."


def test_split_paragraphs_markdown_heading():
    text = "Intro text.\n\n## Methods\n\nMethod details.\n\n## Results\n\nResult details."
    paras = _split_paragraphs(text)
    assert any("Methods" in p for p in paras)
    assert any("Results" in p for p in paras)


def test_chunk_text_respects_paragraphs():
    """Paragraphs that fit within chunk_size should be merged, not split mid-sentence."""
    para1 = " ".join(f"a{i}" for i in range(10))  # 10 words
    para2 = " ".join(f"b{i}" for i in range(10))  # 10 words
    para3 = " ".join(f"c{i}" for i in range(10))  # 10 words
    para4 = " ".join(f"d{i}" for i in range(10))  # 10 words
    text = f"{para1}\n\n{para2}\n\n{para3}\n\n{para4}"
    chunks = chunk_text(text, chunk_size=25, overlap=5)
    # Should merge paras into 2 chunks (10+10=20 fits, 10+10=20 fits) not split mid-para
    assert len(chunks) == 2
    assert "a0" in chunks[0]
    assert "b0" in chunks[0]
    assert "c0" in chunks[1]
    assert "d0" in chunks[1]


def test_chunk_text_oversized_paragraph():
    """A single oversized paragraph should be split with word-level overlap."""
    big_para = " ".join(f"x{i}" for i in range(50))
    small_para = "short paragraph here"
    text = f"{big_para}\n\n{small_para}"
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) >= 3  # big para splits into ~3 chunks + small para
    assert "short paragraph here" in chunks[-1]


def test_chunk_text_no_paragraph_structure():
    """Plain text without paragraph breaks falls back to word-level chunking."""
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 1
    first_words = chunks[0].split()
    assert len(first_words) == 30


def test_extract_pdf_text_mineru_provider(tmp_path: Path):
    """MinerU provider calls subprocess and reads markdown output."""
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    def fake_run(cmd, **kwargs):
        # Simulate mineru writing a markdown file
        # The output dir is the last argument
        output_dir = Path(cmd[-1])
        output_dir.mkdir(parents=True, exist_ok=True)
        auto_dir = output_dir / pdf_file.stem / "auto"
        auto_dir.mkdir(parents=True, exist_ok=True)
        md_file = auto_dir / (pdf_file.stem + ".md")
        md_file.write_text("# Parsed Title\n\nTable | Data\n---|---\n1 | 2")
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("rak.pdf.subprocess.run", side_effect=fake_run):
        text = extract_pdf_text(pdf_file, provider="mineru")

    assert "Parsed Title" in text
    assert "Table" in text


def test_extract_pdf_text_mineru_fallback_on_failure(tmp_path: Path):
    """When MinerU fails, falls back to PyMuPDF."""
    pdf_file = FIXTURES / "sample.pdf"

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "mineru not found"
        return result

    with patch("rak.pdf.subprocess.run", side_effect=fake_run):
        text = extract_pdf_text(pdf_file, provider="mineru")

    # Should fall back to PyMuPDF and extract text
    assert "test PDF document" in text


def test_extract_pdf_text_mineru_fallback_on_exception(tmp_path: Path):
    """When MinerU subprocess raises, falls back to PyMuPDF."""
    pdf_file = FIXTURES / "sample.pdf"

    with patch("rak.pdf.subprocess.run", side_effect=FileNotFoundError("mineru not found")):
        text = extract_pdf_text(pdf_file, provider="mineru")

    assert "test PDF document" in text


def test_extract_file_text_passes_provider(tmp_path: Path):
    """extract_file_text passes provider to extract_pdf_text."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    with patch("rak.pdf.extract_pdf_text", return_value="mineru text") as mock:
        text = extract_file_text(pdf_file, provider="mineru")

    mock.assert_called_once_with(pdf_file, provider="mineru")
    assert text == "mineru text"


def test_extract_pdf_text_docling_provider(tmp_path: Path):
    """Docling provider calls subprocess and reads markdown output."""
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    def fake_run(cmd, **kwargs):
        # Docling outputs markdown to the --output dir
        output_idx = cmd.index("--output") + 1
        output_dir = Path(cmd[output_idx])
        output_dir.mkdir(parents=True, exist_ok=True)
        md_file = output_dir / (pdf_file.stem + ".md")
        md_file.write_text("# Docling Title\n\n| Col1 | Col2 |\n|---|---|\n| a | b |")
        result = MagicMock()
        result.returncode = 0
        return result

    with patch("rak.pdf.subprocess.run", side_effect=fake_run):
        text = extract_pdf_text(pdf_file, provider="docling")

    assert "Docling Title" in text
    assert "Col1" in text


def test_extract_pdf_text_docling_fallback_on_failure(tmp_path: Path):
    """When Docling fails, falls back to PyMuPDF."""
    pdf_file = FIXTURES / "sample.pdf"

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 1
        result.stderr = "docling error"
        return result

    with patch("rak.pdf.subprocess.run", side_effect=fake_run):
        text = extract_pdf_text(pdf_file, provider="docling")

    assert "test PDF document" in text


def test_extract_pdf_text_docling_fallback_on_exception(tmp_path: Path):
    """When Docling subprocess raises, falls back to PyMuPDF."""
    pdf_file = FIXTURES / "sample.pdf"

    with patch("rak.pdf.subprocess.run", side_effect=FileNotFoundError("docling not found")):
        text = extract_pdf_text(pdf_file, provider="docling")

    assert "test PDF document" in text
