from pathlib import Path

from rak.pdf import chunk_text, extract_pdf_text, extract_file_text, find_attachments

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
