from pathlib import Path

from rak.pdf import extract_pdf_text, find_pdf

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_pdf_text():
    text = extract_pdf_text(FIXTURES / "sample.pdf")
    assert "test PDF document" in text
    assert len(text) > 10


def test_extract_pdf_text_missing_file():
    text = extract_pdf_text(Path("/nonexistent/file.pdf"))
    assert text == ""


def test_find_pdf_found(tmp_path: Path):
    key_dir = tmp_path / "ABC12345"
    key_dir.mkdir()
    pdf_file = key_dir / "Smith 2024 - Paper.pdf"
    pdf_file.touch()
    result = find_pdf(tmp_path, "ABC12345")
    assert result == pdf_file


def test_find_pdf_not_found(tmp_path: Path):
    result = find_pdf(tmp_path, "NONEXIST")
    assert result is None


def test_find_pdf_no_pdf_in_dir(tmp_path: Path):
    key_dir = tmp_path / "ABC12345"
    key_dir.mkdir()
    (key_dir / "snapshot.html").touch()
    result = find_pdf(tmp_path, "ABC12345")
    assert result is None
