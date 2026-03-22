from rak.export import to_csv, to_bibtex


def test_to_csv():
    results = [
        {"key": "A1", "title": "Paper One", "score": 0.85, "source": "vector", "date": "2024", "authors": "Doe, J."},
        {"key": "A2", "title": "Paper Two", "score": 0.72, "source": "fused", "date": "2023", "authors": "Smith, A."},
    ]
    csv_text = to_csv(results)
    lines = [l.strip() for l in csv_text.strip().splitlines()]
    assert lines[0] == "key,title,score,source"
    assert "A1" in lines[1]
    assert "Paper One" in lines[1]
    assert "0.85" in lines[1]


def test_to_csv_empty():
    csv_text = to_csv([])
    lines = [l.strip() for l in csv_text.strip().splitlines()]
    assert lines[0] == "key,title,score,source"
    assert len(lines) == 1


def test_to_bibtex():
    results = [
        {"key": "A1", "title": "Paper One", "score": 0.85, "source": "vector", "date": "2024", "authors": "Doe, J."},
    ]
    bib = to_bibtex(results)
    assert "@article{A1," in bib
    assert "title = {Paper One}" in bib
    assert "year = {2024}" in bib
    assert "author = {Doe, J.}" in bib


def test_to_bibtex_empty():
    bib = to_bibtex([])
    assert bib == ""


def test_to_bibtex_conference_paper():
    results = [
        {"key": "C1", "title": "Conf Paper", "score": 0.9, "source": "vector",
         "date": "2024-06-15", "authors": "Lee, K.", "item_type": "conferencePaper"},
    ]
    bib = to_bibtex(results)
    assert "@inproceedings{C1," in bib
    assert "year = {2024}" in bib


def test_to_bibtex_book():
    results = [
        {"key": "B1", "title": "A Book", "score": 0.7, "source": "vector",
         "date": "2020", "authors": "Author", "item_type": "book"},
    ]
    bib = to_bibtex(results)
    assert "@book{B1," in bib


def test_to_bibtex_unknown_type_defaults_to_article():
    results = [
        {"key": "U1", "title": "Unknown", "score": 0.5, "source": "vector",
         "date": "2023", "authors": "X", "item_type": "weirdType"},
    ]
    bib = to_bibtex(results)
    assert "@article{U1," in bib
