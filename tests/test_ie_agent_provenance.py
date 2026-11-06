from dllmforge.IE_agent_provenance import apply_quote_provenance, locate_quote, normalize_text

ELEMENT_INDEX = [
    {
        "page": 2,
        "type": "text",
        "text": (
            "Two 50 m high dams are located in the upper part of the catchment: "
            "Eupen dam and La Gileppe dam."
        ),
    },
    {
        "page": 4,
        "type": "table",
        "text": (
            "| | Eupen | La Gileppe |\n"
            "| Capacity | 25 × 10^6 m³ | 26.4 × 10^6 m³ |\n"
            "| Dam height | 66 m | 68 m |"
        ),
    },
]


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  Two   50 m\nhigh  ") == "two 50 m high"


def test_locate_quote_prefers_case_study_page_over_table():
    quote = "Two 50 m high dams are located in the upper part of the catchment: Eupen dam and La Gileppe dam."
    hit = locate_quote(quote, ELEMENT_INDEX)
    assert hit is not None
    assert hit["page"] == 2
    assert hit["type"] == "text"


def test_locate_quote_maps_table_row_to_table_element():
    quote = "Dam height | 66 m | 68 m"
    hit = locate_quote(quote, ELEMENT_INDEX)
    assert hit is not None
    assert hit["page"] == 4
    assert hit["type"] == "table"


def test_locate_quote_fuzzy_table_scramble():
    quote = "Dam height 66 m 68 m"
    hit = locate_quote(quote, ELEMENT_INDEX, fuzzy_threshold=0.4)
    assert hit is not None
    assert hit["page"] == 4


def test_apply_quote_provenance_overwrites_wrong_llm_page():
    extracted = {
        "basins": [
            {
                "name": "Eupen dam",
                "source_page": 4,
                "source_quote": (
                    "Two 50 m high dams are located in the upper part of the catchment: "
                    "Eupen dam and La Gileppe dam."
                ),
            }
        ]
    }
    out = apply_quote_provenance(extracted, ELEMENT_INDEX)
    basin = out["basins"][0]
    assert basin["source_page"] == 2
    assert basin["quote_verified"] is True
    assert basin["source_type"] == "text"


def test_apply_quote_provenance_unmatched_quote_clears_page():
    extracted = {"source_page": 4, "source_quote": "this sentence is not in the index"}
    out = apply_quote_provenance(extracted, ELEMENT_INDEX)
    assert out["source_page"] is None
    assert out["quote_verified"] is False


def test_apply_quote_provenance_reads_metadata_element_index():
    extracted = {
        "source_page": 9,
        "source_quote": "Dam height | 66 m | 68 m",
    }
    out = apply_quote_provenance(extracted, metadata={"element_index": ELEMENT_INDEX})
    assert out["source_page"] == 4
    assert out["source_type"] == "table"
