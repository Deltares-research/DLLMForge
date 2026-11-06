from dllmforge.IE_agent_provenance import (
    apply_quote_provenance,
    locate_quote,
    normalize_text,
    public_metadata,
)

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


CONCATENATED_QUOTE = (
    "Two 50m high dams are located in the upper part of the catchment: "
    "Eupen dam and La Gileppe dam. The former is situated on the main course "
    "of the River Vesdre, 3km upstream of the town Eupen, while the latter "
    "is on the left-bank tributary La Gileppe.\n\n"
    "|                                             | Eupen               | La Gileppe       |\n"
    "|---------------------------------------------|---------------------|------------------|\n"
    "| Capacity                                    | 25 � 10 6 hm 3      | 26:4 � 10 6 hm 3 |\n"
    "| Dam height                                  | 66 m                | 68 m             |\n"
    "| Natural river                               | Vesdre and Getzbach | La Gileppe       |\n"
    "| Diverted river                              | Helle               | Soor             |\n\n"
    "First, a prescribed \"maximum water level\" may not be exceeded in each "
    "reservoir (Table 1), so as to keep a free storage of about 3�10 6 m 3 "
    "in each reservoir for floods."
)


def test_locate_quote_concatenated_blocks_prefers_table():
    hit = locate_quote(CONCATENATED_QUOTE, ELEMENT_INDEX)
    assert hit is not None
    assert hit["page"] == 4
    assert hit["type"] == "table"


def test_normalize_text_fixes_ocr_units_and_scientific_notation():
    assert "50 m" in normalize_text("Two 50m high dams")
    assert "25 x 10 6" in normalize_text("25 × 10^6")
    assert "26.4" in normalize_text("26:4 × 10 6")


def test_public_metadata_strips_locator_index():
    meta = public_metadata({
        "source_file": "nhess.pdf",
        "element_index": ELEMENT_INDEX,
        "page_index": [{"page": 1, "text": "x"}],
        "processing_method": "docling",
    })
    assert meta["source_file"] == "nhess.pdf"
    assert meta["processing_method"] == "docling"
    assert "element_index" not in meta
    assert "page_index" not in meta
