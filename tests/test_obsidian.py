"""Vault export: names Obsidian accepts, links in both directions, no clobbering."""
from __future__ import annotations

import pytest

from sanctions_rag.obsidian import MARKER, assign_names, export, safe_name
from sanctions_rag.store import Entity, Relation, Store


def _store(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.upsert_entities([
        Entity("e1", "Company", "Alpha Bank", ["ru"], ["sanction"], ["us_ofac_sdn"],
               {"alias": ["AO Alpha"], "registrationNumber": ["123"]}, "Alpha Bank"),
        Entity("e2", "Company", "Alpha/Beta: Holding?", ["cy"], [], ["gb_hmt_invbans"], {}, "Alpha Beta"),
        Entity("e3", "Person", "Lonely Person", [], [], [], {}, "Lonely"),
    ])
    s.upsert_relations([Relation("r1", "Ownership", "e2", "e1", None, {"percentage": ["51"]})])
    return s


def test_safe_name_strips_characters_obsidian_rejects():
    assert safe_name('Alpha/Beta: Holding?') == "Alpha Beta Holding"
    assert safe_name("[[x]]#^") == "x"
    assert safe_name("...") == "unnamed"


def test_duplicate_captions_get_id_suffix():
    ents = [Entity("id-aaaaaa", "Person", "John Smith"), Entity("id-bbbbbb", "Person", "JOHN SMITH")]
    names = assign_names(ents)
    assert len(set(names.values())) == 2 and all("(" in n for n in names.values())


def test_export_writes_linked_notes_and_skips_isolated(tmp_path):
    stats = export(_store(tmp_path), tmp_path / "vault")
    assert stats == {"notes": 2, "relations": 1}
    alpha = (tmp_path / "vault/Entities/Company/Alpha Bank.md").read_text()
    owner = (tmp_path / "vault/Entities/Company/Alpha Beta Holding.md").read_text()
    assert "owned by [[Alpha Beta Holding]] (51)" in alpha
    assert "owns [[Alpha Bank]]" in owner
    assert 'aliases: ["AO Alpha"]' in alpha and "topic/sanction" in alpha
    assert not (tmp_path / "vault/Entities/Person").exists()


def test_all_mode_includes_isolated(tmp_path):
    assert export(_store(tmp_path), tmp_path / "v", mode="all")["notes"] == 3


def test_refuses_to_overwrite_a_foreign_folder(tmp_path):
    (tmp_path / "v/Entities").mkdir(parents=True)
    with pytest.raises(SystemExit):
        export(_store(tmp_path), tmp_path / "v")


def test_rerun_replaces_its_own_output(tmp_path):
    s = _store(tmp_path)
    export(s, tmp_path / "v")
    assert (tmp_path / "v" / MARKER).exists()
    assert export(s, tmp_path / "v")["notes"] == 2
