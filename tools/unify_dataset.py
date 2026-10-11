#!/usr/bin/env python3
"""
Merges dataset_to_unify/ into dataset/, renaming conflicting images
and adjusting each sample's `image` field. Cleans dataset_to_unify
at the end.

    python tools/unify_dataset.py
"""

import argparse
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
TARGET_DATASET = ROOT / "dataset"
SOURCE_DATASET = ROOT / "dataset_to_unify"


def parse_json_objects(raw_line: str):
    line = raw_line.strip()
    if not line:
        return []

    tried = [line]
    if line.count("{") > 1:
        import re
        pieces = re.split(r'(?<=\})\s*(?=\{)|(?<=\])\s*(?=\{)', line)
        tried = [piece.strip() for piece in pieces if piece and piece.strip()]

    parsed = []
    for candidate in tried:
        try:
            item = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            parsed.append(item)
    return parsed


def load_jsonl(path: Path):
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            items = parse_json_objects(text)
            if not items:
                raise ValueError(f"Linha inválida em {path}: {text[:200]}")
            rows.extend(items)
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def normalize_rel_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("/")


def resolve_source_image(source_dir: Path, image_ref: str) -> Path | None:
    if not image_ref:
        return None

    candidate = Path(image_ref)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    normalized = normalize_rel_path(image_ref)
    candidates = [
        source_dir / normalized,
        source_dir / "images" / normalized,
        source_dir / normalized.lstrip("images/"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def sanitize_relative_image_path(value: str) -> str:
    normalized = normalize_rel_path(value)
    if not normalized:
        return ""

    parts = []
    for part in PurePosixPath(normalized).parts:
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)

    return "/".join(parts)


def first_available_target(target_root: Path, rel_image: str, reserved: set[str]) -> Path:
    target_root = target_root.resolve()
    safe_rel = sanitize_relative_image_path(rel_image)
    if not safe_rel:
        safe_rel = "images/unnamed.jpg"

    target_path = (target_root / safe_rel).resolve()
    try:
        target_path.relative_to(target_root)
    except ValueError:
        target_path = (target_root / "images" / Path(safe_rel).name).resolve()

    target_parent = target_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    rel_target = normalize_rel_path(target_path.relative_to(target_root).as_posix())
    if not target_path.exists() and rel_target not in reserved:
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    idx = 1
    while True:
        candidate = target_parent / f"{stem}_{idx}{suffix}"
        rel = normalize_rel_path(candidate.relative_to(target_root).as_posix())
        if not candidate.exists() and rel not in reserved:
            return candidate
        idx += 1


def clear_directory(path: Path):
    if not path.exists():
        return

    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def merge_datasets(source_dir: Path, target_dir: Path):
    source_dir = source_dir.resolve()
    target_dir = target_dir.resolve()

    target_samples_path = target_dir / "samples.jsonl"
    target_images_dir = target_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    existing_rows = load_jsonl(target_samples_path)
    existing_ids = {str(row.get("id")) for row in existing_rows if row.get("id") is not None}
    existing_paths = {
        normalize_rel_path(str(row.get("image", "")))
        for row in existing_rows
        if row.get("image")
    }

    source_jsonl = source_dir / "samples.jsonl"
    if not source_jsonl.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {source_jsonl}")

    merged_rows = list(existing_rows)
    added = 0
    renamed = 0

    for row in load_jsonl(source_jsonl):
        sample_id = row.get("id")
        if sample_id is not None and str(sample_id) in existing_ids:
            continue

        image_ref = row.get("image")
        if not image_ref:
            merged_rows.append(row)
            continue

        source_image = resolve_source_image(source_dir, image_ref)
        if source_image is None:
            print(f"[warn] imagem não encontrada para amostra {sample_id}: {image_ref}")
            merged_rows.append(row)
            continue

        rel_image = normalize_rel_path(image_ref)
        target_path = (target_dir / rel_image).resolve()
        final_path = first_available_target(target_dir, rel_image, existing_paths)

        if target_path != final_path:
            renamed += 1
            row["image"] = normalize_rel_path(final_path.relative_to(target_dir).as_posix())
            rel_image = row["image"]

        if not final_path.exists():
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_image), str(final_path))

        existing_paths.add(normalize_rel_path(rel_image))
        existing_ids.add(str(sample_id)) if sample_id is not None else None
        merged_rows.append(row)
        added += 1

    write_jsonl(target_samples_path, merged_rows)
    return added, renamed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Mescla dataset_to_unify em dataset/ e limpa a pasta de origem.",
    )
    parser.add_argument(
        "--source",
        default=str(SOURCE_DATASET),
        help="Diretório de origem com samples.jsonl e imagens (padrão: dataset_to_unify).",
    )
    parser.add_argument(
        "--target",
        default=str(TARGET_DATASET),
        help="Diretório destino (padrão: dataset).",
    )
    args = parser.parse_args(argv)

    source_dir = Path(args.source).resolve()
    target_dir = Path(args.target).resolve()

    source_dir.mkdir(parents=True, exist_ok=True)

    if not (source_dir / "samples.jsonl").exists():
        clear_directory(source_dir)
        print(f"[info] Nenhum dataset em {source_dir}. Pasta pronta para receber arquivos.")
        return 0

    try:
        added, renamed = merge_datasets(source_dir, target_dir)
    except Exception as exc:
        print(f"[error] Falhou ao mesclar: {exc}")
        return 1

    clear_directory(source_dir)
    print(f"[ok] {added} amostra(s) importada(s) no dataset principal.")
    print(f"[ok] {renamed} conflito(s) de imagem resolvido(s) com renomeação.")
    print(f"[ok] Pasta limpa: {source_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
