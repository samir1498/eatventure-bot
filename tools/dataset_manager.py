"""
Dataset manager: list-sessions, delete-session, remove-session,
list-orphans, move-orphans, delete-orphans. Uses DATASET_DIR from
src/core/config.py.

  python tools/dataset_manager.py list-sessions
  python tools/dataset_manager.py delete-session --session <id> --dry-run
  python tools/dataset_manager.py move-orphans --yes
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.config import DATASET_DIR, LOG_LEVEL  # noqa: E402
from core import log  # noqa: E402
from dataset.layout import shard_roots  # noqa: E402

log.setup(LOG_LEVEL)
logger = log.get("tools.dataset_manager")

SAMPLES = lambda jsonl: Path(jsonl)


def load_samples(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignorando linha inválida %d em %s", lineno, path)


def list_sessions(jsonl_path: Path):
    import collections

    c = collections.Counter()
    for _, sample in load_samples(jsonl_path):
        c[sample.get("session")] += 1

    for session, count in c.most_common():
        print(f"{count}\t{session}")


def find_orphans(jsonl_path: Path, images_root: Path):
    referenced = set()
    for _, s in load_samples(jsonl_path):
        im = s.get("image")
        if im:
            referenced.add(str(Path(im).as_posix()))

    orphans = []
    if not images_root.exists():
        logger.warning("Images root does not exist: %s", images_root)
        return orphans

    for f in images_root.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(images_root.parent).as_posix())
            if rel not in referenced:
                orphans.append(rel)
    return orphans


def prune_missing_images(jsonl_path: Path, images_root: Path):
    lines_to_drop = set()
    dataset_root = images_root.parent.resolve()

    for lineno, sample in load_samples(jsonl_path):
        image_rel = sample.get("image")
        if not image_rel:
            lines_to_drop.add(lineno)
            continue

        image_path = (dataset_root / Path(image_rel)).resolve()
        if dataset_root not in image_path.parents or not image_path.is_file():
            lines_to_drop.add(lineno)

    if not lines_to_drop:
        return 0

    backup = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    if backup.exists():
        backup.unlink()
    jsonl_path.rename(backup)
    try:
        with backup.open("r", encoding="utf-8") as source, jsonl_path.open("w", encoding="utf-8") as target:
            for lineno, line in enumerate(source, start=1):
                if lineno not in lines_to_drop:
                    target.write(line)
    except Exception:
        logger.exception("Erro reescrevendo índice; restaurando backup")
        if jsonl_path.exists():
            jsonl_path.unlink()
        backup.rename(jsonl_path)
        raise

    print(f"Índice limpo: {len(lines_to_drop)} registro(s) removido(s) de {jsonl_path}")
    print(f"Backup do índice em: {backup}")
    return len(lines_to_drop)


def delete_session_images(jsonl_path: Path, session_id: str, yes=False, prune_index=False, dry_run=False):
    to_delete = []
    lines_to_drop = set()

    for lineno, sample in load_samples(jsonl_path):
        if sample.get("session") == session_id:
            image_rel = sample.get("image")
            if not image_rel:
                logger.warning("Amostra %s sem campo 'image' (linha %d), pulando", sample.get("id"), lineno)
                lines_to_drop.add(lineno)
                continue
            image_path = (DATASET_DIR / Path(image_rel)).resolve()
            try:
                if DATASET_DIR.resolve() in image_path.parents or image_path == DATASET_DIR.resolve():
                    to_delete.append((lineno, image_path))
                    lines_to_drop.add(lineno)
                else:
                    logger.warning("Caminho %s fora de %s — não será apagado", image_path, DATASET_DIR)
                    lines_to_drop.add(lineno)
            except Exception:
                logger.exception("Erro verificando caminho %s", image_path)

    if not to_delete:
        print(f"Nenhuma amostra encontrada para session '{session_id}' no índice {jsonl_path}.")
        return 0

    print(f"Acharam {len(to_delete)} imagem(ns) para session '{session_id}':")
    for ln, p in to_delete:
        print(f"  [line {ln}] {p}")

    if dry_run:
        print("Dry-run: nada será apagado.")
        return 0

    if not yes:
        resp = input("Confirma apagar estes arquivos? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Abortado pelo usuário.")
            return 0

    deleted = 0
    missing = 0
    for ln, p in to_delete:
        try:
            if p.exists():
                p.unlink()
                deleted += 1
                logger.info("Apagado %s", p)
            else:
                missing += 1
                logger.warning("Arquivo ausente: %s (linha %d)", p, ln)
        except Exception:
            logger.exception("Falha apagando %s", p)

    print(f"Deletados: {deleted}; faltantes: {missing}.")

    if prune_index:
        backup = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
        jsonl_path.rename(backup)
        try:
            with backup.open("r", encoding="utf-8") as rfh, jsonl_path.open("w", encoding="utf-8") as wfh:
                for lineno, line in enumerate(rfh, start=1):
                    if lineno in lines_to_drop:
                        continue
                    wfh.write(line)
        except Exception:
            logger.exception("Erro reescrevendo índice; restaurando backup")
            if jsonl_path.exists():
                jsonl_path.unlink()
            backup.rename(jsonl_path)
            print("Falha reescrevendo índice; backup restaurado.")
            return 1
        print(f"Índice atualizado (linhas removidas): {jsonl_path} — backup em {backup}")

    return 0


def remove_session(jsonl_path: Path, session_id: str, yes=False, dry_run=False, backup_images=False):
    """Removes a session's images and index lines; always backs up the index before rewriting."""

    to_remove = []
    lines_to_drop = set()

    for lineno, sample in load_samples(jsonl_path):
        if sample.get("session") == session_id:
            image_rel = sample.get("image")
            if not image_rel:
                logger.warning("Amostra %s sem campo 'image' (linha %d), pulando", sample.get("id"), lineno)
                lines_to_drop.add(lineno)
                continue
            image_path = (DATASET_DIR / Path(image_rel)).resolve()
            try:
                if DATASET_DIR.resolve() in image_path.parents or image_path == DATASET_DIR.resolve():
                    to_remove.append((lineno, image_rel, image_path))
                    lines_to_drop.add(lineno)
                else:
                    logger.warning("Caminho %s fora de %s — não será apagado", image_path, DATASET_DIR)
                    lines_to_drop.add(lineno)
            except Exception:
                logger.exception("Erro verificando caminho %s", image_path)

    if not to_remove:
        print(f"Nenhuma amostra encontrada para session '{session_id}' no índice {jsonl_path}.")
        return 0

    print(f"Acharam {len(to_remove)} imagem(ns) para session '{session_id}':")
    for ln, rel, p in to_remove:
        print(f"  [line {ln}] {p}")

    if dry_run:
        print("Dry-run: nada será apagado nem reescrito.")
        return 0

    if not yes:
        resp = input("Confirma remover estas imagens e as linhas do índice? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Abortado pelo usuário.")
            return 0

    backup_json = jsonl_path.with_suffix(jsonl_path.suffix + ".bak")
    jsonl_path.rename(backup_json)

    moved = 0
    deleted = 0
    missing = 0
    errors = 0

    if backup_images:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_backup = DATASET_DIR / f"backup_session_{session_id}_{ts}"
        img_backup.mkdir(parents=True, exist_ok=True)

    for ln, rel, p in to_remove:
        try:
            if p.exists():
                if backup_images:
                    dest = img_backup / Path(rel)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(dest))
                    moved += 1
                else:
                    p.unlink()
                    deleted += 1
            else:
                missing += 1
        except Exception:
            logger.exception("Erro ao manipular %s", p)
            errors += 1

    try:
        with backup_json.open("r", encoding="utf-8") as rfh, jsonl_path.open("w", encoding="utf-8") as wfh:
            for lineno, line in enumerate(rfh, start=1):
                if lineno in lines_to_drop:
                    continue
                wfh.write(line)
    except Exception:
        logger.exception("Erro reescrevendo índice. Backup preservado em %s", backup_json)
        print("Falha reescrevendo índice; verifique backup e estado manualmente.")
        return 1

    print(f"Operação concluída: moved={moved} deleted={deleted} missing={missing} errors={errors}")
    if backup_images:
        print(f"Imagens movidas para: {img_backup}")
    print(f"Backup do índice em: {backup_json}")
    return 0


def move_orphans(images_root: Path, orphan_list, yes=False):
    if not orphan_list:
        print("Nenhuma imagem órfã encontrada.")
        return 0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = images_root.parent / f"backup_orphans_{ts}"
    backup.mkdir(parents=True, exist_ok=True)

    print(f"Mover {len(orphan_list)} arquivos para {backup}")
    if not yes:
        resp = input("Confirma? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Abortado pelo usuário.")
            return 0

    moved = 0
    missing = 0
    for rel in orphan_list:
        src = (images_root.parent / Path(rel)).resolve()
        dest = backup / Path(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dest))
            moved += 1
        else:
            logger.warning("Arquivo ausente: %s", src)
            missing += 1

    # write moved list
    moved_list = backup / "moved_list.txt"
    with moved_list.open("w", encoding="utf-8") as fh:
        for rel in orphan_list:
            fh.write(f"{rel}\n")

    print(f"Resultado: moved={moved} missing={missing} backup_dir={backup}")
    return 0


def delete_orphans(jsonl_path: Path, images_root: Path, orphan_list, backup_before=False, yes=False):
    if not orphan_list:
        print("Nenhuma imagem órfã encontrada.")
        prune_missing_images(jsonl_path, images_root)
        return 0

    if backup_before:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = images_root.parent / f"backup_orphans_{ts}"
        backup.mkdir(parents=True, exist_ok=True)
        for rel in orphan_list:
            src = (images_root.parent / Path(rel)).resolve()
            dest = backup / Path(rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.move(str(src), str(dest))
        print(f"Arquivos movidos para backup em {backup}")
        return 0

    print(f"Apagar {len(orphan_list)} arquivos órfãos (irreversível)")
    if not yes:
        resp = input("Confirma apagar estes arquivos? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Abortado pelo usuário.")
            return 0

    deleted = 0
    missing = 0
    for rel in orphan_list:
        src = (images_root.parent / Path(rel)).resolve()
        if src.exists():
            src.unlink()
            deleted += 1
        else:
            missing += 1
    print(f"Deletados: {deleted}; faltantes: {missing}.")
    prune_missing_images(jsonl_path, images_root)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dataset manager: delete by session, manage orphans, etc.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list-sessions", help="List session ids and counts")

    ds = sub.add_parser("delete-session", help="Delete images for a given session id")
    ds.add_argument("--session", required=True)
    ds.add_argument("--jsonl", help="samples.jsonl path (default: DATASET_DIR/samples.jsonl)")
    ds.add_argument("--yes", action="store_true")
    ds.add_argument("--prune-index", action="store_true")
    ds.add_argument("--dry-run", action="store_true")

    rs = sub.add_parser("remove-session", help="Remove images and index entries for a session (delete images and prune json)")
    rs.add_argument("--session", required=True)
    rs.add_argument("--jsonl", help="samples.jsonl path (default: DATASET_DIR/samples.jsonl)")
    rs.add_argument("--yes", action="store_true")
    rs.add_argument("--dry-run", action="store_true")
    rs.add_argument("--backup-images", action="store_true", help="Move images to backup instead of deleting them")

    lo = sub.add_parser("list-orphans", help="List images that are in dataset/images but not referenced in samples.jsonl")
    lo.add_argument("--jsonl", help="samples.jsonl path")

    mo = sub.add_parser("move-orphans", help="Move orphan images to backup folder")
    mo.add_argument("--jsonl", help="samples.jsonl path")
    mo.add_argument("--yes", action="store_true")

    do = sub.add_parser("delete-orphans", help="Delete orphan images (or backup them first)")
    do.add_argument("--jsonl", help="samples.jsonl path")
    do.add_argument("--backup", action="store_true", help="Move to backup instead of deleting")
    do.add_argument("--yes", action="store_true")

    args = parser.parse_args(argv)

    jsonl_path = Path(args.jsonl) if getattr(args, "jsonl", None) else (DATASET_DIR / "samples.jsonl")

    if args.cmd == "list-sessions":
        if not jsonl_path.exists():
            print(f"Índice não encontrado: {jsonl_path}")
            return 1
        list_sessions(jsonl_path)
        return 0

    if args.cmd == "delete-session":
        if not jsonl_path.exists():
            print(f"Índice não encontrado: {jsonl_path}")
            return 1
        return delete_session_images(jsonl_path, args.session, yes=args.yes, prune_index=args.prune_index, dry_run=args.dry_run)

    if args.cmd == "remove-session":
        if not jsonl_path.exists():
            print(f"Índice não encontrado: {jsonl_path}")
            return 1
        return remove_session(jsonl_path, args.session, yes=args.yes, dry_run=args.dry_run, backup_images=getattr(args, "backup_images", False))

    if args.cmd in ("list-orphans", "move-orphans", "delete-orphans"):
        if getattr(args, "jsonl", None):
            dataset_locations = [(jsonl_path, jsonl_path.parent / "images")]
        else:
            dataset_locations = [
                (shard / "samples.jsonl", shard / "images")
                for shard in shard_roots(DATASET_DIR)
            ]

        if not dataset_locations:
            print(f"Nenhum índice encontrado em {DATASET_DIR / 'data'}")
            return 1

        if args.cmd == "list-orphans":
            total = 0
            for current_jsonl, images_root in dataset_locations:
                orphan_list = find_orphans(current_jsonl, images_root)
                total += len(orphan_list)
                print(f"{len(orphan_list)} orphan images not referenced in {current_jsonl}")
                for orphan in orphan_list:
                    print(orphan)
            print(f"Total: {total} orphan images")
            return 0

        result = 0
        for current_jsonl, images_root in dataset_locations:
            orphan_list = find_orphans(current_jsonl, images_root)
            if args.cmd == "move-orphans":
                result |= move_orphans(images_root, orphan_list, yes=args.yes)
            else:
                result |= delete_orphans(
                    current_jsonl,
                    images_root,
                    orphan_list,
                    backup_before=args.backup,
                    yes=args.yes,
                )
        return result

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
