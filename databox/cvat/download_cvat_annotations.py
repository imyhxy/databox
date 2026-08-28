"""Download the latest CVAT annotations and make them usable locally."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

DEFAULT_FORMAT = "CVAT for images 1.1"
HASH_LENGTH = 6
VOLATILE_METADATA_TIME_TAGS = frozenset({"created", "updated", "dumped"})


def load_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}

    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value

    return values


def get_default_config(dotenv_path: Path = Path(".env")) -> dict[str, str | None]:
    dotenv_values = load_dotenv(dotenv_path)
    config = {}
    for key in ("CVAT_HOST", "CVAT_USERNAME", "CVAT_PASSWORD", "CVAT_ACCESS_TOKEN"):
        config[key] = os.environ.get(key, dotenv_values.get(key))
    return config


@dataclass(frozen=True)
class DownloadTarget:
    kind: str
    identifier: int

    @property
    def stem(self) -> str:
        return f"{self.kind}-{self.identifier}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = get_default_config()
    parser = argparse.ArgumentParser(
        description=(
            "Download CVAT annotations, extract annotations.xml, and update "
            "the local symlink."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--project-id", type=int, help="CVAT project ID")
    target_group.add_argument("--task-id", type=int, help="CVAT task ID")
    target_group.add_argument("--job-id", type=int, help="CVAT job ID")

    parser.add_argument(
        "--host",
        default=default_config["CVAT_HOST"],
        help=(
            "CVAT host, for example http://localhost:8080. Defaults to "
            "CVAT_HOST or .env."
        ),
    )
    parser.add_argument(
        "--username",
        default=default_config["CVAT_USERNAME"],
        help="CVAT username. Defaults to CVAT_USERNAME or .env.",
    )
    parser.add_argument(
        "--password",
        default=default_config["CVAT_PASSWORD"],
        help="CVAT password. Defaults to CVAT_PASSWORD or .env.",
    )
    parser.add_argument(
        "--access-token",
        default=default_config["CVAT_ACCESS_TOKEN"],
        help="CVAT access token. Defaults to CVAT_ACCESS_TOKEN or .env.",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        help=f"CVAT export format. Defaults to {DEFAULT_FORMAT!r}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory that receives timestamped annotations and annotations.xml.",
    )
    parser.add_argument(
        "--strip-prefix",
        default="",
        help="Prefix to remove from each CVAT image path before writing the local XML.",
    )
    parser.add_argument(
        "--add-prefix",
        default="",
        help="Prefix to add to each CVAT image path after stripping.",
    )
    parser.add_argument(
        "--include-images",
        action="store_true",
        help=(
            "Include images in the CVAT export archive. By default only "
            "annotations are downloaded."
        ),
    )
    parser.add_argument(
        "--link-images-from",
        type=Path,
        help="Create output_dir/images as a relative symlink to this image directory.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help=(
            "Override export timestamp, mainly for tests. Defaults to current UTC time."
        ),
    )
    return parser.parse_args(argv)


def target_from_args(args: argparse.Namespace) -> DownloadTarget:
    for kind in ("project", "task", "job"):
        identifier = getattr(args, f"{kind}_id")
        if identifier is not None:
            return DownloadTarget(kind=kind, identifier=identifier)
    raise ValueError("one target ID is required")


def download_export(
    *,
    host: str,
    username: str | None,
    password: str | None,
    access_token: str | None,
    target: DownloadTarget,
    format_name: str,
    output_zip: Path,
    include_images: bool,
) -> None:
    try:
        from cvat_sdk import make_client
    except ImportError as exc:
        raise RuntimeError(
            "cvat-sdk is required. Run this script in the conda 51 environment."
        ) from exc

    if not host:
        raise ValueError("--host or CVAT_HOST is required")
    credentials = None
    if access_token is None:
        if not username or not password:
            raise ValueError(
                "provide --access-token/CVAT_ACCESS_TOKEN or both "
                "--username/CVAT_USERNAME and --password/CVAT_PASSWORD"
            )
        credentials = (username, password)

    with make_client(
        host=host,
        credentials=credentials,
        access_token=access_token,
    ) as client:
        if target.kind == "project":
            resource = client.projects.retrieve(target.identifier)
        elif target.kind == "task":
            resource = client.tasks.retrieve(target.identifier)
        elif target.kind == "job":
            resource = client.jobs.retrieve(target.identifier)
        else:
            raise ValueError(f"unsupported target kind: {target.kind}")

        resource.export_dataset(
            format_name,
            output_zip,
            include_images=include_images,
        )


def extract_annotations_xml(archive_path: Path, extract_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        xml_members = [
            member
            for member in archive.namelist()
            if not member.endswith("/")
            and PurePosixPath(member).suffix.lower() == ".xml"
        ]
        if not xml_members:
            raise ValueError(f"no XML label file found in {archive_path}")

        preferred = [
            member
            for member in xml_members
            if PurePosixPath(member).name == "annotations.xml"
        ]
        if len(preferred) == 1:
            member = preferred[0]
        elif len(xml_members) == 1:
            member = xml_members[0]
        else:
            choices = ", ".join(xml_members)
            raise ValueError(f"multiple XML files found; cannot choose one: {choices}")

        archive.extract(member, extract_dir)
        return extract_dir / member


def normalize_cvat_image_paths(
    xml_path: Path,
    output_path: Path,
    *,
    strip_prefix: str = "",
    add_prefix: str = "",
) -> int:
    tree = ET.parse(xml_path)
    changed = 0
    for image in tree.getroot().iterfind("./image"):
        original = image.attrib.get("name")
        if original is None:
            continue
        updated = rewrite_image_path(
            original,
            strip_prefix=strip_prefix,
            add_prefix=add_prefix,
        )
        if updated != original:
            image.set("name", updated)
            changed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xml_text = ET.tostring(
        tree.getroot(),
        encoding="unicode",
        short_empty_elements=False,
        xml_declaration=True,
    )
    output_path.write_text(xml_text.replace("'", '"'))
    return changed


def rewrite_image_path(
    path: str, *, strip_prefix: str = "", add_prefix: str = ""
) -> str:
    normalized = path.replace("\\", "/")
    if strip_prefix:
        normalized_strip_prefix = strip_prefix.replace("\\", "/").rstrip("/")
        if normalized == normalized_strip_prefix:
            normalized = ""
        elif normalized.startswith(f"{normalized_strip_prefix}/"):
            normalized = normalized[len(normalized_strip_prefix) + 1 :]

    if add_prefix:
        normalized_add_prefix = add_prefix.replace("\\", "/").strip("/")
        normalized = f"{normalized_add_prefix}/{normalized.lstrip('/')}"

    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_cvat_annotation_content(path: Path) -> str:
    """Hash annotation content while ignoring CVAT export-time metadata.

    CVAT changes these timestamps on every export even when labels are
    unchanged.  They are only ignored for deduplication; the downloaded XML
    is still stored with its original metadata intact.
    """
    root = ET.parse(path).getroot()
    metadata = root.find("./meta")
    if metadata is not None:
        for parent in metadata.iter():
            for child in list(parent):
                if child.tag in VOLATILE_METADATA_TIME_TAGS:
                    parent.remove(child)

    canonical_xml = ET.tostring(
        root,
        encoding="utf-8",
        short_empty_elements=False,
        xml_declaration=True,
    )
    return hashlib.sha256(canonical_xml).hexdigest()


def find_existing_annotation(output_dir: Path, digest_prefix: str) -> Path | None:
    matches = sorted(output_dir.glob(f"annotations-*-{digest_prefix}.xml"))
    if matches:
        return matches[0]
    return None


def replace_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_path.name)


def replace_relative_directory_symlink(link_path: Path, target_path: Path) -> None:
    """Replace a directory link with a relative link to an existing directory."""
    # Keep the user-provided path lexical (rather than resolving through a DVC
    # cache symlink) while still making it absolute for a stable relpath.
    target_path = Path(os.path.abspath(os.fspath(Path(target_path).expanduser())))
    if not target_path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {target_path}")

    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.exists():
        raise IsADirectoryError(
            f"Cannot replace non-symlink image directory: {link_path}"
        )

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(os.path.relpath(target_path, start=link_path.parent))


def timestamp_now() -> str:
    return datetime.now(UTC).strftime("%y%m%dT%H%M%SZ")


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    link_images_from = getattr(args, "link_images_from", None)
    target = target_from_args(args)
    timestamp = args.timestamp or timestamp_now()

    with tempfile.TemporaryDirectory(prefix="cvat-export-") as tmp:
        tmp_dir = Path(tmp)
        temp_zip = tmp_dir / "export.zip"
        download_export(
            host=args.host,
            username=args.username,
            password=args.password,
            access_token=args.access_token,
            target=target,
            format_name=args.format,
            output_zip=temp_zip,
            include_images=args.include_images,
        )

        extract_dir = tmp_dir / "extract"
        extracted_xml = extract_annotations_xml(temp_zip, extract_dir)
        temp_xml = tmp_dir / "annotations.xml"
        normalize_cvat_image_paths(
            extracted_xml,
            temp_xml,
            strip_prefix=args.strip_prefix,
            add_prefix=args.add_prefix,
        )
        xml_digest = sha256_cvat_annotation_content(temp_xml)[:HASH_LENGTH]

        existing_xml = find_existing_annotation(output_dir, xml_digest)
        if existing_xml is not None:
            final_xml = existing_xml
            final_xml.touch()
        else:
            final_xml = (
                output_dir / f"annotations-{target.stem}-{timestamp}-{xml_digest}.xml"
            )
            shutil.move(temp_xml, final_xml)
        replace_symlink(output_dir / "annotations.xml", final_xml)

    if link_images_from is not None:
        replace_relative_directory_symlink(
            output_dir / "images", Path(link_images_from)
        )

    return final_xml


def main() -> None:
    final_xml = run(parse_args())
    print(f"Extracted annotations: {final_xml}")
    print(
        f"Updated symlink: {final_xml.parent / 'annotations.xml'} -> {final_xml.name}"
    )


if __name__ == "__main__":
    main()
