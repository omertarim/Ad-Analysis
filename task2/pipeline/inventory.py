"""Scan embedded playable images and join them to the MAP role catalog."""

from __future__ import annotations

import json
import re
import struct
from html import unescape
from pathlib import Path

TASK2_DIR = Path(__file__).resolve().parents[1]
SOURCE_HTML = TASK2_DIR / "source" / "RK_SaveTheKing_Playable.html"
ROLES_PATH = TASK2_DIR / "inventory" / "roles.json"
ARTIFACT_PATH = TASK2_DIR / "artifacts" / "inventory_scan.json"

IMG_RE = re.compile(
    r'<img id="([^"]+)" data-src122="(.*?)" data-mime="([^"]+)"',
    re.DOTALL,
)
_BASE122_SPECIALS = (0, 10, 13, 34, 38, 92)


def decode_base122(payload: str) -> bytes:
    value = 0
    bits = 0
    written = 0
    out = bytearray(int(1.75 * len(payload)) + 8)

    def push(seven_bits: int) -> None:
        nonlocal value, bits, written
        seven_bits <<= 1
        value |= seven_bits >> bits
        bits += 7
        if bits >= 8:
            out[written] = value & 255
            written += 1
            bits -= 8
            value = (seven_bits << (7 - bits)) & 255

    for char in payload:
        code = ord(char)
        if code > 127:
            hi = (code >> 8) & 7
            if hi != 7:
                push(_BASE122_SPECIALS[hi])
            push(code & 127)
        else:
            push(code)
    return bytes(out[:written])


def sniff_kind(blob: bytes) -> str:
    if blob.startswith(b"\x89PNG"):
        return "png"
    if blob[:2] == b"\xff\xd8":
        return "jpeg"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    return "bin"


def png_size(blob: bytes) -> tuple[int, int] | tuple[None, None]:
    if len(blob) < 24:
        return None, None
    width, height = struct.unpack(">II", blob[16:24])
    return width, height


def jpeg_size(blob: bytes) -> tuple[int, int] | tuple[None, None]:
    index = 2
    while index < len(blob) - 9:
        if blob[index] != 0xFF:
            index += 1
            continue
        marker = blob[index + 1]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", blob[index + 5 : index + 9])
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            index += 2
            continue
        if index + 4 > len(blob):
            break
        length = struct.unpack(">H", blob[index + 2 : index + 4])[0]
        index += 2 + length
    return None, None


def webp_size(blob: bytes) -> tuple[int, int] | tuple[None, None]:
    if len(blob) < 30:
        return None, None
    tag = blob[12:16]
    if tag == b"VP8X":
        width = 1 + int.from_bytes(blob[24:27], "little")
        height = 1 + int.from_bytes(blob[27:30], "little")
        return width, height
    if tag == b"VP8 ":
        width = struct.unpack_from("<H", blob, 26)[0] & 0x3FFF
        height = struct.unpack_from("<H", blob, 28)[0] & 0x3FFF
        return width, height
    if tag == b"VP8L":
        bits = struct.unpack_from("<I", blob, 21)[0]
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None, None


def image_size(kind: str, blob: bytes) -> tuple[int | None, int | None]:
    if kind == "png":
        return png_size(blob)
    if kind == "jpeg":
        return jpeg_size(blob)
    if kind == "webp":
        return webp_size(blob)
    return None, None


def load_roles() -> dict:
    return json.loads(ROLES_PATH.read_text(encoding="utf-8"))


def build_role_index(roles: dict) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for role in roles["asset_roles"]:
        for asset in role["assets"]:
            index[str(asset["id"])] = {
                "role_id": role["id"],
                "role_label": role["label"],
                "replaceable": bool(role["replaceable"]),
                "risk": role["risk"],
                "preserve_aspect": role["preserve_aspect"],
                "preserve_silhouette": role["preserve_silhouette"],
                "asset_label": asset.get("label"),
            }
    blocked: set[str] = set()
    for item in roles.get("do_not_touch", []):
        blocked.update(str(asset_id) for asset_id in item.get("asset_ids", []))
    for asset_id in blocked:
        if asset_id in index:
            index[asset_id]["replaceable"] = False
            index[asset_id]["risk"] = "do_not_touch"
        else:
            index[asset_id] = {
                "role_id": None,
                "role_label": None,
                "replaceable": False,
                "risk": "do_not_touch",
                "preserve_aspect": False,
                "preserve_silhouette": False,
                "asset_label": None,
            }
    return index


def scan_html(html: str, role_index: dict[str, dict]) -> list[dict]:
    rows = []
    for path, encoded, mime in IMG_RE.findall(html):
        asset_id = Path(path).stem
        blob = decode_base122(unescape(encoded))
        kind = sniff_kind(blob)
        width, height = image_size(kind, blob)
        role = role_index.get(asset_id, {})
        rows.append(
            {
                "asset_id": asset_id,
                "path": path,
                "mime": mime,
                "kind": kind,
                "bytes": len(blob),
                "width": width,
                "height": height,
                "role_id": role.get("role_id"),
                "role_label": role.get("role_label"),
                "asset_label": role.get("asset_label"),
                "replaceable": bool(role.get("replaceable", False)),
                "risk": role.get("risk", "unmapped"),
                "preserve_aspect": role.get("preserve_aspect"),
                "preserve_silhouette": role.get("preserve_silhouette"),
            }
        )
    rows.sort(key=lambda row: (not row["replaceable"], row.get("role_id") or "", row["asset_id"]))
    return rows


def write_artifact(rows: list[dict], roles: dict) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    replaceable = [row for row in rows if row["replaceable"]]
    payload = {
        "source_html": str(SOURCE_HTML.relative_to(TASK2_DIR.parent)),
        "image_count": len(rows),
        "replaceable_count": len(replaceable),
        "role_count": len(roles["asset_roles"]),
        "safe_override_count": len(roles["safe_overrides"]),
        "assets": rows,
    }
    ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_summary(rows: list[dict]) -> None:
    replaceable = [row for row in rows if row["replaceable"]]
    print(f"source: {SOURCE_HTML}")
    print(f"images: {len(rows)}")
    print(f"replaceable: {len(replaceable)}")
    print(f"artifact: {ARTIFACT_PATH}")
    print()
    print(f"{'id':8} {'role':18} {'kind':5} {'size':>11} {'risk':7} replace")
    for row in replaceable:
        size = f"{row['width']}x{row['height']}" if row["width"] else "?"
        print(
            f"{row['asset_id']:8} {row['role_id'] or '-':18} {row['kind']:5} "
            f"{size:>11} {row['risk']:7} yes"
        )


def main() -> None:
    if not SOURCE_HTML.exists():
        raise FileNotFoundError(f"Playable HTML not found: {SOURCE_HTML}")
    roles = load_roles()
    html = SOURCE_HTML.read_text(encoding="utf-8")
    rows = scan_html(html, build_role_index(roles))
    write_artifact(rows, roles)
    print_summary(rows)


if __name__ == "__main__":
    main()
