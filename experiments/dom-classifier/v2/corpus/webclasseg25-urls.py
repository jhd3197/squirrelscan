"""Extract page_id -> source URL from the WebClasSeg25 MHTML archive.

The released HTML (and the HuggingFace mirror of it) carries no URL: the parquet
columns are page_id, html, topic and language, and only 4% of pages have a
`<base href>` left after the MHTML-to-HTML conversion. The URL survives only in
the MHTML wrapper, as the `Snapshot-Content-Location` header Chrome writes.

That URL is not a nicety. Without it there is no registrable domain, and
without the domain there is no way to check whether one of these 500 public
service sites is also a domain in OUR val or holdout split. A research page
from a holdout domain would contaminate the split that grades the result, and
tagging the row `train` does not undo that.

Only the first few KB of each member is read, so the 1.5 GB archive is streamed
rather than extracted.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

LOCATION = re.compile(rb"^Snapshot-Content-Location:\s*(\S+)", re.MULTILINE)
#: The header sits in the first lines of the file; 8 KiB is generous.
HEAD_BYTES = 8192


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, help="mhtml.zip from the Zenodo record")
    parser.add_argument("--out", required=True, help="where to write page-urls.json")
    args = parser.parse_args()

    urls: dict[str, str] = {}
    missing: list[str] = []
    with zipfile.ZipFile(args.archive) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".mhtml")]
        for name in members:
            page_id = Path(name).stem
            with archive.open(name) as handle:
                head = handle.read(HEAD_BYTES)
            match = LOCATION.search(head)
            if match:
                urls[page_id] = match.group(1).decode("utf-8", "replace")
            else:
                missing.append(page_id)

    Path(args.out).write_text(json.dumps(urls, indent=0, sort_keys=True) + "\n")
    hosts = {u.split("/")[2] for u in urls.values() if "//" in u}
    print(
        json.dumps(
            {
                "members": len(members),
                "withUrl": len(urls),
                "withoutUrl": len(missing),
                "distinctHosts": len(hosts),
                "out": args.out,
            }
        )
    )


if __name__ == "__main__":
    main()
