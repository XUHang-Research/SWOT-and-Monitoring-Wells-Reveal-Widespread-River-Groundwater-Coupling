#!/usr/bin/env python3
"""Create exact 1000 m and 3000 m nested tables in one pass over a 5000 m TSV."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-1000", type=Path, required=True)
    parser.add_argument("--output-3000", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_1000.parent.mkdir(parents=True, exist_ok=True)
    args.output_3000.parent.mkdir(parents=True, exist_ok=True)
    temporary_1000 = Path(str(args.output_1000) + ".tmp")
    temporary_3000 = Path(str(args.output_3000) + ".tmp")

    total = kept_1000 = kept_3000 = 0
    with ExitStack() as stack:
        source = stack.enter_context(
            args.input.open("r", encoding="utf-8-sig", newline="")
        )
        out_1000 = stack.enter_context(
            temporary_1000.open("w", encoding="utf-8-sig", newline="")
        )
        out_3000 = stack.enter_context(
            temporary_3000.open("w", encoding="utf-8-sig", newline="")
        )
        header_line = source.readline()
        if not header_line:
            raise ValueError(f"Input file is empty: {args.input}")
        columns = header_line.rstrip("\r\n").split("\t")
        try:
            distance_index = columns.index("distance_prior_bank_m")
        except ValueError as exc:
            raise KeyError("Input table lacks distance_prior_bank_m") from exc
        out_1000.write(header_line)
        out_3000.write(header_line)

        for line_number, line in enumerate(source, start=2):
            total += 1
            values = line.rstrip("\r\n").split("\t")
            if len(values) != len(columns):
                raise ValueError(
                    f"Unexpected field count at line {line_number:,}: {len(values)} != {len(columns)}"
                )
            try:
                distance = float(values[distance_index])
            except ValueError as exc:
                raise ValueError(
                    f"Unparseable distance_prior_bank_m at line {line_number:,}"
                ) from exc
            if distance <= 3000.0:
                out_3000.write(line)
                kept_3000 += 1
                if distance <= 1000.0:
                    out_1000.write(line)
                    kept_1000 += 1

    temporary_1000.replace(args.output_1000)
    temporary_3000.replace(args.output_3000)

    print(f"5000-m input records: {total:,}")
    print(f"3000-m retained records: {kept_3000:,} -> {args.output_3000}")
    print(f"1000-m retained records: {kept_1000:,} -> {args.output_1000}")


if __name__ == "__main__":
    main()
