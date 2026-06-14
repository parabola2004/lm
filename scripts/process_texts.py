from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from lm.text_pass import load_texts


def main():
    parser = ArgumentParser(
        "Text Processing CLI Tool",
        description="Load a text pass config and write results to file[s].",
    )

    parser.add_argument(
        "config",
        help="path to text pass JSON config",
    )
    parser.add_argument(
        "output",
        help="output directory (default) or file (with -s/--single)",
    )
    parser.add_argument(
        "-s",
        "--single",
        action="store_true",
        help="treat output as a single file, joining all results",
    )
    parser.add_argument(
        "--sep",
        default="\n",
        help='separator string when using --single (default: "\n")',
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    if args.single:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            count = 0
            sep = ""
            for text in tqdm(load_texts(args.config), unit="texts"):
                f.write(sep)
                f.write(text)
                sep = args.sep
                count += 1
        print(f"Wrote {count} result(s) to {output_path.resolve()}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0
        for i, text in enumerate(tqdm(load_texts(args.config), unit="texts"), start=1):
            file_path = output_path / f"{i}.txt"
            file_path.write_text(text, encoding="utf-8")
            count = i
        print(f"Wrote {count} file(s) to {output_path.resolve()}")


if __name__ == "__main__":
    main()
