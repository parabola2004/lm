from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset

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
        help="output dateset directory",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    def gen():
        for text in load_texts(args.config):
            yield {"text": text}

    ds = Dataset.from_generator(gen)
    ds.save_to_disk(output_path)

    print(f"Dataset saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
