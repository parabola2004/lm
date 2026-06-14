from argparse import ArgumentParser

from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def action_info(path: str):
    tokenizer = Tokenizer.from_file(path)
    print(f"Model:      {type(tokenizer.model).__name__}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(
        f"Pre-tok:    {type(tokenizer.pre_tokenizer).__name__ if tokenizer.pre_tokenizer else 'None'}"
    )
    print(
        f"Decoder:    {type(tokenizer.decoder).__name__ if tokenizer.decoder else 'None'}"
    )
    special_tokens = [
        t for t in tokenizer.get_added_tokens_decoder().values() if t.special
    ]
    if special_tokens:
        print(f"Specials:   {', '.join(t.content for t in special_tokens)}")


def action_init(path: str):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFD()
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.save(path, pretty=True)
    print(f"Initialized tokenizer at {path}.")


def action_train(path: str, texts_path: str):
    tokenizer = Tokenizer.from_file(path)
    trainer = BpeTrainer(special_tokens=["<unk>", "<pad>", "<sot>", "<eot>"])
    ds = load_from_disk(texts_path)
    tokenizer.train_from_iterator((x["text"] for x in ds), trainer)
    tokenizer.save(path, pretty=True)
    print(f"Saved tokenizer at {path}.")


def action_encode(path: str, text: str, *, show_id: bool = False):
    tokenizer = Tokenizer.from_file(path)
    output = tokenizer.encode(text)
    if show_id:
        print(output.ids)
    else:
        print(output.tokens)


def create_parser() -> ArgumentParser:
    parser = ArgumentParser("Vocabulary Management CLI Tool", fromfile_prefix_chars="@")

    subparsers = parser.add_subparsers(dest="action", help="actions", required=True)

    # info
    info_parser = subparsers.add_parser("info", help="show tokenizer info")
    info_parser.add_argument("path", help="tokenizer JSON file")

    # init
    init_parser = subparsers.add_parser("init", help="initialize tokenizer")
    init_parser.add_argument("path", help="tokenizer JSON file")

    # train
    train_parser = subparsers.add_parser("train", help="train tokenizer")
    train_parser.add_argument("path", help="tokenizer JSON file")
    train_parser.add_argument("texts_path", help="path to dataset directory")

    # encode
    encode_parser = subparsers.add_parser("encode", help="encode text to tokens")
    encode_parser.add_argument("path", help="tokenizer JSON file")
    encode_parser.add_argument("text", help="text to encode")
    encode_parser.add_argument("--id", action="store_true", help="show token ids")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    match args.action:
        case "info":
            action_info(args.path)
        case "init":
            action_init(args.path)
        case "train":
            action_train(args.path, args.texts_path)
        case "encode":
            action_encode(args.path, args.text, show_id=args.id)


if __name__ == "__main__":
    main()
