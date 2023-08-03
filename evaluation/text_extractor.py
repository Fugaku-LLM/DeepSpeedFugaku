import argparse
from typing import Literal
import json
import random


def save_text(texts: list[str], output: str) -> None:
    with open(output, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(f"{i}\t{text}\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--text-max-len", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--random-choice", action="store_true")
    args = parser.parse_args()

    return args


def is_meet_num_samples(num_samples: int, samples: list[str]) -> bool:
    if len(samples) >= num_samples:
        return True
    else:
        return False


def main() -> None:
    args: argparse.Namespace = get_args()
    sample_texts: list[str] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data: dict[Literal["id", "text", "revid", "url", "title"], str] = json.loads(line)
            text: str = data["text"]

            if len(text) >= args.text_max_len * 2:
                if args.random_choice:
                    if random.randint(0, 1000) % 10 == 0:
                        sample_texts.append(text[: args.text_max_len])
                else:
                    sample_texts.append(text[: args.text_max_len])

            if is_meet_num_samples(num_samples=args.num_samples, samples=sample_texts):
                save_text(texts=sample_texts, output=args.output)
                break

    print("Done")


if __name__ == "__main__":
    main()
