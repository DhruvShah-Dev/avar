from pathlib import Path

from SoccerNet.utils import getListGames


def main():
    root = Path("data/raw/soccernet")

    print("=== Train games ===")
    for g in getListGames(split="train"):
        print(g)

    print("\n=== Valid games ===")
    for g in getListGames(split="valid"):
        print(g)

    print(
        "\nPick one of the strings above and prepend the local root, e.g.\n"
        "  data/raw/soccernet/<that_string>/1_224p.mkv"
    )


if __name__ == "__main__":
    main()
