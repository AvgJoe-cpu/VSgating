from datasets import load_dataset


def main():
    ds = load_dataset("avgJo3/fineweb-subset-10M", split="train")
    print(ds)

if __name__ == "__main__":
    main()