from datasets import load_dataset

# Load dataset in streaming mode
dataset = load_dataset(
    "jian-0/GenVidDet",
    split="train",
    streaming=True
)

print(dataset)
