from transformers import ViTForImageClassification, ViTImageProcessor, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os
from evaluate import load
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("💻 Training on:", device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "../models/vit-ai-detector")
DATASET_DIR = os.path.join(BASE_DIR, "../frames_dataset")
os.makedirs(MODEL_DIR, exist_ok=True)

dataset = load_dataset("imagefolder", data_dir=DATASET_DIR)
print("Dataset Loaded Successfully!")
print(f"Train Images: {len(dataset['train'])}\n")

# Random train-test split
dataset = dataset["train"].train_test_split(test_size=0.2)
dataset["validation"] = dataset["test"]
del dataset["test"]
print("Dataset Split: Train vs Validation")

names = dataset["train"].features["label"].names

print(f"Dataset Traing: \n{dataset["train"]}\n")

print(f"Dataset Train Label: \n{dataset["train"].features["label"]}\n")

label2id = {name: i for i, name in enumerate(names)}
id2label = {i: name for i, name in enumerate(names)}
print("Label mapping:", label2id)
processor = ViTImageProcessor.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(names),
    ignore_mismatched_sizes=True,
    label2id=label2id,
    id2label=id2label,
    )

def transform(example_batch):
    # print(f"Example: {example_batch}")
    images = example_batch["image"]
    inputs = processor(images=images, return_tensors="pt")
    example_batch["pixel_values"] = inputs["pixel_values"]
    return example_batch

dataset = dataset.with_transform(transform)

print(f"Dataset: \n{dataset}")

# Model setup
print("Loading ViT base model...")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.to(device)
print("Model Loaded")

accuracy_metric = load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    if isinstance(logits, np.ndarray):
        preds = np.argmax(logits, axis=-1)   # ✅ numpy version
    else:
        preds = logits.argmax(dim=-1).cpu().numpy()  # ✅ torch fallback
    return accuracy_metric.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=8,
    num_train_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    logging_dir="./logs",
    remove_unused_columns=False
)

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

print("Training Started...")
trainer.train()

trainer.save_model(MODEL_DIR)
print("Final Model Saved!")

# Performance metrics
metrics = trainer.evaluate()
print(f"\nEvaluation Results:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")