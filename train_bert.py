from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoModelForSequenceClassification
import torch
import os
import evaluate
import numpy as np


seed = 42


def collator():
    def return_func(batch):
        words = [item["data"] for item in batch]
        words = [sequence[:500] if len(sequence) > 500 else sequence for sequence in words]
        words = tokenizer(words, return_tensors="pt", padding=True, trunation=True).to(device)
        gener = [item["generated"] for item in batch]
        sequence_class = [item["model"] for item in batch]
        return words, gener, sequence_class
    return return_func


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def process_batch(batch, model):
    pass


def main(model):
    pass


if __name__ == "__main__":

    pre_train = "distilbert/distilbert-base-uncased"

    device = torch.device("cuda")

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    tokenizer = AutoTokenizer.from_pretrained(pre_train)

    if not os.path.isdir("dataset\\ahma"):
        dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")['train']
        dataset.save_to_disk("dataset\\ahma")
        exit()

    if os.path.isdir("dataset\\tokenized\\ahma"):
        dataset = load_from_disk("dataset\\tokenized\\ahma")
    else:
        dataset = load_from_disk("dataset\\ahma")
        print(len(dataset))
        print(dataset[0])
        working_with = 300_000 / len(dataset)
        dataset = dataset.train_test_split(test_size=working_with, seed=seed, shuffle=False)["test"]
        dataset = dataset.remove_columns(["model"])
        dataset = dataset.rename_column("data", "text")
        dataset = dataset.rename_column("generated", "label")
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        valid_percent = .15
        test_percent = .15
        non_train = valid_percent + test_percent
        train_test_valid = tokenized_dataset.train_test_split(test_size=non_train, seed=seed, shuffle=False)
        # Split (valid and test) into train and test
        new_test_size = test_percent / non_train
        test_valid = train_test_valid['test'].train_test_split(test_size=new_test_size, seed=seed, shuffle=False)

        train_test_valid_dataset = DatasetDict({
            'train': train_test_valid['train'].shuffle(seed=seed),
            'validation': test_valid['train'].shuffle(seed=seed),
            'test': test_valid['test'].shuffle(seed=seed),
        })

        train_test_valid_dataset.save_to_disk("dataset\\tokenized\\ahma")
        exit()

    model = AutoModelForSequenceClassification.from_pretrained(pre_train, num_labels=2,
                                                               id2label=id2label, label2id=label2id).to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    accuracy = evaluate.load("accuracy")
    training_args = TrainingArguments(
        output_dir="bert/800_000_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
