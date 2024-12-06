from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict
from bert_classifier import BERTClassifier
import torch
from torch import optim, tensor, nn, float32, no_grad
import os
from time import time
from my_secrets import WORKING_DIR
from loading_bar import LoadingBar
from numpy import mean

seed = 42
BATCH_SIZE = 16
# BATCH_SIZE = 32
TRAIN_STEPS = 2
pre_train = "distilbert/distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(pre_train)

BERT_BASE_DIR = f"{WORKING_DIR}/bert/model"
BERT_BASE_FILENAME = f"{BERT_BASE_DIR}/checkpoints"

EPOCH_FILE = f"{BERT_BASE_DIR}/epochs.txt"
LOSSES_FILE = f"{BERT_BASE_DIR}/train_losses.txt"
VALIDATION_LOSSES_FILE = f"{BERT_BASE_DIR}/val_losses.txt"


def collator():
    def return_func(batch):
        words = [item["data"] for item in batch]
        gener = tensor([[item["generated"] ]for item in batch], dtype=float32)
        return words, gener
    return return_func


def train(model, epochs, train_loader, val_loader, optimizer, objective, device):
    print(f"Batches in epoch: {len(train_loader)}")

    if not os.path.isfile(EPOCH_FILE):
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write("0\n")

    for epoch in range(epochs):
        print(f"DOING EPOCH {epoch}")
        model.train()
        train_loss = -1.0
        bar = LoadingBar(time(), 100, len(train_loader), batch_size=BATCH_SIZE)
        for batch, (x, y_truth) in enumerate(train_loader):
            print(bar.update(batch), end="")
            x = tokenizer(x, return_tensors="pt", padding=True, truncation=True)
            x, y_truth = x.to(device), y_truth.to(device)
            optimizer.zero_grad()
            y_hat = model(x)

            loss = objective(y_hat, y_truth)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            del loss
        print(bar.finish())
        model.save(epoch)

        with open(f"{LOSSES_FILE}", "a+") as f:
            f.write(f"{epoch}, {train_loss}\n")

        print(f"Just did {epoch} epochs\nStarting evaluation.")
        bar = LoadingBar(time(), 100, len(val_loader), batch_size=BATCH_SIZE)
        with no_grad():
            temp_val = []
            for batch_index, (x_l, y_l) in enumerate(val_loader):
                print(bar.update(batch_index), end="")
                x_l = tokenizer(x_l, return_tensors="pt", padding=True, truncation=True)
                x_v, y_v = x_l.to(device),  y_l.to(device)
                temp_val.append(objective(model(x_v), y_v).item())
            print(bar.finish())
            val = mean(temp_val)
        with open(f"{EPOCH_FILE}", "w") as f:
            f.write(f"{epoch}\n")

        with open(f"{VALIDATION_LOSSES_FILE}", "a+") as f:
            f.write(f"{epoch}, {val}\n")


def get_dataset():
    dataset = load_from_disk("dataset\\ahma")
    working_with = 300_000 / len(dataset)
    dataset = dataset.train_test_split(test_size=working_with, seed=seed, shuffle=False)["test"]
    valid_percent = .15
    test_percent = .15
    non_train = valid_percent + test_percent
    train_test_valid = dataset.train_test_split(test_size=non_train, seed=seed, shuffle=False)
    new_test_size = test_percent / non_train
    test_valid = train_test_valid['test'].train_test_split(test_size=new_test_size, seed=seed, shuffle=False)

    dataset = DatasetDict({
        'train': train_test_valid['train'].shuffle(seed=seed),
        'validation': test_valid['train'].shuffle(seed=seed),
        'test': test_valid['test'].shuffle(seed=seed),
    })
    return dataset


def main():
    device = torch.device("cuda")

    if not os.path.isdir("dataset\\ahma"):
        dataset = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")['train']
        dataset.save_to_disk("dataset\\ahma")
        exit()
    dataset = get_dataset()

    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, collate_fn=collator())
    validation_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE, collate_fn=collator())

    model = BERTClassifier(BERT_BASE_DIR).to(device)

    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    objective = nn.BCEWithLogitsLoss()

    train(model, TRAIN_STEPS, train_loader, validation_loader, optimizer, objective, device)


if __name__ == "__main__":
    now = time()
    try:
        main()
        print("##################################################")
        print(f"Time taken = {time() - now}")
    except Exception as e:
        print("##################################################")
        print(f"Time taken = {time() - now}")
        raise e
