from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
import torch
import os
import evaluate
import numpy as np

from bert_classifier import BERTClassifier
from loading_bar import LoadingBar
from time import time
from transformers import pipeline


seed = 42
BATCH_SIZE = 16


def collator(tokenizer):
    def return_func(batch):
        tokens = [tokenizer(tokenizer.decode(item['input_ids']), return_tensors='pt', truncation=True, padding=True).to(device) for item in batch]
        gener = [item["label"] for item in batch]
        tokens = tokenizer.batch_encode_plus(tokens)
        tokens.to(device)
        return tokens, gener
    return return_func


def collator_other():
    def return_func(batch):
        words = [tokenizer.decode(item['input_ids'][1:-1]) for item in batch]
        gener = [item['label'] for item in batch]
        return words, gener
    return return_func


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train(trainer):
    trainer.train()


def proper_dataset(pipeline, tokenizer):
    def collator_internal():
        def return_func(batch):
            words = [item["data"] for item in batch]
            gener = [item["generated"] for item in batch]
            real_words = []
            real_gener = []
            for i, wordy in enumerate(words):
                if 100 <= len(wordy) <= 400:
                    real_words.append(wordy)
                    real_gener.append(gener[i])
            return real_words, real_gener

        return return_func

    def process(input_batch, _is_generated):
        results = pipeline(input_batch)
        _total_labeled_human_correctly = 0
        _total_labeled_human = 0
        _total_labeled_ai = 0
        _ai_labeled_ai = 0
        _humans_labeled_ai = 0
        _total_humans_labeled_ai = 0
        for i, result in enumerate(results):
            if result["label"] == "HUMAN":
                if _is_generated[i] == 0:
                    _total_labeled_human_correctly += 1
                else:
                    _total_humans_labeled_ai += 0
                _total_labeled_human += 1
            else:  # GENERATED
                if _is_generated[i] == 1:
                    _ai_labeled_ai += 1
                else:
                    _humans_labeled_ai += 1
                _total_labeled_ai += 1
        return _total_labeled_human_correctly, _total_humans_labeled_ai, _total_labeled_human, _ai_labeled_ai, _humans_labeled_ai, _total_labeled_ai

    if os.path.isdir("dataset\\ahma"):
        sequences = load_from_disk("dataset\\ahma")
    else:
        sequences = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")['train']
        sequences.save_to_disk("dataset\\ahma")
        exit()
    if not sequences:
        exit()

    batch_size = 16

    total_labeled_human_correctly = 0
    total_labeled_human = 0
    total_humans_labeled_ai = 0
    total_labeled_ai = 0
    ai_labeled_ai = 0
    humans_labeled_ai = 0
    amount_do_max = int(1_000 * (batch_size / 64))
    # amount_do = 400
    # amount_do = 2
    # main_pipe = pipeline("sentiment-analysis", model=main_model, tokenizer=tokenizer)

    sequences_gpt: Dataset = sequences.filter(lambda seq: seq["model"] == "GPT4")
    data_loader = DataLoader(sequences_gpt.shuffle(seed=seed), batch_size=batch_size, collate_fn=collator_internal())
    amount_do = min(len(data_loader), amount_do_max)

    for batch, (inp_text, is_generated) in enumerate(data_loader):
        if batch >= amount_do:
            break
        t_1, t_2, t_3, t_4, t_5, t_6 = process(inp_text, is_generated)

        total_labeled_human_correctly += t_1
        total_humans_labeled_ai += t_2
        total_labeled_human += t_3
        ai_labeled_ai += t_4
        humans_labeled_ai += t_5
        total_labeled_ai += t_6

    print("done 1")
    sequences_wiki: Dataset = sequences.filter(lambda seq: seq["model"] == "wikipedia")
    data_loader = DataLoader(sequences_wiki.shuffle(seed=seed), batch_size=batch_size, collate_fn=collator_internal())
    amount_do = min(len(data_loader), amount_do_max)
    for batch, (inp_text, is_generated) in enumerate(data_loader):
        if batch >= amount_do:
            break
        t_1, t_2, t_3, t_4, t_5, t_6 = process(inp_text, is_generated)

        total_labeled_human_correctly += t_1
        total_humans_labeled_ai += t_2
        total_labeled_human += t_3
        ai_labeled_ai += t_4
        humans_labeled_ai += t_5
        total_labeled_ai += t_6
    print("done 2")
    sequences_claude: Dataset = sequences.filter(lambda seq: seq["model"] == "claude")
    data_loader = DataLoader(sequences_claude.shuffle(seed=seed), batch_size=batch_size, collate_fn=collator_internal())
    amount_do = min(len(data_loader), amount_do_max)
    for batch, (inp_text, is_generated) in enumerate(data_loader):
        if batch >= amount_do:
            break
        t_1, t_2, t_3, t_4, t_5, t_6 = process(inp_text, is_generated)

        total_labeled_human_correctly += t_1
        total_humans_labeled_ai += t_2
        total_labeled_human += t_3
        ai_labeled_ai += t_4
        humans_labeled_ai += t_5
        total_labeled_ai += t_6
    print('done 3')
    sequences_opus: Dataset = sequences.filter(lambda seq: seq["model"] == "Claude3-Opus")
    data_loader = DataLoader(sequences_opus.shuffle(seed=seed), batch_size=batch_size, collate_fn=collator_internal())
    amount_do = min(len(data_loader), amount_do_max)
    for batch, (inp_text, is_generated) in enumerate(data_loader):
        if batch >= amount_do:
            break
        t_1, t_2, t_3, t_4, t_5, t_6 = process(inp_text, is_generated)

        total_labeled_human_correctly += t_1
        total_humans_labeled_ai += t_2
        total_labeled_human += t_3
        ai_labeled_ai += t_4
        humans_labeled_ai += t_5
        total_labeled_ai += t_6
    print('done 4')
    sequences_gemini: Dataset = sequences.filter(lambda seq: seq["model"] == "gemini-1.5-pro")
    data_loader = DataLoader(sequences_gemini.shuffle(seed=seed), batch_size=batch_size, collate_fn=collator_internal())
    amount_do = min(len(data_loader), amount_do_max)
    for batch, (inp_text, is_generated) in enumerate(data_loader):
        if batch >= amount_do:
            break
        t_1, t_2, t_3, t_4, t_5, t_6 = process(inp_text, is_generated)

        total_labeled_human_correctly += t_1
        total_humans_labeled_ai += t_2
        total_labeled_human += t_3
        ai_labeled_ai += t_4
        humans_labeled_ai += t_5
        total_labeled_ai += t_6
    print('done 5')
    precision = total_labeled_human_correctly / total_labeled_human
    recall = total_labeled_human_correctly / (total_labeled_human_correctly + total_humans_labeled_ai)

    f1 = (2 * precision * recall) / (precision + recall)
    print(f"Precision: {precision}\n"
          f"Recall   : {recall}\n"
          f"F1       : {f1}\n"
          f"Num humans labeled human : {total_labeled_human_correctly}\n"
          f"Num AI labeled human     : {total_labeled_human - total_labeled_human_correctly}\n"
          f"Num AI labeled AI        : {total_labeled_ai - humans_labeled_ai}\n"
          f"Num humans_labeled AI    : {humans_labeled_ai}")


def other_infer(pipeline, data_loader, amount_do):
    amount_do = min(amount_do, len(data_loader))
    bar = LoadingBar(time(), 100, amount_do, batch_size=BATCH_SIZE)
    total_labeled_human_correctly = 0
    total_labeled_human = 0
    total_humans_labeled_ai = 0
    total_labeled_ai = 0
    ai_labeled_ai = 0
    humans_labeled_ai = 0
    for batch, (inp_words, labels) in enumerate(data_loader):
        if batch >= amount_do:
            break
        print(bar.update(batch), end="")
        results = pipeline(inp_words)
        for i, result in enumerate(results):
            if result["label"] == "HUMAN":
                if labels[i] == 0:
                    total_labeled_human_correctly += 1
                else:
                    total_humans_labeled_ai += 0
                total_labeled_human += 1
            else:   # Positive
                if labels[i] == 1:
                    ai_labeled_ai += 1
                else:
                    humans_labeled_ai += 1
                total_labeled_ai += 1
    print(bar.finish())
    # precision = true_positives / total_positives
    # recall = true_positives / (true_positives + (total_positives - true_negatives))

    precision = total_labeled_human_correctly / total_labeled_human
    recall = total_labeled_human_correctly / (total_labeled_human_correctly + total_humans_labeled_ai)

    f1 = (2 * precision * recall) / (precision + recall)
    print(f"Precision: {precision}\n"
          f"Recall   : {recall}\n"
          f"F1       : {f1}\n"
          f"Num humans labeled human : {total_labeled_human_correctly}\n"
          f"Num AI labeled human     : {total_labeled_human - total_labeled_human_correctly}\n"
          f"Num AI labeled AI        : {total_labeled_ai - humans_labeled_ai}\n"
          f"Num humans_labeled AI    : {humans_labeled_ai}")


if __name__ == "__main__":
    pre_train = "distilbert/distilbert-base-uncased"

    device = torch.device("cuda")

    id2label = {0: "HUMAN", 1: "GENERATED"}       #
    label2id = {"HUMAN": 0, "GENERATED": 1}

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

    # main_model = AutoModelForSequenceClassification.from_pretrained(pre_train, num_labels=2,
    #                                                                 id2label=id2label, label2id=label2id).to(device)
    local = "bert\\model\\"
    # main_model = AutoModelForSequenceClassification.from_pretrained(local, num_labels=2,
    #                                                                 id2label=id2label, label2id=label2id).to(device)

    main_model = BERTClassifier(local).to(device)
    main_model.update_model_from_checkpoint(1)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    accuracy = evaluate.load("accuracy")
    # training_args = TrainingArguments(
    #     output_dir="bert/800_000_model",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     num_train_epochs=2,
    #     weight_decay=0.01,
    #     eval_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     push_to_hub=False,
    # )
    # main_trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["validation"],
    #     processing_class=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )
    # train(main_trainer)

    # test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=collator(tokenizer))
    # infer(main_model, tokenizer, test_loader)
    main_pipe = pipeline("sentiment-analysis", model=main_model, tokenizer=tokenizer, device=device)
    test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, collate_fn=collator_other())
    #  amount_do = (64 / BATCH_SIZE) * 1_000
    amount_do = len(test_loader)
    other_infer(main_pipe, test_loader, amount_do)

    print("MINE IMPLEMENTATION NOW")

    proper_dataset(main_pipe, tokenizer)
