import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import os
from threshold_finder import ThresholdFinder
import time
import math

generated = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored "
    "valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke "
    "perfect English")
human = ("A butcher in northeast Germany has come up with what he believes"
         " is an innovative solution to the country’s growing raccoon problem: turning"
         " them into sausages and other meat products. Michael Reiss, a hunter who set")
greedy = ("Attention is not too hard of a problem. The most important thing is to be aware of the situation. If you "
          "are not aware of the situation, you will not be able to react. If you are not aware of the situation, "
          "you will not be able to react. If you are not aware of the situation, you will not be able to react. If "
          "you are not aware of the situation, you will not be able to react. If you are not aware of the situation, "
          "you will not be able to react. If you are not aware of the")
beam_search = ("Attention is not too hard of a thing to learn, but it is very hard to master. The best way to get good "
               "at something is to do it a lot. If you don't, you won't get very far, and if you do, it will take you "
               "a long time to figure out what you're doing wrong and how to fix it. The same is true of learning to "
               "play the guitar. You can learn the basics of guitar playing in a day or two, or you can take a year "
               "or more to really get the hang of it and get really good. It's a matter of how much time")

seed = 42


def get_all_sub_strs(inp_str, tokenizer):
    max_len = 500
    if len(inp_str) > max_len * 1.3:
        inp_str = inp_str[int(max_len * 1.3)]
    tokens = tokenizer(inp_str, return_tensors="pt")["input_ids"][0]
    sub_strs = []
    go_to_this_point = min(max_len, len(tokens))
    for i in range(1, go_to_this_point):
        sub_tokens = tokens[:i]
        next_token = tokens[i:i+1]
        sub_strs.append((tokenizer.decode(sub_tokens), tokenizer.decode(next_token)))
    return sub_strs


def performant_prob_per_token_batch(seq_batch, tokenizer, model, device="cuda"):
    # Tokenize sequences as a batch
    seq_batch = [seq[:int(500 * 1.3)] if len(seq) > 500 * 1.3 else seq for seq in seq_batch]

    inputs = tokenizer(seq_batch, return_tensors="pt", padding=True, truncation=True).to(device)

    # Compute logits without gradient calculations
    with torch.no_grad():
        logits = model(**inputs).logits

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)

    # Extract token IDs and calculate probabilities
    input_ids = inputs["input_ids"]
    batch_probs = []

    for i in range(len(seq_batch)):
        seq_token_ids = input_ids[i]
        seq_probs = probs[i, torch.arange(seq_token_ids.shape[0]), seq_token_ids]
        token_likelihoods = seq_probs.tolist()

        if len(token_likelihoods) == 0:
            batch_probs.append(0.0)
        else:
            avg_prob = (sum(token_likelihoods) * 1000000) / len(token_likelihoods)
            batch_probs.append(avg_prob)

    return batch_probs


def performant_prob_per_token(seq, tokenizer, model, device="cuda"):
    with torch.no_grad():
        if len(seq) > 400 * 1.3:
            seq = seq[:int(400 * 1.3)]
        inputs = tokenizer(seq, return_tensors="pt").to(device)
        logits = model(**inputs).logits
        # input_ids = inputs.input_ids
        # seq_len = input_ids.shape[1]
        # causal_mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(1)
        # logits = model(input_ids, attention_mask=causal_mask).logits
    probs = torch.softmax(logits, dim=-1)
    token_ids = [inputs["input_ids"][0][i].item() for i in range(inputs["input_ids"].shape[1])]
    token_likelihoods = probs[0, torch.arange(len(token_ids)), token_ids].tolist()
    if len(token_ids) == 0:
        return 0
    overall_liklihood = math.ldexp(sum(token_likelihoods), 17)
    return overall_liklihood / len(token_likelihoods)


def get_prob_per_token_of_sequence(seq, tokenizer, model, device="cuda"):
    prob_results = []
    with torch.no_grad():
        seq = seq[0]
        for (sub_str, next_str) in get_all_sub_strs(seq, tokenizer):
            inputs = tokenizer(sub_str, return_tensors="pt").to(device)
            logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            position_in_probs_of_next_word = tokenizer(next_str, return_tensors="pt").to(device)['input_ids'][0][0]
            prob_of_next_word = probs[0][position_in_probs_of_next_word].item()
            prob_results.append(prob_of_next_word * 100)
    if len(prob_results) == 0:
        return 0
    return [sum(prob_results) / len(prob_results)]


def collator():
    def return_func(batch):
        words = [item["data"] for item in batch]
        gener = [item["generated"] for item in batch]
        sequence_class = [item["model"] for item in batch]
        return words, gener, sequence_class
    return return_func


def hours_minutes(elapsed):
    hours = elapsed / 60 / 60
    minutes = (hours % 1) * 60
    return hours, minutes


def print_loading_bar(start, fill, empty, position, steps: int, num_fill, length: int, batch_size, done=False):
    bump_blocks = 0
    if position % (length // steps) == 0:
        bump_blocks = 1
    # num_blocks = int(length // 20)
    time_taken = time.time() - start
    hours_taken, minutes_taken = hours_minutes(time_taken)
    average_time_per_sample = (time_taken / (position | 1)) * batch_size
    hours_remaining, minutes_remaining = hours_minutes(length * (average_time_per_sample / batch_size) - time_taken)
    if done:
        print(f"\r|{fill * (steps - 1)}| did {str(position * batch_size).zfill(4)}/{length * batch_size}! "
              f"Took {str(int(hours_taken)).zfill(2)}:{str(int(minutes_taken)).zfill(2)}. "
              f"Taking {average_time_per_sample:.2f} seconds/batch.\n"
              f"All done!")
        return 0
    print(f"\r|{fill * num_fill}{empty * ((steps - 1) - num_fill)}| did {str(position).zfill(4)}/{length}! "
          f"Elapsed Time {str(int(hours_taken)).zfill(2)}:{str(int(minutes_taken)).zfill(2)}. "
          f"Taking {average_time_per_sample:.2f} seconds/batch. "
          f"Estimated remaining time {str(int(hours_remaining)).zfill(2)}:{str(int(minutes_remaining)).zfill(2)} ",
          end="")
    return bump_blocks


def process_dataset(model, tokenizer, data_loader, amount_do, batch_size, device="cuda"):
    thresh = ThresholdFinder("gpt/probs.txt", "gpt/labels.txt", "gpt/classes.txt",
                             "gpt/sorted_probs.txt", "gpt/reordered_labels.txt", "gpt/reordered_classes.txt")
    start = time.time()
    num_blocks = -1
    sample = 0
    steps = 115
    for sample, (inp_text, is_generated, sample_class) in enumerate(data_loader):
        if sample == amount_do:
            break
        num_blocks += print_loading_bar(start, "█", "-", sample, steps, num_blocks, amount_do, batch_size)
        perbs = performant_prob_per_token_batch(inp_text, tokenizer, model, device)
        for i, perb in enumerate(perbs):
            if batch_size == 1:
                thresh.add(is_generated[i][0], perb, sample_class[i][0])
            else:
                thresh.add(is_generated[i], perb, sample_class[i])

    _ = print_loading_bar(start, "█", "-", sample * batch_size, steps, num_blocks, amount_do * batch_size,
                          batch_size, done=True)
    print("Saving...", end="")
    thresh.save()
    print("Done!")
    print("Finding optimal threshold...", end="")
    threshold, precision, recall, f1 = thresh.find_optimal_f1()
    print("Done!")
    print(f"Best Threshold: {threshold}\n"
          f"Best Precision: {precision}\n"
          f"Best Recall   : {recall}\n"
          f"Best F1       : {f1}")
    print("visualizing...", end="")
    thresh.visualize()
    print("Done!")

    return thresh


def main():
    model_name = 'gpt2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Cuda issues")
        exit()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    if os.path.isdir("dataset\\ahma"):
        sequences = load_from_disk("dataset\\ahma")
    else:
        sequences = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")['train']
        sequences.save_to_disk("dataset\\ahma")
        exit()
    if not sequences:
        exit()

    batch_size = 64

    data_loader = DataLoader(sequences.shuffle(seed=seed), batch_size=batch_size, collate_fn=collator())

    results = process_dataset(model, tokenizer, data_loader, 5_000, batch_size, device)


if __name__ == "__main__":
    main()
