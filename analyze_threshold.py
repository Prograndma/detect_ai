import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
import os
from threshold_finder import ThresholdFinder

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


def get_all_sub_strs(inp_str, tokenizer, device):
    sub_strs = []
    inputs = tokenizer(inp_str, return_tensors="pt").to(device)
    tokens = inputs["input_ids"]
    go_to_this_point = min(1000, len(tokens[0]))
    for i in range(go_to_this_point):
        sub_strs.append((tokenizer.decode(tokens[0][:i]), tokenizer.decode(tokens[0][i])))
    return sub_strs[1:]


def get_prob_per_token_of_sequence(seq, tokenizer, model, device="cuda"):
    prob_results = []
    with torch.no_grad():
        for (sub_str, next_str) in get_all_sub_strs(seq, tokenizer, device):
            inputs = tokenizer(sub_str, return_tensors="pt").to(device)
            logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            position_in_probs_of_next_word = tokenizer(next_str, return_tensors="pt").to(device)['input_ids'][0][0]
            prob_of_next_word = probs[0][position_in_probs_of_next_word].item()
            prob_results.append(prob_of_next_word * 100)
    if len(prob_results) == 0:
        return 0
    return sum(prob_results) / len(prob_results)


def collator():
    # todo: I'll want to keep track of the classes, not just generated or not, to see if my model works better on some
    #  models or others.
    def return_func(batch):
        words = [item["data"] for item in batch]
        gener = [item["generated"] for item in batch]
        return words, gener
    return return_func


def process_dataset(model, tokenizer, data_loader, device="cuda"):
    thresh = ThresholdFinder("probs.txt", "labels.txt")
    for sample, (inp_text, is_generated) in enumerate(data_loader):
        if num_blocks := (sample % (len(data_loader) // 20)) == 0:
            block = "█"
            dash = "-"
            print(f"\r|{block * num_blocks}{dash * (20 - num_blocks)}|", end="")
        for inp in inp_text:
            perb = get_prob_per_token_of_sequence(inp, tokenizer, model, device)
            thresh.add(is_generated, perb)
    thresh.save_probs()
    thresh._reorganize()
    print(" DONE!")


def main():
    model_name = 'gpt2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Cuda issues")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    print(f"generated:\t{get_prob_per_token_of_sequence(generated, tokenizer, model, device)}")
    print(f"human    :\t{get_prob_per_token_of_sequence(human, tokenizer, model, device)}")
    print(f"greedy   :\t{get_prob_per_token_of_sequence(greedy, tokenizer, model, device)}")
    print(f"beam     :\t{get_prob_per_token_of_sequence(beam_search, tokenizer, model, device)}")
    if os.path.isdir("dataset\\ahma"):
        sequences = load_from_disk("dataset\\ahma")
    else:
        sequences = load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")['train']
        sequences.save_to_disk("dataset\\ahma")
        exit()

    if sequences == False:
        exit()

    data_loader = DataLoader(sequences.shuffle(seed=seed), batch_size=1, collate_fn=collator())

    results = process_dataset(model, tokenizer, data_loader, device)
    results_file = "results.txt"
    with open(f"{results_file}", "w") as f:
        f.write(results)



if __name__ == "__main__":
    main()


# MAJOR TODO:: So I'm pretty sure, not confident, but preeeeetty sure that the model will output logits for every token
#  passed in, which could speed my code up by like, a thousand times. Literally a thousand times. Since I wouldn't have
#  to pass in each sequence n times (n being the amount of tokens in the sequence.) Also, here's another major speed up.
#  I'm preeeetty sure that if I can do this, I could do batches as well. So. Things to look into. Right now this is slow
#  as balls. I'll never get through the dataset like this. After hours I still haven't gotten through like a 20th of it.
#  Rip.