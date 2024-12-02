import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd

model_name = 'gpt2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("Cuda issues")
    exit()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

seq = 'In a shocking finding, scientists discovered a herd of unicorns living' \
 ' in a remote, previously unexplored valley, in the Andes Mountains. Even ' \
 'more surprising to the researchers was the fact that the unicorns spoke ' \
 'perfect English'

second_seq = "A butcher in northeast Germany has come up with what he believes" \
" is an innovative solution to the country’s growing raccoon problem: turning" \
" them into sausages and other meat products. Michael Reiss, a hunter who set"
print("\nInput sequence: ")
print(seq)

print(f"Length of generated sequence: {len(seq.split(' '))}")
print(f'length of real human text: {len(second_seq.split(" "))}')

def make_string(w):
  str = ""
  for word in w:
    str += f"{word} "
  return str

def get_all_sub_strs(str):
  words = str.split(" ")
  sub_strs = []

  for i in range(1, len(words)):
    sub_strs.append( (make_string(words[:i]), words[i]) )
  return sub_strs

def get_probs_of_seq(seq):
  prob_results = []
  with torch.no_grad():
    for (sub_seq, next_word) in get_all_sub_strs(seq):
      inputs = tokenizer(sub_seq, return_tensors="pt").to(device)
      logits = model(**inputs).logits[:, -1, :]
      probs = torch.softmax(logits, dim=-1)
      position_in_probs_of_next_word = tokenizer(next_word, return_tensors="pt").to(device)['input_ids'][0][0]
      prob_of_next_word = probs[0][position_in_probs_of_next_word].item()
      prob_results.append((prob_of_next_word * 100, next_word))
  return prob_results

import math

def performant_prob_per_token(seq, tokenizer, model, device="cuda"):
    prob_results = []
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
    for i, token_likelihood in enumerate(token_likelihoods):
        prob_results.append((token_likelihood * 100000, tokenizer.decode(token_ids[i])))
    if len(token_ids) == 0:
        return 0
    return prob_results


probs_generated = get_probs_of_seq(seq)
probs_human = get_probs_of_seq(second_seq)



print(probs_generated)
print(probs_human)


probs_generated = performant_prob_per_token(seq, tokenizer, model)
probs_human = performant_prob_per_token(second_seq, tokenizer, model)
print(probs_generated)
print(probs_human)
exit()
get_all_sub_strs(seq)
inputs = tokenizer(seq, return_tensors="pt").to(device)
print("\nTokenized input data structure: ")
print(inputs)
word = tokenizer("researchers", return_tensors="pt").to(device)
print(f'token for word researchers: {word["input_ids"][0][0]}')
input_ids = inputs["input_ids"]  # just IDS, no attn mask
print("\nToken IDs and their words: ")
for id in input_ids[0]:
  word = tokenizer.decode(id)
  print(id, word)
with torch.no_grad():
  print(model(**inputs).logits.shape)
  logits = model(**inputs).logits[:, -1, :]
print("\nAll logits for next word: ")
print(logits)
print(logits.shape)


probs = torch.softmax(logits, dim=-1)
print("\nAll probabilities: ")
print(probs)
pred_id = torch.argmax(logits).item()
pred_word = tokenizer.decode(pred_id)
pd.DataFrame([pred_id, logits[0, pred_id].cpu(), probs[0, pred_id].cpu(), pred_word],
              index=['Token ID', 'Logits', 'Probability', 'Predicted Word'], columns =['Value'])
import pandas as pd

#input_txt = "Transformers are the"
input_txt = "Transformers are the "                    # This one is interesting to see how things change
#input_txt = "Transformers are built using the"         # Note the word pieces

input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
iterations = []
n_steps = 10
choices_per_step = 5

with torch.no_grad():
    for _ in range(n_steps):
        iteration = dict()
        iteration["Input"] = tokenizer.decode(input_ids[0])
        output = model(input_ids=input_ids)

        # Select logits of the first batch and the last token and apply softmax to get the probability
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)

        # Store tokens with highest probabilities in our little table
        for choice_idx in range(choices_per_step):
            token_id = sorted_ids[choice_idx]
            token_prob = next_token_probs[token_id].cpu().numpy()
            token_choice = (
                f"{tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)"
            )
            iteration[f"Choice {choice_idx+1}"] = token_choice
        iterations.append(iteration)


        # Append predicted next token to input
        input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)

pd.DataFrame(iterations)

input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))
max_length = 128
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
input_txt = "Attention is not too hard of"
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
output_greedy = model.generate(input_ids, max_length=max_length, do_sample=False)
print(tokenizer.decode(output_greedy[0]))
# encode context the generation is conditioned on

input_ids = tokenizer.encode(input_txt, return_tensors='pt').to(device)

# Some scoring functions

import torch.nn.functional as F

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label

def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(
            output.logits[:, :-1, :], labels[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()

# generate text until the output length (which includes the context length) reaches 50
greedy_output = model.generate(input_ids, max_length=50)

logp = sequence_logprob(model, greedy_output, input_len=len(input_ids[0]))
print(tokenizer.decode(greedy_output[0]))
print(f"\nlog-prob: {logp:.2f}")
