# LLM Text Checker

**A quick and lightweight tool to test whether text is generated by a Large Language Model (LLM) or written by a human.**

---

## 🚀 Features
- **Fast and Easy to Use**: Paste your text and get results instantly.
- **Personally Created Algorithm**: I was inspired by a friends masters project, but made some key changes.
- **No Finetuning required**: This approach requires no dataset, and zero fine-tuning. 
- **Interactive Google Colab Notebook**: No installation required to test it out!
- **Lightweight**: Could be easily modified to run on device.

---

## 📚 How It Works
This tool uses the GPT2 model and tokenizer to analyze the liklihood of text as decided by GPT2, this analysis can provide a faily accurate guess as to whether or not text is generated.

---

## 🛠️ Getting Started

### 1. Open the Google Colab Notebook
Click the link below to access the interactive notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com//github/Prograndma/detect_ai/blob/main/Detecting_AI.ipynb)


### 2. Run the Notebook
1. Click on the "Run All" button or execute each cell step-by-step.
2. Paste your text in the designated input field.
3. View the results to see if the text is AI-generated or human-written.

---

## 🔧 Requirements
- A Google account to access Google Colab.
- Internet connection.

---

## 📖 Example
1. Input: 
   ```
   Now, after almost two centuries, the mystery of what sparked this odd phenomenon is finally solved. Dr. Jorge Pérez, an evolutionary biologist from the University of La Paz, and several companions, were exploring the Andes Mountains when they found a small valley, with no other animals or humans. Pérez noticed that the valley had what appeared to be a natural fountain, surrounded by two peaks of rock and silver snow.
   ```
2. Output: 
   ```
   Might Be Generated
   ```

---

## 🤝 Contributing
Contributions are welcome! Feel free to submit a pull request or open an issue for feedback and suggestions.

---

## 🌟 Acknowledgments
- Inspired by the growing need to differentiate between AI-generated and human-written content.
- Thanks to the open-source community for making tools like this possible.
