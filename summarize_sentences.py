import json
from transformers import pipeline

def load_sentences(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data['sentence1'], data['sentence2'], data['sentence3']

def summarize_sentences(sentence1, sentence2, sentence3):
    combined_text = f"{sentence1} {sentence2} {sentence3}"
    summarizer = pipeline("summarization")
    summary = summarizer(combined_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

def save_summary(output_file, summary_text):
    with open(output_file, 'w') as file:
        json.dump({"summary": summary_text}, file)

if __name__ == "__main__":
    input_file = 'input.json'
    output_file = 'output.json'

    sentence1, sentence2, sentence3 = load_sentences(input_file)
    summary_text = summarize_sentences(sentence1, sentence2, sentence3)
    save_summary(output_file, summary_text)
