import json
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def load_texts(input_file):
    with open(input_file, 'r') as file:
        data = json.load(file)
    return data['text_kerala'], data['text_rajasthan'], data['text_assam']

def summarize_texts(text_kerala, text_rajasthan, text_assam):
    combined_text = f"{text_kerala} {text_rajasthan} {text_assam}"
    
    # Initialize a tokenizer and stemmer
    tokenizer = Tokenizer("english")
    stemmer = Stemmer("english")
    
    # Parse the combined text
    parser = PlaintextParser.from_string(combined_text, tokenizer)
    
    # Initialize the LSA summarizer
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words("english")
    
    # Generate the summary
    summary = summarizer(parser.document, sentences_count=7)  # Adjust sentences_count as needed
    
    # Combine the summarized sentences into a single string
    summary_text = " ".join([str(sentence) for sentence in summary])
    
    return summary_text

def save_summary(output_file, summary_text):
    with open(output_file, 'w') as file:
        json.dump({"summary": summary_text}, file)

if __name__ == "__main__":
    input_file = 'input.json'
    output_file = 'output.json'

    text_kerala, text_rajasthan, text_assam = load_texts(input_file)
    summary_text = summarize_texts(text_kerala, text_rajasthan, text_assam)
    save_summary(output_file, summary_text)
