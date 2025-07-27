import sys
import os
from transformers import BertTokenizer
import json

BERT_VOCAB = "bert-base-uncased"
MAX_SEQ_LENGTH = 128

# Converts input data into a format suitable for hierarchical sequential labeling (HSLN).
def write_in_hsln_format(input,hsln_format_txt_dirpath,tokenizer):


    final_string = ''
    filename_sent_boundries = {}
    for file in input:
        file_name=file['id']
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for annotation in file['annotations'][0]['result']:
            filename_sent_boundries[file_name]['sentence_span'].append([annotation['value']['start'],annotation['value']['end']])

            sentence_txt=annotation['value']['text']
            sentence_label = annotation['value']['labels'][0]
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=128)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + sentence_label + "\t" + sent_tokens_txt + "\n"
        final_string = final_string + "\n"
    with open(hsln_format_txt_dirpath , "w+") as file:
        file.write(final_string)


def tokenize_and_save(input_json_path, output_dir, file_suffix, tokenizer, label_type):
    """
    Tokenizes the input JSON data and saves it in HSLN format.

    Args:
    - input_json_path (str): Path to the input JSON file.
    - output_dir (str): Directory where the output file should be saved.
    - file_suffix (str): Suffix for the output file name (e.g., '_scibert').
    - tokenizer (BertTokenizer): BERT tokenizer instance.
    - label_type (str): The type of label to use (e.g., 'steps', 'category', 'rhetorical_function').
    """
    os.makedirs(output_dir, exist_ok=True)

    input_data = json.load(open(input_json_path))
    output_file_path = os.path.join(output_dir, f"{file_suffix}_scibert.txt")

    write_in_hsln_format(input_data, output_file_path, tokenizer)

    print(f"Tokenized data for {label_type} saved to: {output_file_path}")


def tokenize():
    """
    Main function to handle tokenization for 'steps', 'category', and 'rhetorical function'.
    """
    if len(sys.argv) != 5:
        print("Usage: python tokenize.py <train.json> <dev.json> <test.json> <label_type>")
        print("label_type should be one of: 'steps', 'category', 'rhetorical_function'")
        sys.exit(1)

    train_input_json, dev_input_json, test_input_json, label_type = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    # Ensure the label type is valid
    valid_label_types = ["steps", "category", "rhetorical_function"]
    if label_type not in valid_label_types:
        print(f"Invalid label_type '{label_type}'. Choose from {valid_label_types}.")
        sys.exit(1)

    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    # Define the base output directory for the specified label type
    base_output_dir = os.path.join("tokenized-files", label_type)

    # Tokenize and save train, dev, and test datasets
    tokenize_and_save(train_input_json, base_output_dir, "train", tokenizer, label_type)
    tokenize_and_save(dev_input_json, base_output_dir, "dev", tokenizer, label_type)
    tokenize_and_save(test_input_json, base_output_dir, "test", tokenizer, label_type)


# Entry point of the script
if __name__ == "__main__":
    tokenize()
