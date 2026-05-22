import os
import sys
import json
from transformers import BertTokenizer
from tqdm import tqdm

BERT_VOCAB = "bert-base-uncased"
MAX_SEQ_LENGTH = 128

def write_in_hsln_format(input_data, hsln_format_txt_dirpath, tokenizer):
    """
    Converts input data into HSLN-compatible format.

    Args:
    - input_data: List of input JSON data.
    - hsln_format_txt_dirpath: Path to save the HSLN-formatted file.
    - tokenizer: BERT tokenizer instance.
    """
    final_string = ''
    filename_sent_boundaries = {}

    for file in tqdm(input_data, desc="Processing files", unit="file"):
        file_name = file['id']
        final_string += f'###{file_name}\n'
        filename_sent_boundaries[file_name] = {"sentence_span": []}

        for annotation in file['annotations'][0]['result']:
            filename_sent_boundaries[file_name]['sentence_span'].append(
                [annotation['value']['start'], annotation['value']['end']]
            )

            sentence_txt = annotation['value']['text']
            sentence_label = annotation['value']['labels'][0]
            sentence_txt = sentence_txt.replace("\r", "")
            if sentence_txt.strip():
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=MAX_SEQ_LENGTH)
                sent_tokens_txt = " ".join(map(str, sent_tokens))
                final_string += f"{sentence_label}\t{sent_tokens_txt}\n"

        final_string += "\n"

    with open(hsln_format_txt_dirpath, "w") as file:
        file.write(final_string)

def tokenize_and_save(task_dir, output_dir, tokenizer):
    """
    Tokenizes the input JSON data and saves it in HSLN format.

    Args:
    - task_dir: Path to the directory containing `train.json`, `dev.json`, and `test.json`.
    - output_dir: Directory where the output files will be saved.
    - tokenizer: BERT tokenizer instance.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Locate train, dev, and test files
    file_suffixes = ["train", "dev", "test"]
    for suffix in file_suffixes:
        input_json_path = os.path.join(task_dir, f"{suffix}.json")
        if not os.path.exists(input_json_path):
            print(f"Warning: {suffix}.json not found in {task_dir}. Skipping.")
            continue

        input_data = json.load(open(input_json_path))
        output_file_path = os.path.join(output_dir, f"{suffix}_scibert.txt")

        write_in_hsln_format(input_data, output_file_path, tokenizer)

        print(f"Tokenized data for ({suffix}) saved to: {output_file_path}")

def tokenize():
    """
    Main function to handle tokenization for a given task.
    """
    if len(sys.argv) != 3:
        print("Usage: python process_dataset.py <datasets_dir> <output_dir>")
        print("datasets_dir: Directory containing subdirectories for each dataset")
        print("output_dir: Directory where tokenized files will be saved")
        sys.exit(1)

    datasets_dir = sys.argv[1]
    output_dir = sys.argv[2]
    print(f"datasets_dir: {datasets_dir}, output_dir: {output_dir}")
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)

    # Process each dataset in datasets_dir
    for dataset_name in os.listdir(datasets_dir):
        print(f"Processing {dataset_name}")
        dataset_path = os.path.join(datasets_dir, dataset_name)
        if os.path.isdir(datasets_dir):
            print("hellooooo")
            # Use the dataset name as the label_type
            label_output_dir = os.path.join(output_dir, dataset_name)
            tokenize_and_save(dataset_path, label_output_dir, tokenizer)

# Entry point of the script
if __name__ == "__main__":
    tokenize()
