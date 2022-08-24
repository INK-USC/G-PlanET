import json
from transformers import AutoTokenizer
from random import shuffle

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


def build_dataset(train_file, test_file, sample_num):
    # take random examples from train_file
    output_file = test_file.replace(".json", "_prompt_{}.json".format(sample_num))
    write_f = open(output_file, "w", encoding="utf8")
    with open(train_file, "r", encoding="utf8") as read_train_f:
        train_examples = read_train_f.readlines()
    with open(test_file, "r", encoding="utf8") as read_test_f:
        test_examples = read_test_f.readlines()
    for test_example in test_examples:
        # take the input
        next_generate = True
        final_prompt = ""
        test_input = json.loads(test_example)["input"]
        while next_generate:
            final_prompt = ""
            shuffle(train_examples)
            prompt_format = "Input: {} Output: {} "
            for line in train_examples[: sample_num]:
                line_obj = json.loads(line)
                line_input, line_output = line_obj["input"], line_obj["output"]
                # randomly sample 5 examples
                final_prompt += prompt_format.format(line_input.strip(), line_output.strip())
            final_prompt += "Input: {} Output: ".format(test_input.strip())
            tokens = tokenizer.encode(final_prompt)
            if len(tokens) <= 750:
                next_generate = False
            print(len(tokens))
        write_f.write(final_prompt.replace("\n", " ") + "\n")
    write_f.close()


if __name__ == '__main__':
    build_dataset(TRAIN_PATH,
                  TEST_PATH, 5)