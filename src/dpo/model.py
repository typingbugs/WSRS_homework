from transformers import AutoModelForCausalLM, AutoTokenizer


def init_model(model_args):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)

    return model, tokenizer


if __name__ == "__main__":
    def test(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        test_data = {
            "prompt": "<HITMAN ABSOLUTION (PC)><The Orange Box - PC><Mass Effect><Mass Effect>",
            "chosen": "<Mass Effect 2 - PC>", 
            "rejected": "<Mass Effect - PC>"
        }
        print(tokenizer(test_data["prompt"], add_special_tokens=False)["input_ids"])
        print(tokenizer(test_data["chosen"], add_special_tokens=False)["input_ids"])
        print(tokenizer(test_data["rejected"], add_special_tokens=False)["input_ids"])

    test("model_configs/model_1")
        