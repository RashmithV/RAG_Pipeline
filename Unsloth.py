from unsloth import FastLanguageModel

# From Zurab
# Model Parameters
MODEL_PATH = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_SEQUENCE_LENGTH = 2048  # Choose any! We auto support RoPE Scaling internally!
DATA_TYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage. Can be False.
MAX_NEW_TOKENS = 275  # Maximum number of tokens the model should generate.

#From Zurab
print(f"Loading model {MODEL_PATH}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dtype=DATA_TYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# From Zurab
def infer(input_prompt: str) -> str:
    tokenized_input = tokenizer([input_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **tokenized_input,
        max_new_tokens=2048,
        use_cache=True,
        pad_token_id=model.config.eos_token_id
    )
    return tokenizer.batch_decode(outputs)[0][len(input_prompt) + len(tokenizer.bos_token):-len(tokenizer.eos_token)].strip()