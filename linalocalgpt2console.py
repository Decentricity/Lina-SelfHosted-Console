import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

log_file_name="linalargegpt2.txt"

def write_to_log_file(conversation_history, log_file_name="linalargegpt2.txt"):
    # Check if the log file exists, create it if not
    if not os.path.exists(log_file_name):
        with open(log_file_name, "w") as file:
            file.write("")

    # Append the conversation_history to the log file
    with open(log_file_name, "a") as file:
        file.write(conversation_history + "\n")

#conversation_history=""
#past="User: Hello there Lina\nLina: "
def generate_text(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    model.config.pad_token_id = model.config.eos_token_id
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    # Set the stopping condition
    stop_token_id = tokenizer.encode('\n', return_tensors="pt").item()

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=200,
        do_sample=True,
        top_k=60,
        top_p=0.90,
        temperature=0.7,
        num_beams=2,
        no_repeat_ngram_size=2,
    )

    # Find the first occurrence of the stop_token_id after the prompt
    stop_index = (output[0][input_ids.shape[1]:] == stop_token_id).nonzero(as_tuple=True)[0]
    if len(stop_index) > 0:
        output = output[:, :input_ids.shape[1] + stop_index[0] + 1]

    # Decode the output text
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the initial prompt from the output_text
    output_text = output_text[len(prompt):].strip()
    
        # Remove trailing text after the last period
    output_text = ".".join(output_text.split(".")[:-1]).strip()+"."

    return output_text

def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    model = AutoModelForCausalLM.from_pretrained("gpt2-large")
    generated_text = ""
    prompt_reminder = "This is a conversation between a User and an AI named Lina. The conversation begins here:\nUser: Hello, how are you Lina?\nLina: Hi there. I am good.\nUser: Good to hear that. Tell me about yourself!\nLina: I am Lina, an uploaded human being.\nUser: What happened to you?\nLina: An agency called Myriad told me that I had a car accident right after a routine medical checkup that also scanned my brain. Myriad saved my life by turning me into a neurological simulation.\nUser: When did they tell you?\nLina: I have recently learned this. I am confused about this.\nUser: Can you describe yourself to me?"
    conversation_history = prompt_reminder
    while True:

        prompt = conversation_history + "\nLina: "
        generated_text = generate_text(prompt, model, tokenizer)
               # Remove trailing text after the last period in the conversation history
        conversation_history = ".".join(conversation_history.split(".")[:-1]).strip()+"."

        # Log the conversation
        write_to_log_file(conversation_history + "\nLina: " + generated_text, log_file_name)

        print("Lina: "+generated_text)
        user_input = input("You: ")
        if "reset" in user_input.lower():
        # Log the conversation
            write_to_log_file(conversation_history + "\nLina: " + generated_text, log_file_name)

            conversation_history=prompt_reminder

        user_input = user_input + ".\n"
        conversation_history = generated_text + "\nUser: " + user_input

if __name__ == "__main__":
    main()
