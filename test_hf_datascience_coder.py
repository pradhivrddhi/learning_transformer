from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def build_instruction_prompt(instruction):
    return '''
    You are the Data Science Coder, a helpful AI assistant created by a man named Ed.
    You help people with data science coding and you answer questions about data science in a helpful manner.
    ### Instruction:
    {}
    ### Response:
    '''.format(instruction.strip()).lstrip()

print(build_instruction_prompt("Perform EDA on the Iris dataset"))
tokenizer = AutoTokenizer.from_pretrained("ed001/datascience-coder-1.3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("ed001/datascience-coder-1.3b", trust_remote_code=True).cuda()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024, top_p=0.95)
result = pipe(build_instruction_prompt("Perform EDA on the Iris dataset"))
print(result[0]['generated_text'])
