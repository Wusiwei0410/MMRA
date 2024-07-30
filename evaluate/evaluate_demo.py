import base64
import csv
import json
import os

from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool

import datasets 
import ast
import argparse
from tqdm import tqdm

import json
from collections import Counter
import re
from collections import defaultdict
from transformers.image_utils import load_image

def extract_option_labels(text, options=None):
    if isinstance(text, dict):
        return "error"
    pattern = r"\(([A-D])\)"
    matches = re.findall(pattern, text)

    if not matches:
        pattern = r"\b([A-D])\b"
        matches = re.findall(pattern, text)

    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    else:
        if options:
            counter = Counter()
            for i, option in enumerate(options, start=1):
                label = chr(64 + i)
                option_stripped = option.strip()
                if option_stripped in text:
                    counter[label] += 1
                elif text in option:
                    counter[label] += 1
            if counter:
                most_common = counter.most_common()
                max_count = most_common[0][1]
                candidates = [item for item in most_common if item[1] == max_count]
                return candidates[-1][0]
    return None

def extract_yes_no(response):
    pattern = r'\b(yes|no)\b'
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    
    return None

def clients(client_name, args):
    if client_name == 'gpt4':
        return OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
        )

    else:
        raise ValueError('wrong client name')

def gpt_4_api(messages, model, args):
    client = clients('gpt4', args)
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content

def gpt_message(args, prompt, sample):
    def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

    if os.path.exists(f'./{args.model_name}_tmp/') == False:
        os.mkdir(f'./{args.model_name}_tmp/')

    sample['image1'].save(f'./{args.model_name}_tmp/image1.jpg')
    sample['image2'].save(f'./{args.model_name}_tmp/image2.jpg')
    image1 = f'./{args.model_name}_tmp/image1.jpg'
    image2 = f'./{args.model_name}_tmp/image2.jpg'

    if args.model_name == 'GPT4v':

        base64_image1 = encode_image(image_path=image1)
        base64_image2 = encode_image(image_path=image2)

        message = [
                {
                'role': 'assistant',
                'content':
                    [
                        {
                          "type": "text",
                          "text": f"{prompt}"
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}"
                          }
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image2}"
                          }
                        },
                    ]
                }
           ]

    elif args.model_name == 'Qwen-VL-Chat':
        message=f'<img>{image1}</img><img>{image2}</img> {prompt}'
    elif args.model_name == 'Phi3v':
        message = [ 
                {"role": "user", "content": f"<|image_1|>\n<|image_2|>\n{prompt}"}, 
            ]
    elif args.model_name in ['Idefics2', 'Mantis_Idefics2']:
        
        message = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": f"{prompt}"},
            ]
        },
    ]

    return message


def gen_prompt(sample):
    if sample['options'] != None:
        options = ast.literal_eval(sample['options'])
    else:
        options = None

    if sample['QA_type'] == 'Chooice QA' or sample['QA_type'] == 'Choice QA':
        if len(list(options.keys())) == 4:
            prompt = 'You will be giving one question, two images,  and four answers, one of them is correct. Please choose one of the four answers.\
                    please only answer the question with A, B, C or D.\
                    questions: {question} \
                    answer:  A: {A}  B: {B}  C: {C}  D: {D}\
                    Your answer is '.format(
                        question=sample['question'],
                        A=options['A'], B=options['B'], C=options['C'], D=options['D'])

        if len(list(options.keys())) == 3:
            prompt = 'You will be giving one question, two images,  and four answers, one of them is correct. Please choose one of the four answers.\
                    please only answer the question with A, B, C.\
                    questions: {question} \
                    answer:  A: {A}  B: {B}  C: {C}\
                    Your answer is '.format(
                        question=sample['question'],
                        A=options['A'], B=options['B'], C=options['C'])

    elif sample['QA_type'] == 'True or false QA':
        prompt = 'You will be giving one question and two images. Please answer the question using "Yes" or "No". \
                  Please only answer the question with Yes or No.\
                  questions: {question} \
                  Your answer is '.format(
                      question=sample['question']
                  )
    else:
        raise ValueError('wrong task type')

    return prompt

def gen_samples(args):
    data = datasets.load_dataset(args.data_name)['train']
    for item in data:
        yield item

def process_sample(args, sample, model = None, tokenizer = None):

    answer = sample['answer']

    prompt = gen_prompt(sample)
    message = gpt_message(args=args, prompt=prompt, sample=sample)

    # try to connect until success
    while True:
        if args.model_name == "GPT4v":
            try:
                option = gpt_4_api(message, 'gpt-4-vision-preview', args)
                break
            except Exception as e:
                print("error: ", e)
        elif args.model_name == "Qwen-VL-Chat":
            try:
                option, _ = model.chat(tokenizer, query=message, history=None)
                break
            except Exception as e:
                print("error: ", e)
        elif args.model_name == 'Phi3v':

            prompt = tokenizer.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

            inputs = tokenizer(prompt, [sample['image1'], sample['image2']], return_tensors="pt").to("cuda:0") 

            generation_args = { 
                "max_new_tokens": 1024, 
                "temperature": 0.0, 
                "do_sample": False,
                "top_p": None,
                "num_beams":1,
                "use_cache": True,
            } 

            generate_ids = model.generate(**inputs, pad_token_id = tokenizer.tokenizer.pad_token_id, eos_token_id=tokenizer.tokenizer.eos_token_id, **generation_args) 

            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            option = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
        elif args.model_name in ['Idefics2', 'Mantis_Idefics2']:
            prompt = tokenizer.apply_chat_template(message, add_generation_prompt=True)

            if os.path.exists(f'./{args.model_name}_tmp/') == False:
                os.mkdir(f'./{args.model_name}_tmp/')

            sample['image1'].save(f'./{args.model_name}_tmp/image1.jpg')
            sample['image2'].save(f'./{args.model_name}_tmp/image2.jpg')
            image1 = f'./{args.model_name}_tmp/image1.jpg'
            image2 = f'./{args.model_name}_tmp/image2.jpg'

            image1 = load_image(f'./{args.model_name}_tmp/image1.jpg')
            image2 = load_image(f'./{args.model_name}_tmp/image2.jpg')

            inputs = tokenizer(text=[prompt], images=[image1, image2], padding=True, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # Generate
            generated_ids = model.generate(**inputs,  max_new_tokens= 16, temperature= 0, do_sample= False)
            res = []
            for generated_id, input_id in zip(generated_ids, inputs["input_ids"]):
                generated_id = generated_id[len(input_id) :]
                generated_text = tokenizer.tokenizer.decode(generated_id, skip_special_tokens=True)
                res.append(generated_text)
            option = res[0]
        return  [sample, option]


def load_model(args):
    if args.model_name == 'Qwen-VL-Chat':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="cuda", trust_remote_code=True).eval()

        return model, tokenizer
    elif args.model_name == 'Phi3v':
        from transformers import AutoModelForCausalLM 
        from transformers import AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True) 
        return model, processor
    elif args.model_name in ['Idefics2', 'Mantis_Idefics2']:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from transformers.image_utils import load_image
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
        ).to("cuda")

        return model, processor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for MMRA benchmark")

    parser.add_argument("--dataset_name" , type = str , default = 'm-a-p/MMRA')
    parser.add_argument("--model_name" , type = str , default = 'Idefics2')
    # parser.add_argument("--model_name" , type = str , default = 'Qwen-VL-Chat')
    # parser.add_argument("--model_name" , type = str , default = 'Mantis_Idefics2')
    # parser.add_argument("--model_name" , type = str , default = 'Phi3v')
    parser.add_argument("--model_id" , type = str , default = 'HuggingFaceM4/idefics2-8b')
    # parser.add_argument("--model_id" , type = str , default = 'Qwen/Qwen-VL-Chat')
    # parser.add_argument("--model_id" , type = str , default = 'TIGER-Lab/Mantis-8B-Idefics2')
    # parser.add_argument("--model_id" , type = str , default = 'microsoft/Phi-3-vision-128k-instruct')
    parser.add_argument("--multiprocess" , type = int , default = 0)
    parser.add_argument("--API_KEY" , type = str , default = 'your api key')
    parser.add_argument("--BASE_URL" , type = str , default = 'base url')
    parser.add_argument("--client_name" , type = str , default = None)
    args = parser.parse_args()

    MMRA_data = datasets.load_dataset(args.dataset_name)['train']
    
    model, tokenizer = load_model(args)

    if args.multiprocess == 0:
        results = []
        
        for item in tqdm(MMRA_data):
            sample, option =  process_sample(args, item, model = model, tokenizer = tokenizer)
            answer = sample['answer']
            task = sample['Task']
            qa_type = sample['QA_type']
            if option != None:
                results.append({
                    'task': task,
                    'qa_type': qa_type,
                    'answer': answer,
                    'option': option,
                })
    else:
        process_sample = partial(process_sample, model_name=model_name, tokenizer = tokenizer)
        pool = ThreadPool(processes=16)
        results = pool.map(process_sample, gen_samples(args))

    score_dict = defaultdict(list)
    for line in results:
        gt=line['answer']
        task=line['task']
        qa_type=line['qa_type']
        response=line['option']
        if qa_type  == 'True or false QA':
            result=extract_yes_no(response)
        elif qa_type  == 'Choice QA':
            result=extract_option_labels(response)
        if result == None:
            score_dict[task].append(0.0)
        else:
            if gt.lower()==result.lower():
                score_dict[task].append(1.0)
            else:
                score_dict[task].append(0.0)


    print(score_dict)
    average_scores = {task: sum(scores) / len(scores) for task, scores in score_dict.items()}

    print("Score Dictionary:", score_dict)
    print("Average Scores:", average_scores)
