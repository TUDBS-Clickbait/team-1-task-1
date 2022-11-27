#!/usr/bin/env python3
import argparse
import json
import asyncio
from asyncio import create_task

from simplifier import models
from simplifier import config
from simplifier import simplifier

config.lang = "en"
models.embeddings = models.load_embeddings(config.lang)

simplify_fields = ["targetTitle", "targetParagraphs"]

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=True)

    return parser.parse_args()

async def simplify_txt(txt, was_str, i, f, list_index=None):
    print("Complex: " + txt)
    
    simplified_text = ""

    while len(txt) > 512: 
        short_text = txt[:512]
        txt = txt[512:]
        simplified_text += simplifier.simplify_text(short_text)

    simplified_text += simplifier.simplify_text(txt)

    print("Simple: " + simplified_text)

    return simplified_text, was_str, i, f, list_index

async def simplify_data(input_file, output_file):
    data = []

    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    data_new = data.copy()

    tasks = []

    for i, d in enumerate(data):
        for f in simplify_fields:
            data_compl = d[f]
            data_simple = []

            if isinstance(data_compl, list):
                for i2, part in enumerate(data_compl):
                    tasks.append(create_task(simplify_txt(part, False, i, f, i2)))
            elif isinstance(data_compl, str):
                tasks.append(create_task(simplify_txt(data_compl, True, i, f)))

            data_new[i][f] = data_simple

    while tasks:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        txt_simple, was_str, i, f, list_index = done.pop().result()

        print("Finished 1 task.")

        if not was_str:
            data_new[i][f][list_index] = txt_simple
        else:
            data_new[i][f] = data_simple

            
    with open(output_file, 'w') as out:
        for line in data_new:
            out.write(json.dumps(line) + '\n')



if __name__ == '__main__':
    args = parse_args()
    asyncio.run(simplify_data(args.input, args.output))
