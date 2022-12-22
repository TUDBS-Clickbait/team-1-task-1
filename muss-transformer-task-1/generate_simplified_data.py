#!/usr/bin/env python3
import argparse
import asyncio
import json
from asyncio import create_task

from muss_wrapper.simplify import init_simplifier, simplify_sentences

simplify_fields = ["targetTitle", "targetParagraphs"]

model_name = 'muss_en_wikilarge_mined'


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=True)

    return parser.parse_args()


async def simplify_stcs(sentences, was_str, i, f):
    #source_path = get_temp_filepath()
    #write_lines(sentences, source_path)
    simplified = simplify_sentences(sentences)
    print(simplified)
    return simplified
    #return read_lines(pred_path), was_str, i, f

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
            was_str = False
            data_simple = []

            if isinstance(data_compl, str):
                was_str = True
                data_compl = [data_compl]

            tasks.append(create_task(simplify_stcs(data_compl, was_str, i, f)))

    while tasks:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        data_simple, was_str, i, f = done.pop().result()

        print("Finished 1 task.")

        if was_str:
            data_simple = data_simple[0]

        data_new[i][f] = data_simple


    with open(output_file, 'w') as out:
        for line in data_new:
            out.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    init_simplifier('muss_en_wikilarge_mined')
    args = parse_args()
    asyncio.run(simplify_data(args.input, args.output))

