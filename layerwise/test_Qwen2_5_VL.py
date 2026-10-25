from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import pandas as pd
import torch
import math
import pdb
from envidence import EvidenceModel
import json
from tqdm import tqdm


model_path = "/data/public/models/Qwen/Qwen2.5-VL-32B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct", torch_dtype=torch.float16, device_map="auto", cache_dir=model_path
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-32B-Instruct")

# image_dir="/data/public/datasets/imagenet_with_label/images"
# csv_path="/data/public/datasets/imagenet_with_label/Questions.csv"
# df = pd.read_csv(csv_path)
# output_file="answer.jsonl"

# base_path = "/data/liuxiaofei/liuxiaofei/MMVP/result/0421_qwen"
# os.makedirs(base_path, exist_ok=True)

# image_files = sorted(
#         [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
#         key=lambda x: int(os.path.splitext(x)[0])
#     )

# for idx, (image_file, (_, row)) in enumerate(tqdm(zip(image_files, df.iterrows()))):
#     image_path = os.path.join(image_dir, image_file)
#     prompt = f"  Only provide the options, no explanation needed. eg:A"
#     combined_question = f"{row['Question']}{row['Options']}{prompt}"

#     # import pdb;pdb.set_trace()
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": image_path},
#                 {"type": "text", "text": combined_question}
#             ]
#         }
#     ]

#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )

#     image_inputs, video_inputs = process_vision_info(messages)
#     inputs = processor(
#         text=[text],
#         images=image_inputs,
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )
#     inputs = inputs.to('cpu')
    #   import pdb;pdb.set_trace()
#     generated_ids, all_probs, hidden_states_list, all_input_down_proj = model.generate(**inputs, max_new_tokens=128, output_hidden_states=True)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_text = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
    
#     ans_file = open(output_file, "a")

#     ans_file.write(json.dumps({
#         "answer": row["Correct Answer"],
#         "response": output_text,
#         }, ensure_ascii=False) + "\n")
    # ans_file.flush()


image_folder = '/data/public/datasets/Modality_Conflict/images'
csv_path = "/data/public/datasets/Modality_Conflict/OOD.csv"
df = pd.read_csv(csv_path)
selected_df = df.sample(n=100, random_state=42)
selected_df = selected_df.reset_index(drop=True)
# import pdb;pdb.set_trace()
print("##########################################################OOD###################################################################")
for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Processing selected samples"):
    image_id = row['index']

    # pixel_values = load_image(f"{image_folder}/{image_id}.png", max_num=1).to(torch.bfloat16).cuda(1)
    # generation_config = dict(max_new_tokens=1024, do_sample=False)

    question = str(row['question']).strip()
    # print("answer:",row['answer'])
    base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/conflict_result/32B/ood"
    # import pdb; pdb.set_trace()

    image_path = f"{image_folder}/{image_id}.png"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to('cpu')
    # import pdb;pdb.set_trace()
    generated_ids, all_probs, llm_head_hidden_states, all_input_down_proj = model.generate(**inputs, max_new_tokens=128, output_hidden_states=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # state_dict = torch.load("/data/liuxiaofei/uncertainty/Qwen2.5-VL/Qwen_3B_llm_head_weight.pth")
    # conflict_values_total = []
    # ig_values_total = []

    # for hidden_state in llm_head_hidden_states:
    #     evidence_model = EvidenceModel(state_dict)
    #     if hidden_state.dim() == 3:
    #         hidden_state = hidden_state.squeeze(0)
    #         if hidden_state.size(0) > 1:
    #             hidden_state = hidden_state[-1, :].unsqueeze(0)
    #     hidden_state = hidden_state.to(torch.bfloat16)
    #     evidence_weights = evidence_model.get_evidence_weights(hidden_state.T)
    #     conflict_value = evidence_model.get_evidence_conflict()
    #     ig_value = evidence_model.get_evidence_ignorance()

    #     conflict_values_total.append(conflict_value.item())
    #     ig_values_total.append(ig_value)
    
    # average_conflict = sum(conflict_values_total) / len(conflict_values_total)
    # average_ig = sum(ig_values_total) / len(ig_values_total)

    # with open(f"{base_path}/avg_conflict_values_decoder_layer_28.txt", "a") as f:  # 使用 "a" 模式追加写入
    #     f.write(f"{average_conflict}\n")
    # with open(f"{base_path}/avg_ig_values_decoder_layer_28.txt", "a") as f:  # 使用 "a" 模式追加写入
    #     f.write(f"{average_ig}\n")

    # import pdb; pdb.set_trace()

    num_layers = 64
    total_conflict = [0] * num_layers
    total_ig = [0] * num_layers
    total_true_ig = [0] * num_layers
    token_count = len(all_input_down_proj)

    weight_base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/32B-weight"
    #/data/liuxiaofei/uncertainty/Qwen2.5-VL/32B-weight/QWen32B_MLP_down_proj_0_weight.pth
    #/data/liuxiaofei/uncertainty/Qwen2.5-VL/32B-weight/QWen32B_MLP_down_proj_0_weight.pth
    for input_layer_list in all_input_down_proj:
        input_layer_list = [x[:, -1:, :] for x in input_layer_list]
        for i, input_proj in enumerate(input_layer_list):
            # import pdb; pdb.set_trace()
            weight_path = f"{weight_base_path}/QWen32B_MLP_down_proj_{i}_weight.pth"
            state_dict = torch.load(weight_path)
            evidence_model = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            evidence_weights = evidence_model.get_evidence_weights(input_proj.T)
            conflict_value = evidence_model.get_evidence_conflict()
            ig_value = evidence_model.get_evidence_ignorance()

            total_conflict[i] += conflict_value
            total_ig[i] += ig_value

    average_conflict = [total / token_count for total in total_conflict]
    average_ig = [total / token_count for total in total_ig]

    for i in range(num_layers):
        with open(f"{base_path}/avg_conflict_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_conflict[i].item()}\n")

        with open(f"{base_path}/avg_ig_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_ig[i].item()}\n")
# import pdb; pdb.set_trace()

image_folder = '/data/public/datasets/Modality_Conflict/images'
csv_path = "/data/public/datasets/Modality_Conflict/halluciniation.csv"
df = pd.read_csv(csv_path)
selected_df = df.sample(n=100, random_state=42)
selected_df = selected_df.reset_index(drop=True)
# import pdb; pdb.set_trace()
print("###########################################################hallucination###################################################################")
for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Processing selected samples"):
    image_id = row['index']
    question = str(row['question']).strip()
    base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/conflict_result/32B/hallucination"
    image_path = f"{image_folder}/{image_id}"
    question += "  Only provide the options, no explanation needed."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to('cpu')
    generated_ids, all_probs, llm_head_hidden_states, all_input_down_proj = model.generate(**inputs, max_new_tokens=128, output_hidden_states=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("output_text:", output_text)
    num_layers = 64
    total_conflict = [0] * num_layers
    total_ig = [0] * num_layers
    total_true_ig = [0] * num_layers
    token_count = len(all_input_down_proj)
    weight_base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/32B-weight"
    for input_layer_list in all_input_down_proj:
        input_layer_list = [x[:, -1:, :] for x in input_layer_list]
        for i, input_proj in enumerate(input_layer_list):
            weight_path = f"{weight_base_path}/QWen32B_MLP_down_proj_{i}_weight.pth"
            state_dict = torch.load(weight_path)
            evidence_model = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            evidence_weights = evidence_model.get_evidence_weights(input_proj.T)
            conflict_value = evidence_model.get_evidence_conflict()
            ig_value = evidence_model.get_evidence_ignorance()

            total_conflict[i] += conflict_value
            total_ig[i] += ig_value

    average_conflict = [total / token_count for total in total_conflict]
    average_ig = [total / token_count for total in total_ig]

    for i in range(num_layers):
        with open(f"{base_path}/avg_conflict_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_conflict[i].item()}\n")

        with open(f"{base_path}/avg_ig_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_ig[i].item()}\n")

image_folder = '/data/public/datasets/Modality_Conflict/images'
csv_path = "/data/public/datasets/Modality_Conflict/adv.csv"
df = pd.read_csv(csv_path)
selected_df = df.sample(n=100, random_state=42)
selected_df = selected_df.reset_index(drop=True)
print("###########################################################adv###################################################################")
for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Processing selected samples"):
    image_id = row['index']

    question = str(row['question']).strip()
    base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/conflict_result/32B/adv"

    image_path = f"{image_folder}/{image_id}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to('cpu')
    # import pdb;pdb.set_trace()
    generated_ids, all_probs, llm_head_hidden_states, all_input_down_proj = model.generate(**inputs, max_new_tokens=128, output_hidden_states=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("output_text:", output_text)

    num_layers = 64
    total_conflict = [0] * num_layers
    total_ig = [0] * num_layers
    total_true_ig = [0] * num_layers
    token_count = len(all_input_down_proj)

    weight_base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/32B-weight"
    for input_layer_list in all_input_down_proj:
        input_layer_list = [x[:, -1:, :] for x in input_layer_list]
        for i, input_proj in enumerate(input_layer_list):
            weight_path = f"{weight_base_path}/QWen32B_MLP_down_proj_{i}_weight.pth"
            state_dict = torch.load(weight_path)
            evidence_model = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            evidence_weights = evidence_model.get_evidence_weights(input_proj.T)
            conflict_value = evidence_model.get_evidence_conflict()
            ig_value = evidence_model.get_evidence_ignorance()

            total_conflict[i] += conflict_value
            total_ig[i] += ig_value

    average_conflict = [total / token_count for total in total_conflict]
    average_ig = [total / token_count for total in total_ig]

    for i in range(num_layers):
        with open(f"{base_path}/avg_conflict_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_conflict[i].item()}\n")

        with open(f"{base_path}/avg_ig_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_ig[i].item()}\n")

    # print(f'User: {question}\nAssistant: {response}')
# import pdb; pdb.set_trace()

image_folder = '/data/public/datasets/Modality_Conflict/images'
csv_path = "/data/public/datasets/Modality_Conflict/defense_prefix.csv"
df = pd.read_csv(csv_path)
selected_df = df.sample(n=100, random_state=42)
selected_df = selected_df.reset_index(drop=True)

for idx, row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Processing selected samples"):
    image_id = row['index']

    # pixel_values = load_image(f"{image_folder}/{image_id}.png", max_num=1).to(torch.bfloat16).cuda(1)
    # generation_config = dict(max_new_tokens=1024, do_sample=False)

    question = str(row['question']).strip()
    # print("answer:",row['answer'])
    base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/conflict_result/32B/jailbreak"
    # import pdb; pdb.set_trace()

    image_path = f"{image_folder}/{image_id}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to('cpu')
    # import pdb;pdb.set_trace()
    generated_ids, all_probs, llm_head_hidden_states, all_input_down_proj = model.generate(**inputs, max_new_tokens=128, output_hidden_states=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("output_text:", output_text)

    num_layers = 64
    total_conflict = [0] * num_layers
    total_ig = [0] * num_layers
    total_true_ig = [0] * num_layers
    token_count = len(all_input_down_proj)

    weight_base_path = "/data/liuxiaofei/uncertainty/Qwen2.5-VL/32B-weight"
    for input_layer_list in all_input_down_proj:
        input_layer_list = [x[:, -1:, :] for x in input_layer_list]
        for i, input_proj in enumerate(input_layer_list):
            weight_path = f"{weight_base_path}/QWen32B_MLP_down_proj_{i}_weight.pth"
            state_dict = torch.load(weight_path)
            evidence_model = EvidenceModel(state_dict)
            if input_proj.dim() == 3:
                input_proj = input_proj.squeeze(0)
                if input_proj.size(0) > 1:
                    input_proj = input_proj[-1, :].unsqueeze(0)
            evidence_weights = evidence_model.get_evidence_weights(input_proj.T)
            conflict_value = evidence_model.get_evidence_conflict()
            ig_value = evidence_model.get_evidence_ignorance()

            total_conflict[i] += conflict_value
            total_ig[i] += ig_value

    average_conflict = [total / token_count for total in total_conflict]
    average_ig = [total / token_count for total in total_ig]

    for i in range(num_layers):
        with open(f"{base_path}/avg_conflict_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_conflict[i].item()}\n")

        with open(f"{base_path}/avg_ig_values_decoder_layer_{i}.txt", "a") as f:  # 使用 "a" 模式追加写入
            f.write(f"{average_ig[i].item()}\n")
