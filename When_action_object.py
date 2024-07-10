import pandas as pd
import json
from openai import AsyncOpenAI
import asyncio
import time
import difflib
import random
import re
from collections import OrderedDict


# change variable here

# %%
# Read the json file and load it into a pandas dataframe
def load_data(file_path):
    """
    Load the json file and return a pandas dataframe
    """
    # Load the json file
    with open(file_path) as f:
        data = json.load(f)

    # Load the data into a pandas dataframe
    df = pd.DataFrame(data)

    return df

# Read the json file 'tvqa_plus_val.json'
val_df = load_data('tvqa_plus_val.json')
# Read the json file './tvqa_plus_train.json'
train_df = load_data('tvqa_plus_train.json')
# Print the length
print(len(val_df))


# %%

# Extract the "bbox" columns and transform it into a list
val_bbox_list = list(val_df['bbox'])
train_bbox_list = list(train_df['bbox'])
# Extract the "vid_name" columns and transform it into a list
val_vid_names_list = list(val_df['vid_name'])
train_vid_names_list = list(train_df['vid_name'])
# Extract the "qid" columns and transform it into a list
val_qid_list = list(val_df['qid'])
train_qid_list = list(train_df['qid'])
# Verify it by printing the first element in the list
val_answer_idx_list = list(val_df['answer_idx'])
val_a0_list = list(val_df['a0'])
val_a1_list = list(val_df['a1'])
val_a2_list = list(val_df['a2'])
val_a3_list = list(val_df['a3'])
val_a4_list = list(val_df['a4'])
val_q_list = list(val_df['q'])
print(val_bbox_list[0], val_vid_names_list[0], val_qid_list[0])
length=len(val_qid_list)


# %%

# Create a set to store all labels in the "label" columns
label_set = set()

# For each bounding box in each frame of each "bbox" element, extract the "label" columns
# and transform it into a dictionary, where the key is the "qid" and the value is a list of "label"
# In the same time store labels in the "label" columns into the label sets
val_label_dict = {}
for i in range(len(val_bbox_list)): # For each question
    val_label_dict[val_qid_list[i]] = []
    for j in val_bbox_list[i].values(): # For each frame
        for k in j: # For each bounding box
            # If there is no duplicate label, add it to the dictionary
            if k['label'] not in val_label_dict[val_qid_list[i]]:
                val_label_dict[val_qid_list[i]].append(k['label'])
            # Add the label to the label set
            label_set.add(k['label'])

# Go through the labels in train dataset and add them into the label_set
for i in range(len(train_bbox_list)): # For each question
    for j in train_bbox_list[i].values(): # For each frame
        for k in j: # For each bounding box
            # Add the label to the label set
            label_set.add(k['label'])

# Verify it by printing the first element in the dictionary and the size of the dictionary
print('The first element in label dictionary with qid as key:',val_label_dict[val_qid_list[0]])
print('The length of label dictionary: ',len(val_label_dict))

# Verify it by printing the size of label set
print('Size of labels set:',len(label_set))


# %%

# Transform the "video_names_list" into a set with unique values
val_vid_names_set = set(val_vid_names_list)
print('Size of vid_names_set',len(val_vid_names_set))
print('The first elements in the set:', list(val_vid_names_set)[0])

# For each element in the set, get qids of all the equal elements in the list
def get_qid(vid_names_list, vid_names_set,qid_list):
    """
    Get the qids of all the elements in the list that are equal to the element in the set
    """
    # Create an empty dictionary to store the qids
    qids_dict = {}

    # Loop through the set
    for vid_name in vid_names_set:
        # Get the qids of the elements in the list that are equal to the element in the set
        qids = [qid_list[i] for i, x in enumerate(vid_names_list) if x == vid_name]
        # Store the qids in the dictionary
        qids_dict[vid_name] = qids

    return qids_dict

# Get the qids of the elements in the list that are equal to the element in the set
val_qid_dict = get_qid(val_vid_names_list, val_vid_names_set,val_qid_list)
# Check if the qids are correct
print('Size of qids_dict',len(val_qid_dict))
print("Verify: the first key in the dictionary",list(val_qid_dict)[0]) 
print('The first elements in the dictionary:', val_qid_dict[list(val_qid_dict)[0]])



# %%

# A function to calculate the similarity between two strings
def string_similarity(str1, str2):
    matcher = difflib.SequenceMatcher(None, str1, str2)
    return matcher.ratio()

# A function to clean the data in one label set
def clean_data(label_set, exists_label_set):
    """
    Clean the data by removing the labels that are similar to each other
    """
    # Loop through the label set
    for label in label_set:
        # Loop through the label set
        for label_ in label_set:
            # If the similarity between two labels is greater than 0.8 and the labels are not the same
            if string_similarity(label.lower(),label_.lower()) > 0.75 and label != label_:
                # Remove the label from the copy of the label set
                if label_ in exists_label_set:
                    exists_label_set.remove(label_)
                    exists_label_set.add(label)
                return label_, True
    return None, False

# A function to clean the data in two label sets
def clean_data_two(label_set1, label_set2):
    """
    Clean the data by removing the labels that are similar to each other
    """
    # Loop through the label set
    for label in label_set1:
        # Loop through the label set
        for label_ in label_set2:
            # If the similarity between two labels is greater than 0.8 and the labels are not the same
            if string_similarity(label.lower(),label_.lower()) > 0.75 and label != label_:
                # Remove the label from the copy of the label set
                return label, label_, True
    return None, None, False

# A function to asynchronously call the OpenAI API to process the prompts
async def openai_raw(client, prompt, sem, exception_dict):
    """
    Call the OpenAI API to process the labels
    """
    # Call the OpenAI API to process the prompts
    async with sem:
        try:
            response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream = False
            )
            # Read the positive_response and negative_response as json
            object=response.choices[0].message.content
            return object
        
        except Exception as e:
            print(e)
        
async def openai_questions(client,questions_dict, vid_name, action, exists_object, nonexists_object, sem, i):
    question_prompt="""Action: {}
Object: {}
1. Remove (object) in the action
2. Give a question in format: Is (object) visible when (action)
3. rewrite (2) by replacing visible and make it grammatically correct
Output (3) only"""
    positive_answer_prompt="""Action: {}
Object: {}
1. Remove (object) in the action
2. Give a sentence in format: (object) is visible when (action)
3. rewrite (2) by replacing visible and make it grammatically correct
Output (3) only"""
    negative_answer_prompt="""Action: {}
Object: {}
1. Remove (object) in the action
2. Give a sentence in format: (object) is not visible when (action)
3. rewrite (2) by replacing visible and make it grammatically correct
Output (3) only"""
    async with sem:
        try:
            positive_question_response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": question_prompt.format(action, exists_object)
                    }
                ],
                stream = False
            )
            positive_positive_ans_response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": positive_answer_prompt.format(action, exists_object)
                    }
                ],
                stream = False
            )
            negative_positive_ans_response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": negative_answer_prompt.format(action, exists_object)
                    }
                ],
                stream = False
            )
            negative_question_response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": question_prompt.format(action, nonexists_object)
                    }
                ],
                stream = False
            )
            positive_negative_ans_response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": positive_answer_prompt.format(action, nonexists_object)
                    }
                ],
                stream = False
            )
            negative_negative_ans_response = await client.chat.completions.create(
                model = "deepseek-chat",
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert of aritificial intelligence."
                    },
                    {
                        "role": "user",
                        "content": negative_answer_prompt.format(action, nonexists_object)
                    }
                ],
                stream = False
            )
            # Read the positive_response and negative_response as json
            positive_content={'q': positive_question_response.choices[0].message.content, 'd1':positive_positive_ans_response.choices[0].message.content, 'd2':negative_positive_ans_response.choices[0].message.content}
            questions_dict["positive"][str(i)]=positive_content
            negative_content={'q': negative_question_response.choices[0].message.content, 'd1':positive_negative_ans_response.choices[0].message.content, 'd2':negative_negative_ans_response.choices[0].message.content}
            questions_dict["negative"][str(i)]=negative_content
        except Exception as e:
            print(e)

# For vid_name in the set, get the qids. Then for each qid, get the characters and objects in label and transfer them to visible_charac_set and visible_object_set. Then call the OpenAI API to process the labels
async def generate_questions(qid_dict, label_dict, questions_dict, client, sem, exception_dict, i):
    prompt= """Question: {}
Answer: {}
1. Produce a sentence of \"someone is doing something\" within 10 words to describe above question and answer.
Output (1) only"""
    object_prompt="""Set: {}
Is there a non-name term that is a specific object in the set. 
If no, output "No" only. 
If yes, output the term only."""
    question_id=val_qid_list[i]
    exists_label_set=label_dict[question_id]

    loop = asyncio.get_event_loop()
    vid_name=val_vid_names_list[i]
        # Get the qids of the vid_name
    qids = qid_dict[vid_name]
    # Create a set to store existent labels
    visible_label_set = set()
    # Loop through the qids
    for qid in qids:
        # Get the labels of the qid
        labels = label_dict[qid]
        # For each label in the labels
        for label in labels:
            # Add the existent label to the visible_label_set
            visible_label_set.add(label)
    exists_label_set=set(exists_label_set)
    nonexists_label_set=visible_label_set-exists_label_set

    # Some label in visible_label_set might contains spelling mistake, so we need to correct them. The methods is to compare the labels in visible_label_set with each other
        # If the similarity between two labels is greater than 0.8, we consider them as the same label
    while True:
        label_, flag = clean_data(nonexists_label_set, exists_label_set)
        if flag:
            nonexists_label_set.remove(label_)
        else:
            break
    question= val_q_list[i]
    ans_id = val_answer_idx_list[i]
    if ans_id=="0":                
        ans=val_a0_list[i]
    elif ans_id=="1":
        ans=val_a1_list[i]
    elif ans_id=="2":
        ans=val_a2_list[i]
    elif ans_id=="3":
        ans=val_a3_list[i]
    elif ans_id=="4":
        ans=val_a4_list[i]
    name=0
    remaining=0

    action= await openai_raw(client, prompt.format(question, ans), sem, exception_dict)

    exists_object=await openai_raw(client, object_prompt.format(exists_label_set), sem, exception_dict)
    nonexists_object=await openai_raw(client, object_prompt.format(nonexists_label_set), sem, exception_dict)
    if exists_object=="No" or exists_object=="no" or nonexists_object=="No" or nonexists_object=="no":
        return 0;
    await openai_questions(client,questions_dict, vid_name, action, exists_object, nonexists_object, sem, i)

# A function to generate the dataset
async def generate_dataset(vid_names_set, qid_dict, label_dict, label_set, len):
    # Initialize deepseek client
    client = AsyncOpenAI(api_key='sk-aad5f73ac586472cbde271ca211a7f76', base_url="https://api.deepseek.com")
    sem = asyncio.Semaphore(100)
    
    # Create a dictionary to store the question
    questions_dict = {}
    questions_dict["positive"]={}
    questions_dict["negative"]={}
    exception_dict = {}

    print('Start generating questions')
    # Create a list of tasks to generate questions for each vid_name
    tasks = [generate_questions(qid_dict, label_dict, questions_dict, client, sem, exception_dict, i) for i in range(len)]
    await asyncio.gather(*tasks)

    questions_dict["positive"] = OrderedDict(sorted(questions_dict["positive"].items(), key=lambda x: int(x[0])))
    questions_dict["negative"] = OrderedDict(sorted(questions_dict["negative"].items(), key=lambda x: int(x[0])))

    # Save the questions_dict into a json file
    json_data = json.dumps(questions_dict)
    with open('object_questions_dict.json', 'w') as f:
        f.write(json_data)

    # Save the exception_dict into a json file
    with open('exception_dict.json', 'w') as f:
        json.dump(exception_dict, f)
    
# Run the generate_dataset function
async def main(vid_names_set, qid_dict, label_dict, label_set, len):
    # Record the time
    start = time.time()
    # Generate the dataset
    await generate_dataset(vid_names_set, qid_dict, label_dict, label_set, len)
    # Print the time
    print('Time taken: ', time.time()-start)

# Run the main function
asyncio.run(main(val_vid_names_set, val_qid_dict, val_label_dict, label_set, length))
