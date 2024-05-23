# %%
import numpy as np
import matplotlib.pyplot as plt
import requests
from copy import deepcopy


# %%
def get_output(sentence):

    if sentence == "":
        return "Attention! The input is empty. Please enter a sentence."

    debug = False

    url = "http://127.0.0.1:5000"
    data = {"user_input": sentence}

    response = requests.post(url, data=data)

    # get the p with the class "text-result"
    if len(response.text) == 0:
        return "Attention! No output was found."
    try:
        out = response.text.split('<p class="text-result">')[1].split("</p>")[0]
    except:
        pass
    # create list
    out = out.split("</span></span>")[:-1]
    d = {"tokens": [], "ner_tags": [], "confidence": []}
    for o in out:
        # find where highlight label- is and add the number after it to the dictionary
        d["ner_tags"].append(int(o.split("highlight label-")[1][0]))
        # find the text of the label
        text = o.split('">')[1].split("<span")[0]
        d["tokens"].append(text)
        if len(text) == 1:
            confidence = 100.0
        else:
            confidence = o[::-1][1:6][::-1]
            if confidence[0] == ">":
                confidence = confidence[1:]
            confidence = float(confidence)
        d["confidence"].append(confidence)

    if debug:
        print(1, d["tokens"])

    # map d, 0->'B-O', 1->'B-AC', 2->'B-LF', 3->'I-LF'
    d["ner_tags"] = [
        (
            "B-O"
            if x == 0
            else (
                "B-AC"
                if x == 1
                else "B-LF" if x == 2 else "I-LF" if x == 3 else "ERROR"
            )
        )
        for x in d["ner_tags"]
    ]

    return d


# %%
from colorama import Fore, Back, Style


def print_in_color(sentence):
    for i, word in enumerate(sentence["tokens"]):
        if sentence["ner_tags"][i] == "B-O":
            print(Fore.WHITE + word, end=" ")
        elif sentence["ner_tags"][i] == "B-AC":
            print(Fore.RED + word, end=" ")
        elif sentence["ner_tags"][i] == "B-LF":
            print(Fore.BLUE + word, end=" ")
        elif sentence["ner_tags"][i] == "I-LF":
            print(Fore.GREEN + word, end=" ")
    print(Style.RESET_ALL)


# %%
def tests():
    # Test 1
    print("-----> Test 1: special characters")
    # Put some special characters in the input

    sentence = "I am a sentence with a special character LMAO: ©. i can have others: ®, ™, 2, ½, "
    d = get_output(sentence)
    print_in_color(d)

    # Test 2
    print("\n-----> Test 2: many inputs")
    # Many many inputs
    sentence = "Hello, my name is John. I am a student of UCLA, which means University of California, Los Angeles"
    sentences = [sentence for i in range(80)]
    for i, sentence in enumerate(sentences):
        d = get_output(sentence)
        if i % 40 == 0:
            print("Step", i)
    print_in_color(d)

    # Test 3
    print("\n-----> Test 3: HTML tags")
    # Add some HTML tags
    sentence = "Hello, my name <span><b>BOLD</b></span> is John. <br> I am a student of UCLA <br> which means University of California, Los Angeles"
    d = get_output(sentence)
    print_in_color(d)

    # Test 4
    print("\n-----> Test 4: Javascript code")
    # Create an alert message in javascript
    sentence = "<script>alert('Hello');</script> I am a student of UCLA, which means University of California, Los Angeles. "
    d = get_output(sentence)
    print_in_color(d)

    # Test 5
    print("\n-----> Test 5: Very long sentence")
    # Create a very long sentence
    sentence = "I am VERY LONG, " * 10000
    d = get_output(sentence)
    print_in_color(d)
    print("Output size:", len(d["tokens"]))

    # Test 6
    print("\n-----> Test 6: Empty sentence")
    # Create an empty sentence
    sentence = ""
    d = get_output(sentence)
    print(d)

    # Test 7
    print("\n-----> Test 7: Sentence with a lot of spaces")
    # Create a sentence with a lot of spaces
    sentence = "I am a sentence with a lot of spaces:      . I can continue here."
    d = get_output(sentence)
    print_in_color(d)


tests()
