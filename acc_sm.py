import difflib


def sequence_matcher(path1, path2):
    with open(path1, 'r', encoding='utf-8') as file1:
        text1 = file1.read()
    with open(path2, 'r', encoding='utf-8') as file2:
        text2 = file2.read()

    answer_bytes = bytes(text1, 'utf-8')
    input_bytes = bytes(text2, 'utf-8')
    answer_bytes_list = list(answer_bytes)
    input_bytes_list = list(input_bytes)

    sm = difflib.SequenceMatcher(None, answer_bytes_list, input_bytes_list)
    similar = sm.ratio()

    return similar
