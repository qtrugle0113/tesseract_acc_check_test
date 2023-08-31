def jaccard_similarity(path1, path2):
    with open(path1, 'r', encoding='utf-8') as file1:
        text1 = file1.read()
    with open(path2, 'r', encoding='utf-8') as file2:
        text2 = file2.read()

    intersection_cardinality = len(set.intersection(*[set(text1), set(text2)]))
    union_cardinality = len(set.union(*[set(text1), set(text2)]))
    similar = intersection_cardinality / float(union_cardinality)

    return similar
