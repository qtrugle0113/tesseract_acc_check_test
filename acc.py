def read_text_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    return content


def text_compare(result_text, ground_truth_text):
    result_words = result_text.split()
    ground_truth_words = ground_truth_text.split()

    correct_words = sum(1 for word1, word2 in zip(result_words, ground_truth_words) if word1 == word2)
    total_words = len(ground_truth_words)

    accuracy = (correct_words / total_words) * 100
    return accuracy


def calculate_accuracy(result, ground_truth):
    result_text = read_text_file(result)
    ground_truth_text = read_text_file(ground_truth)

    accuracy = text_compare(result_text, ground_truth_text)
    return accuracy

