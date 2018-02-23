from main import find_answers
from custom_exceptions import PaperDetectionError, BubbleDetectionError

def compute_accuracy(original_ans, detected_ans):
    correct = 0.0
    for i in range(len(original_ans)):
        if original_ans[i] == detected_ans[i]:
            correct += 1
        else:
            print(i + 1, original_ans[i], detected_ans[i])
    return correct / len(original_ans)

filenames = ["assets/IMG_" + num + ".JPG" for num in ['6388', '2826','1243', '3064', '6591', '8836', '8760', '6024', '6112', '1738',
                                                        '3031', '5241', '7230', '7394', '3497', '6415']]

fails = ['9891', '8880']
multiple_bubbles = ["1299"]

ORIGINAL_ANS = ['A', 'C', 'B', 'D', 'B', 'A', 'D', 'B', 'C', 'C', 'A', 'A', 'A',
                'A', 'D', 'C', 'A', 'D', 'D', 'B', 'B', 'A', 'C', 'C', 'C', 'B',
                'B', 'D', 'A', 'B', 'A', 'C', 'D', 'B', 'D', 'D', 'C', 'B', 'A',
                'B', 'B', 'D', 'A', 'D', 'B', 'C', 'A', 'A', 'D', 'D', 'A', 'A']

cumulative_accuracy = 0

for file in filenames:
    print(file)
    try:
        detected_ans = find_answers(file)
    except (PaperDetectionError, BubbleDetectionError) as e:
        print(e)
        continue
    else:
        accuracy = compute_accuracy(ORIGINAL_ANS, detected_ans)
        cumulative_accuracy += accuracy
        print("Accuracy: " + str(accuracy))

print("Cumulative Accuracy: " + str(cumulative_accuracy / len(filenames)))