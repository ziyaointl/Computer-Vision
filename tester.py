from main import find_answers

# filenames = ["assets/IMG_" + num + ".JPG" for num in ['0232','5361', '4783','4203', '5485', '5785', '8083', '9001']]

filenames = ["assets/IMG_" + num + ".JPG" for num in ['2588', '0739', '1071', '1522', '1982', '2434',
                                                      '2588', '2826', '3061', '3819', '4542', '6236', '6261', '6388']]

original_ans = ['A', 'C', 'B', 'D', 'B', 'A', 'D', 'B', 'C', 'C', 'A', 'A', 'A',
                'A', 'D', 'C', 'A', 'D', 'D', 'B', 'B', 'A', 'C', 'C', 'C', 'B',
                'B', 'D', 'A', 'B', 'A', 'C', 'D', 'B', 'D', 'D', 'C', 'B', 'A',
                'B', 'B', 'D', 'A', 'D', 'B', 'C', 'A', 'A', 'D', 'D', 'A', 'A']

for file in filenames:
    print(file)
    detected_ans = find_answers(file)