from main import load_json
import json
import os


if __name__ == "__main__":
    train_path = "./Data/tmp/new_data_train_5986.json"
    # train_path = os.path.abspath(train_path)
    test_path = "./Data/tmp/new_data_test_1600.json"
    # test_path = os.path.abspath(test_path)

    train_data = load_json(train_path)
    test_data = load_json(test_path)
    test_data = json.dumps(test_data)
    print(len(test_data))

    count = 0
    for train_elm in train_data:
        content = train_elm["content"][:100]
        if content in test_data:
            count += 1
            print(count)
            print(train_elm)
        # break
