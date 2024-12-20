import json


def read_descriptions(filename):
    dictionary = {}
    with open(filename, encoding='utf-8') as file:
        for line in file:
            if line := line.strip():
                key, english, chinese = line.split(',')
                dictionary[key] = f'{english}, {chinese}'
    return dictionary


def read_json_file(filename):
    with open(filename, encoding='utf-8') as file:
        json_data = json.load(file)
    return json_data


def replace_keys(json_data, key_map):
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, str) and value in key_map:
                json_data[key] = f'{value}, {key_map[value]}'
            replace_keys(value, key_map)
    elif isinstance(json_data, list):
        for i in range(len(json_data)):
            replace_keys(json_data[i], key_map)
    else:
        return


def main():
    descriptions = read_descriptions('D:/User/Desktop/class-descriptions-boxable.csv')
    json_file = read_json_file('D:/User/Desktop/bbox_labels_600_hierarchy.json')
    replace_keys(json_file, descriptions)
    with open('D:/User/Desktop/result.json', 'w', encoding='utf-8') as f:
        json.dump(json_file, f, indent=4, ensure_ascii=False)
    print('\nAll process success\n')


if __name__ == '__main__':
    main()
