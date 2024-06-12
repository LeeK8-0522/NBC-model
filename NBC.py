from matplotlib import pyplot as plt

def get_chunk(inputfile):
    def is_start(line):
        line = line.rstrip('\n')
        if len(line) >= 2:
            if(line[0].isdigit() and line[1] == ','):
                return True
        return False

    with open(inputfile, 'r', encoding='utf-8') as fr:
        chunks = []
        fr.readline()  # header 제거
        cur_text = ""

        flag = False

        for line in fr:
            if flag is not True:
                flag = True
                if(is_start(line)):
                    cur_text = line + " "
                else:
                    print("Wrong Action!")
            elif(is_start(line)):
                chunks.append(cur_text)
                cur_text = line + " "
            else:
                cur_text += line + " "

        chunks.append(cur_text)

        return chunks


def filtering(text):
    special_characters = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    stop_words = []
    with open('stopwords.txt', 'r', encoding='utf-8') as fr:
        for line in fr:
            stop_words.append(line.strip().lower())

    text = text.lower()  # change to lower case
    filtered_text = ''.join(char for char in text if char not in special_characters)  # delete special characters
    words = filtered_text.split()  # split into words
    filtered_words = [word for word in words if word not in stop_words]  # delete stop words

    return filtered_words


def tokenize(chunks, percentage, debug):
    size = int(len(chunks) * percentage / 100)
    # print(f"{size} ({percentage}%) data tokenizing...")
    cnt = 0

    total_words_size = 0

    result = []

    for item in chunks:
        if cnt == size:
            break
        comma_index = item.find(',')  # 첫 번째로 등장하는 ',' 찾기
        if comma_index != -1:
            stars = item[:comma_index]
            if stars == '5':
                stars = 'P'
            else:
                stars = 'N'

            text = item[comma_index + 1:].strip()
            words = filtering(text)
            result.append((stars, words))
            cnt += 1
            total_words_size += len(words)

    # print(f"{cnt} ({percentage}%) data tokenized!")
    if debug:
        print(f"Total number of words in text: {total_words_size}")

    return result, total_words_size


def get_features(tokens, total_cnt):
    freq = {}  # dictionary 자료구조를 이용하여 빈도수 세기

    for _, words in tokens:
        for word in words:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    sorted_words = sorted(freq.items(), key=lambda item: item[1], reverse=True)  # 빈도수를 기준으로 정렬하기
    features = [word for word, _ in sorted_words[:1000]]

    cnt = 0
    for word in features:
        cnt += freq[word]

    selectivity = int(100 * cnt/total_cnt)

    return features, selectivity


def featuring(tokens, features):
    featured_tokens = []

    for label, words in tokens:
        words = [word for word in words if word in features]
        featured_tokens.append((label, words))

    return featured_tokens

def calculate_prior(tokens):
    prior = {}

    total_size = len(tokens)
    P_size = sum(1 for label, _ in tokens if label == 'P')
    N_size = total_size - P_size

    prior['P'] = P_size / total_size
    prior['N'] = N_size / total_size

    return prior


def calculate_likelihood(tokens, features):
    P_word_cnt = {}
    N_word_cnt = {}
    P_total_cnt = 0
    N_total_cnt = 0

    for word in features:  # initialize
        P_word_cnt[word] = 0
        N_word_cnt[word] = 0

    for label, words in tokens:
        if label == 'P':
            P_total_cnt += len(words)
            for word in words:
                P_word_cnt[word] += 1
        else:
            N_total_cnt += len(words)
            for word in words:
                N_word_cnt[word] += 1

    likelihood = {'P': {}, 'N': {}}

    size = len(features)
    for word in features:
        likelihood['P'][word] = (P_word_cnt[word] + 1) / (P_total_cnt + size)
        likelihood['N'][word] = (N_word_cnt[word] + 1) / (N_total_cnt + size)

    return likelihood

def predict(words, prior, likelihood):
    prob_P = prior['P']
    prob_N = prior['N']

    for word in words:
        if word in likelihood['P']:
            prob_P *= likelihood['P'][word]
        else:
            print("Wrong Action!")
        if word in likelihood['N']:
            prob_N *= likelihood['N'][word]
        else:
            print("Wrong Action!")

    if prob_P > prob_N:
        return 'P'
    else:
        return 'N'

def calculate_accuracy(tokens, prior, likelihood):
    correct_cnt = 0
    total_cnt = 0

    for label, words in tokens:
        prediction = predict(words, prior, likelihood)

        if prediction == label:
            correct_cnt += 1
        total_cnt += 1

    return (correct_cnt/total_cnt)*100

percentages = [0.05, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracies = []
selectivities = []

for percentage in percentages:
    chunks = get_chunk('train.csv')
    tokens, total_cnt = tokenize(chunks, percentage, False)
    features, selectivity = get_features(tokens, total_cnt)
    tokens = featuring(tokens, features)
    prior = calculate_prior(tokens)
    likelihood = calculate_likelihood(tokens, features)
    test_chunks = get_chunk('test.csv')
    test_tokens, _ = tokenize(test_chunks, 100, False)
    test_tokens = featuring(test_tokens, features)
    accuracy = calculate_accuracy(test_tokens, prior, likelihood)
    accuracies.append(accuracy)
    selectivities.append(selectivity)

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Training Data Percentage (%)')
ax1.set_ylabel('Accuracy (%)', color=color)
ax1.plot(percentages, accuracies, marker='o', color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Selectivity (%)', color=color)
ax2.plot(percentages, selectivities, marker='x', linestyle='--', color=color, label='Selectivity')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Accuracy and Selectivity vs Training Data Percentage')
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
plt.show()




