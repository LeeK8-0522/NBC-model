import numpy as np

def get_chunk(inputfile):
    def is_start(line):
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
            line = line.strip()
            if flag is not True:
                flag = True
                if(is_start(line)):
                    cur_text += line
                else:
                    print("Wrong Action!")
            elif(is_start(line)):
                chunks.append(cur_text)
                cur_text = line
            else:
                cur_text += line

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


def tokenize(chunks):
    result = []

    for item in chunks:
        comma_index = item.find(',')  # 첫 번째로 등장하는 ',' 찾기
        if comma_index != -1:
            stars = item[:comma_index]
            if(stars == '5'):
                stars = 'P'
            else:
                stars = 'N'

            text = item[comma_index + 1:].strip()
            words = filtering(text)
            result.append((stars, words))

    return result


def get_featrues(tokens):
    freq = {}  # dictionary 자료구조를 이용하여 빈도수 세기

    for _, words in tokens:
        for word in words:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1

    sorted_words = sorted(freq.items(), key=lambda item: item[1], reverse=True)  # 빈도수를 기준으로 정렬하기
    features = [word for word, _ in sorted_words[:1000]]

    return features


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

    # Calculate likelihood probabilities with Laplace smoothing
    likelihood = {'P': {}, 'N': {}}

    for word in features:
        likelihood['P'][word] = (P_word_cnt[word] + 1) / (P_total_cnt + 1000)
        likelihood['N'][word] = (N_word_cnt[word] + 1) / (N_total_cnt + 1000)

    return likelihood

def save_model(prior, likelihood, features, filename):
    np.savez(filename, prior=prior, likelihood_P=likelihood['P'], likelihood_N=likelihood['N'], features=features)


chunks = get_chunk('train.csv')
tokens = tokenize(chunks)
features = get_featrues(tokens)
tokens = featuring(tokens, features)
prior = calculate_prior(tokens)
likelihood = calculate_likelihood(tokens, features)

save_model(prior, likelihood, features, 'model.npz')






