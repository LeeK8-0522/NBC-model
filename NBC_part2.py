import numpy as np
import NBC_part1

def load_model(filename):
    data = np.load(filename, allow_pickle=True)
    prior = data['prior'].item()
    likelihood = {'P': data['likelihood_P'].item(), 'N': data['likelihood_N'].item()}
    features = data['features'].tolist()
    return prior, likelihood, features

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

    # Compare probabilities and return the class with the higher probability
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

    return correct_cnt/total_cnt



prior, likelihood, features = load_model('model.npz')

chunks = NBC_part1.get_chunk('test.csv')
tokens = NBC_part1.tokenize(chunks)
tokens = NBC_part1.featuring(tokens, features)
accuracy = calculate_accuracy(tokens, prior, likelihood) * 100

print(f"Total accuracy : {accuracy}")



