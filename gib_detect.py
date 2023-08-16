import pickle
import gib_detect_train

model_data = pickle.load(open('gib_model.pkl', 'rb'))

while True:
    l = input("Enter a word, type exit to exit: \n")
    if l.lower() == "exit":
        break
    else: 
        model_mat = model_data['mat']
        threshold = model_data['thresh']
        print(gib_detect_train.avg_transition_prob(l, model_mat) > threshold)
