# !python -m spacy download en_core_web_sm

import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
import io
import numpy as np
import spacy # imported because of generate_description
from keras.callbacks import ModelCheckpoint # used because keras is used later in the code
import main_definitions as md

# Connect to the database
db = md.connect_to_database(database="code_examples", user="root", password="", host="localhost")
cursor = db.cursor()

# Retrieve code examples from the database
correct_examples, incorrect_examples = md.get_code_examples(cursor)

# Implement mutation or crossover techniques to modify the code examples
modified_correct_examples, modified_incorrect_examples = md.modify_code_examples(correct_examples, md.crossover_probability, incorrect_examples)

# Retrieve test cases from the database
test_cases = md.get_test_cases(cursor)

# Define a set of tests or use existing test suites to evaluate the generated code
generated_code = "print('Hello, Ukraine!!')"
score = md.evaluate_generated_code(generated_code, test_cases)
if score < 0:
    score = 0

# Assign a numeric value as a reward to the program based on the generated code's functionality
generated_code = md.generate_code()
test_cases = md.get_test_cases()
reward = md.assign_reward(generated_code, test_cases)

# Prepare data for LSTM model
data, labels, token_to_num, num_to_token = md.prepare_data_for_lstm(database="code_examples", user="root", password="")

# Explore the hyperparameter space using a Bayesian optimization algorithm to find the optimal set of hyperparameters that minimize the error on a validation set
validation_data, val_labels = data[:len(data)//10], labels[:len(labels)//10]
reward = md.optimize_neural_network(data, labels, validation_data, val_labels, len(token_to_num), len(set(labels)), reward)

# Write the reward to a file
md.write_reward_to_file(reward, "e:\\__results\\models\\coderman\\reward.txt")

# Define function to tokenize code
def tokenize_code(code):
    tokens = []
    try:
        g = tokenize.generate_tokens(io.StringIO(code).readline)
        for toknum, tokval, _, _, _ in g:
            tokens.append(tokval)
    except tokenize.TokenError:
        pass
    return tokens

# Define function to prepare data for LSTM model
def prepare_data_for_lstm(database, user, password):
    # Retrieve code examples from the database
    db = connect_to_database(database, user, password)
    cursor = db.cursor()
    correct_examples, incorrect_examples = get_code_examples(cursor)

    # Modify code examples
    modified_correct_examples, modified_incorrect_examples = modify_code_examples(correct_examples, incorrect_examples)

    # Tokenize code examples
    tokenized_correct_examples = []
    tokenized_incorrect_examples = []
    for example in modified_correct_examples:
        tokenized_correct_examples.append(tokenize_code(example))
    for example in modified_incorrect_examples:
        tokenized_incorrect_examples.append(tokenize_code(example))

    # Use pairs of input-output sequences, where the input sequence is a partial code snippet and the output sequence is the next token in the code       
    data = []
    labels = []
    for example in tokenized_correct_examples:
        for i in range(len(example)-1):
            data.append(example[:i+1])
            labels.append(example[i+1])
    for example in tokenized_incorrect_examples:
        for i in range(len(example)-1):
            data.append(example[:i+1])
            labels.append(example[i+1])

    # Convert data and labels to numerical format
    token_to_num = dict(zip(set(labels), range(len(set(labels)))))
    num_to_token = dict(zip(range(len(set(labels))), set(labels)))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = token_to_num[data[i][j]]
        labels[i] = token_to_num[labels[i]]

    # Split data into training and validation sets
    data = np.array(data)
    labels = keras.utils.to_categorical(labels, num_classes=len(token_to_num))
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    validation_split = 0.1
    num_validation_samples = int(validation_split * data.shape[0])
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return x_train, y_train, x_val, y_val, token_to_num, num_to_token

# Call prepare_data_for_lstm() with database connection arguments in the main file
x_train, y_train, x_val, y_val, token_to_num, num_to_token = prepare_data_for_lstm(database="code_examples", user="root", password="")

# Use backpropagation with the help of a suitable optimizer and loss function to train the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(len(max(x_train, key=len)), len(token_to_num))))
model.add(Dense(len(token_to_num), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the LSTM model
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_val, y_val))

# Use the trained LSTM model to generate new code by sampling from the output sequence based on predicted probabilities
max_seq_len = 50 # set maximum length of generated code
generated_code = md.generate_start_seq()
start_seq = 'def ' # set starting sequence
for i in range(max_seq_len):
    # Convert start_seq to numerical format
    start_seq_num = [token_to_num[t] for t in start_seq]
    # Pad sequence if it is shorter than max_seq_len
    if len(start_seq_num) < max_seq_len:
        start_seq_num = [0]*(max_seq_len - len(start_seq_num)) + start_seq_num
    # Generate probabilities for the next token in the sequence
    pred_probs = model.predict(np.array([start_seq_num]))
    # Sample the next token from the predicted probabilities
    next_token_num = np.random.choice(range(len(token_to_num)), p=pred_probs[0])
    next_token = num_to_token[next_token_num]
    # Add the next token to the generated code
    generated_code += next_token
    # Update the start sequence with the next token
    start_seq = start_seq[1:] + next_token

# Generate code from a high-level description
def generate_code_from_description(description):
    # Load the spaCy language model
    nlp = spacy.load('en_core_web_sm')

    # Parse the description to extract relevant information
    doc = nlp(description)
    program_type = None
    variable_names = []
    variable_types = []
    operations = []

    for token in doc:
        if token.pos_ == "VERB" and not program_type:
            program_type = token.lemma_
        elif token.pos_ == "NOUN":
            variable_names.append(token.text)
        elif token.pos_ == "ADJ" and variable_names:
            variable_types.append(token.text)
        elif token.pos_ == "ADP" and token.text == "to":
            operations.append("assignment")
        elif token.pos_ == "VERB" and token.lemma_ == "print":
            operations.append("print")
        elif token.pos_ == "CCONJ" and token.text == "and" and variable_names and variable_types:
            variable_types[-1] += " " + token.text

    # Generate the code based on the parsed information
    code = ""

    if program_type == "write":
        # Create a new file and write to it
        if variable_names and operations and operations[0] == "assignment" and len(operations) == 1:
            code += f"with open('{variable_names[0]}', 'w') as f:\n"
            code += f"    f.write({variable_names[1]})\n"
        else:
            raise ValueError("Invalid description for write program")
    elif program_type == "read":
        # Read from a file
        if variable_names and len(variable_names) == 1 and not variable_types and operations and operations[0] == "print" and len(operations) == 1:
            code += f"with open('{variable_names[0]}', 'r') as f:\n"
            code += f"    print(f.read())\n"
        else:
            raise ValueError("Invalid description for read program")
    elif program_type == "calculation":
        # Perform a calculation
        if variable_names and variable_types and operations and len(operations) == 1:
            if variable_types[0] == "integer":
                code += f"{variable_names[0]} = int({variable_names[1]})\n"
            elif variable_types[0] == "float":
                code += f"{variable_names[0]} = float({variable_names[1]})\n"
            else:
                raise ValueError("Invalid description for calculation program")
            if operations[0] == "print":
                code += f"print({variable_names[0]})\n"
            else:
                raise ValueError("Invalid description for calculation program")
        else:
            raise ValueError("Invalid description for calculation program")
    else:
        raise ValueError("Invalid program type")

    return code

# Get user input and generate high-level description
user_input = input("Enter task description: ")
description = generate_description(user_input)
description = "write contents of variable1 to file variable2"
code = generate_code_from_description(description)
print(code)
