import numpy as np

# 시그모이드 활성화 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 시그모이드의 미분
def sigmoid_derivative(x):
    return x * (1-x)

# 가중치 초기화 함수
def initialize_weights(raw_size, column_size):
    weights = [[np.random.uniform(0, 1) for _ in range(column_size)] for _ in range(raw_size)]
    return weights

# winner를 찾아 반환시키는 함수
def findeMax(input):
    max_value = max(input)
    max_indices = []
    for i, value in enumerate(input):
        if value == max_value:
            max_indices.append(i)
    
    winner = [0] * len(input)
    for max_index in max_indices:
        winner[max_index] = 1
    
    return winner



# 순전파 계산
def forward_propagation(inputs, weight_ih, weight_ho):

    # 입력층에서 은닉층까지
    hidden = []
    for j in range(len(weight_ih[0])):
        net_hidden = sum([inputs[i] * weight_ih[i][j] for i in range(len(inputs))])
        hidden.append(sigmoid(net_hidden))

    # 은닉층에서 출력층까지
    output = []
    for j in range(len(weight_ho[0])):
        net_out = sum([hidden[i] * weight_ho[i][j] for i in range(len(hidden))])
        output.append(sigmoid(net_out))
    
    return hidden, output


# 역전파 계산
def back_propagation(inputs, target, hidden, output, weight_ih, weight_ho, learning_rate):
    transposed_ho = np.array(weight_ho).T

    # 출력층에서 은닉층으로 역전
    output_error = [target[i] - output[i] for i in range(len(target))]
    output_delta = [output_error[i] * sigmoid_derivative(output[i]) for i in range(len(output))]

    # 은닉층에서 에러 계산
    hidden_error = []
    for i in range(len(hidden)):
        error_hi = sum([output_delta[j] * weight_ho[i][j] for j in range(len(output_delta))])
        hidden_error.append(error_hi)
    hidden_delta = [hidden_error[i] * sigmoid_derivative(hidden[i]) for i in range(len(hidden))]

    # 가중치 업데이트
    for i in range(len(weight_ho)):
        for j in range(len(weight_ho[0])):
            weight_ho[i][j] += hidden[i] * output_delta[j] * learning_rate
    for i in range(len(weight_ih)):
        for j in range(len(weight_ih[0])):
            weight_ih[i][j] += inputs[i] * hidden_delta[j] * learning_rate

    return weight_ih, weight_ho


# 정확도를 구하는 함수
def calculate_accuracy(predictions, targets):
    correct_predictions = []

    # 예측값과 목표값 비교
    for i in range(len(predictions)):
        if predictions[i].index(max(predictions[i])) == targets[i].index(max(targets[i])):
            correct_predictions.append(1)
        else:
            correct_predictions.append(0)

    # 정확도 계산
    accuracy = np.mean(correct_predictions)

    return accuracy



# 신경망 학습
def train_neural_network(inputs, targets, hidden_size, output_size, epochs, learning_rate):

    # 초기화
    input_size = len(inputs[0])
    weights_input_hidden = initialize_weights(input_size, hidden_size)
    weights_hidden_output = initialize_weights(hidden_size, output_size)

    # 반복 학습
    for epoch in range(epochs):
        total_error = 0
        predictions = []

        for input_data, target in zip(inputs, targets):
            hidden, output = forward_propagation(input_data, weights_input_hidden, weights_hidden_output)
            weights_input_hidden, weights_hidden_output = back_propagation(input_data, target, hidden, output, weights_input_hidden, weights_hidden_output, learning_rate)

            total_error += sum([abs(target[j] - output[j]) for j in range(len(output))])
            predictions.append(output)

        accuracy = calculate_accuracy(predictions, targets)

        # 학습 결과 출력
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1} \tError: { round( total_error/len(inputs), 6) } \tAccuracy: {round(accuracy, 3)}")

    return weights_input_hidden, weights_hidden_output



# 테스트 평가 수행
def test_neural_network(inputs, targets, weights_input_hidden, weights_hidden_output):
    success = 0
    fail = 0

    for i in range(len(inputs)):
        input_data = inputs[i]
        target = targets[i]

        _, output = forward_propagation(input_data, weights_input_hidden, weights_hidden_output)
        rounded_output = [round(value, 6) for value in output]
        result = findeMax(output)

        if sum([abs(target[j] - result[j]) for j in range(len(target))]) == 0:
            success += 1
        else: 
            fail += 1
        print(f"Target: {target} \tOutput: {rounded_output} \tsuccess: {success} \tfail : {fail}")



# 입력 데이터
training_value = []
testing_value = []

training_label = [[1, 0, 0]] * 25 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25
testing_label = [[1, 0, 0]] * 25 + [[0, 1, 0]] * 25 + [[0, 0, 1]] * 25

with open("./training.dat", 'r') as file:
    for i in file:
        training_value += [i.split()]


with open("./testing.dat", 'r') as file:
    for i in file:
        testing_value += [i.split()]

train_inputs = [[float(element) for element in inner_list] for inner_list in training_value]
test_inputs = [[float(element) for element in inner_list] for inner_list in testing_value]


# 학습
learning_rate = 0.1
epochs = 10000

print("\n","\t[ 신경망 학습을 통한 가중치 업데이트 ]")
trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network(train_inputs, training_label, 3, 3, epochs, learning_rate)


print("\n","\t\t\t\t--[ 테스트 결과 출력 내용 ]--")
test_neural_network(test_inputs, testing_label, trained_weights_input_hidden, trained_weights_hidden_output)
