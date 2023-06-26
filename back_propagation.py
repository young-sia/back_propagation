import numpy as np
from scipy.stats import norm
from pretreatment import *


def nonlinear(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def main():
    eta = 0.2
    train, train_result = pretreatment_train()
    print(f'train data: {train}')
    print(f'train answer data:{train_result}')
    hidden1_w = norm.ppf(np.random.random((4, 3)), 0, 1)
    hidden2_w = norm.ppf(np.random.random((4, 3)), 0, 1)
    hidden3_w = norm.ppf(np.random.random((4, 3)), 0, 1)

    hidden1_b = norm.ppf(np.random.random(), 0, 1)
    hidden2_b = norm.ppf(np.random.random(), 0, 1)
    hidden3_b = norm.ppf(np.random.random(), 0, 1)

    output1_w = norm.ppf(np.random.random(3), 0, 1)
    output2_w = norm.ppf(np.random.random(3), 0, 1)

    output1_b = norm.ppf(np.random.random(), 0, 1)
    output2_b = norm.ppf(np.random.random(), 0, 1)

    c_record = []
    trial = 0
    test_results = []

    for j in range(50):
        hidden1_gradiant_ct_w = 0
        hidden2_gradiant_ct_w = 0
        hidden3_gradiant_ct_w = 0

        hidden1_gradiant_ct_b = 0
        hidden2_gradiant_ct_b = 0
        hidden3_gradiant_ct_b = 0

        ct = 0

        count = 0
        trial += 1

        output1_gradiant_ct_w = np.array([0.0, 0.0, 0.0])
        output2_gradiant_ct_w = np.array([0.0, 0.0, 0.0])

        output1_gradiant_ct_b = 0
        output2_gradiant_ct_b = 0

        if j == 49:
            test = pretreatment_test()
            print(f'test data: {test}')
            test_number = 0

            for data in test:
                l0 = data
                a2_1 = nonlinear(sum(sum(l0 * hidden1_w)) + hidden1_b)
                a2_2 = nonlinear(sum(sum(l0 * hidden2_w)) + hidden2_b)
                a2_3 = nonlinear(sum(sum(l0 * hidden3_w)) + hidden3_b)
                # print(f'{count}:{a2_1, a2_2, a2_3}')

                z3_1 = sum([a2_1, a2_2, a2_3] * output1_w) + output1_b
                z3_2 = sum([a2_1, a2_2, a2_3] * output2_w) + output2_b
                # print(f'output_w: {output1_w, output2_w}')
                # print(f'output_b: {output1_b, output2_b}')
                # print(f'{count}: {z3_1, z3_2}')

                a3_1 = nonlinear(z3_1)
                a3_2 = nonlinear(z3_2)
                # print(f'{count}:{a3_1, a3_2}')

                result = 0 if a3_1 > a3_2 else 1

                test_number += 1
                print(f'{test_number}번째 테스트 결과: {result}')

                test_results.append(result)

        for data in train:
            l0 = data
            a2_1 = nonlinear(sum(sum(l0 * hidden1_w)) + hidden1_b)
            a2_2 = nonlinear(sum(sum(l0 * hidden2_w)) + hidden2_b)
            a2_3 = nonlinear(sum(sum(l0 * hidden3_w)) + hidden3_b)

            az2_1 = nonlinear(a2_1, True)
            az2_2 = nonlinear(a2_2, True)
            az2_3 = nonlinear(a2_3, True)
            # print(f'{count}az2: {az2_1, az2_2, az2_3}')

            z3_1 = sum(output1_w * np.array([a2_1, a2_2, a2_3])) + output1_b
            z3_2 = sum(output2_w * np.array([a2_1, a2_2, a2_3])) + output2_b
            # print(f'{count} z3:{z3_1, z3_2}')

            a3_1 = round(nonlinear(z3_1), 3)
            a3_2 = round(nonlinear(z3_2), 3)
            # print(f'{count} a3_1 a3_1: {a3_1, a3_2}')

            az3_1 = round(nonlinear(a3_1, True), 3)
            az3_2 = round(nonlinear(a3_2, True), 3)
            # print(f'{count} az3_1: {az3_1, az3_2}')

            c = ((train_result[count][0] - a3_1) ** 2 + (train_result[count][1] - a3_2) ** 2) / 2

            # print(f'{count} c: {c}')

            delta_c_a3_1 = a3_1 - train_result[count][0]
            delta_c_a3_2 = a3_2 - train_result[count][1]
            # print(f'{count} delta_c_a3: {delta_c_a3_1, delta_c_a3_2}')

            delta3_1 = round(delta_c_a3_1 * az3_1, 3)
            delta3_2 = round(delta_c_a3_2 * az3_2, 3)
            # print(f'{count} delta3: {delta3_1, delta3_2}')

            sigma_w_delta3 = output1_w * delta3_1 + output2_w * delta3_2
            # print(f'{count} sigma_w_delta3: {sigma_w_delta3}')
            delta2 = sigma_w_delta3 * [az2_1, az2_2, az2_3]
            # print(f'{count} delta2: {delta2}')

            hidden1_gradiant_c_w = np.multiply(l0, delta2[0])
            hidden2_gradiant_c_w = np.multiply(l0, delta2[1])
            hidden3_gradiant_c_w = np.multiply(l0, delta2[2])
            # print(f'{count} hidden_gradiant_c_w: {hidden1_gradiant_c_w, hidden2_gradiant_c_w, hidden3_gradiant_c_w}')

            output1_gradiant_c_w = [round(a2_1 * delta3_1, 3), round(a2_2 * delta3_1, 3), round(a2_3 * delta3_1, 3)]
            output2_gradiant_c_w = [round(a2_1 * delta3_2, 3), round(a2_2 * delta3_2, 3), round(a2_3 * delta3_2, 3)]
            # print(f'{count} output_gradiant_c_w: {output1_gradiant_c_w, output2_gradiant_c_w}')

            hidden1_gradiant_ct_w += hidden1_gradiant_c_w
            hidden2_gradiant_ct_w += hidden2_gradiant_c_w
            hidden3_gradiant_ct_w += hidden3_gradiant_c_w

            hidden1_gradiant_ct_b += delta2[0]
            hidden2_gradiant_ct_b += delta2[1]
            hidden3_gradiant_ct_b += delta2[2]
            # print(f'{count} delta2: {delta2}')
            # print(f'{count}후 hidden_graiant_ct_b: {hidden1_gradiant_ct_b, hidden2_gradiant_ct_b, hidden3_gradiant_ct_b}')

            output1_gradiant_ct_w += np.array(output1_gradiant_c_w)
            output2_gradiant_ct_w += np.array(output2_gradiant_c_w)
            output1_gradiant_ct_b += delta3_1
            output2_gradiant_ct_b += delta3_2

            ct += c

            count += 1
            # print(f'{trial}번째 시도 {count} train data c 값:{c}')
        # print(f'{trial}회차 hidden_gradiant_ct_w: {hidden1_gradiant_ct_w, hidden2_gradiant_ct_w, hidden3_gradiant_ct_w}')
        # print(f'{trial}회차 hidden_gradiant_ct_b:{hidden1_gradiant_ct_b, hidden2_gradiant_ct_b, hidden3_gradiant_ct_b}')
        # print(f'{trial}회차 output_gradiant_ct_w: {output1_gradiant_ct_w, output2_gradiant_ct_w}')
        # print(f'{trial}회차 output_gradiant_ct_b: {output1_gradiant_ct_b, output2_gradiant_ct_b}')

        hidden1_w -= (eta * hidden1_gradiant_ct_w)
        hidden2_w -= (eta * hidden2_gradiant_ct_w)
        hidden3_w -= (eta * hidden3_gradiant_ct_w)

        hidden1_b -= (eta * hidden1_gradiant_ct_b)
        hidden2_b -= (eta * hidden2_gradiant_ct_b)
        hidden3_b -= (eta * hidden3_gradiant_ct_b)
        # print(f'{trial}회차 hidden_w: {hidden1_w, hidden2_w, hidden3_w}')
        # print(f'{trial}회차 hidden_b: {hidden1_b, hidden2_b, hidden3_b}')

        output1_w -= (eta * output1_gradiant_ct_w)
        output2_w -= (eta * output2_gradiant_ct_w)
        # print(f'{trial}회차 output_w:{output1_w, output2_w}')

        output1_b -= (eta * output1_gradiant_ct_b)
        output2_b -= (eta * output2_gradiant_ct_b)
        # print(f'{trial}회차 output_b:{output1_b, output2_b}')

        c_record.append(ct)
    print(f'hidden_w:{hidden1_w, hidden2_w, hidden3_w}')
    print(f'hidden_b:{hidden1_b, hidden2_b, hidden3_b}')
    print(f'output_w:{output1_w, output2_w}')
    print(f'hidden_b:{output1_b, output2_b}')
    print(f'c이력:{c_record}')

    print(f'식별테스트 결과:{test_results}')


if __name__ == '__main__':
    main()
