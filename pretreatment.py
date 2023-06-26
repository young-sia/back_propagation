from import_dataset import *


def pretreatment_train():
    data = import_dataset('Data')
    data = data.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], axis = 1)
    data = data.iloc[4:9]

    answer = data.iloc[4]
    answer = list(map(int, [x for x in answer if not pd.isna(x)]))
    train_result = []

    for criteria in answer:
        if criteria == 0:
            train_result.append([1, 0])
        elif criteria == 1:
            train_result.append([0, 1])
        else:
            train_result.append([0, 0])

    data = data.drop(8, axis=0)
    data.dropna(inplace=True, axis=1)
    row0 = list(map(int,data.iloc[0]))
    row1 = list(map(int,data.iloc[1]))
    row2 = list(map(int, data.iloc[2]))
    row3 = list(map(int, data.iloc[3]))

    divided0 = [row0[i:i + 3] for i in range(0, len(row0), 3)]
    divided1 = [row1[i:i + 3] for i in range(0, len(row1), 3)]
    divided2 = [row2[i:i + 3] for i in range(0, len(row2), 3)]
    divided3 = [row3[i:i + 3] for i in range(0, len(row3), 3)]

    result = [list(x) for x in zip(divided0, divided1, divided2, divided3)]
    return result, train_result


def pretreatment_test():
    data = import_dataset('테스트')
    data = data.drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14], axis = 1)

    data = data.loc[[2,3,4,5]]

    row0 = list(map(int, data.iloc[0]))
    row1 = list(map(int, data.iloc[1]))
    row2 = list(map(int, data.iloc[2]))
    row3 = list(map(int, data.iloc[3]))

    divided0 = [row0[i:i + 3] for i in range(0, len(row0), 3)]
    divided1 = [row1[i:i + 3] for i in range(0, len(row1), 3)]
    divided2 = [row2[i:i + 3] for i in range(0, len(row2), 3)]
    divided3 = [row3[i:i + 3] for i in range(0, len(row3), 3)]

    result = [list(x) for x in zip(divided0, divided1, divided2, divided3)]

    return result













