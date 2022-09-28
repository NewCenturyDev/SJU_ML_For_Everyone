import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 경사하강법을 이용하여 로지스틱 회귀 문제를 푸는 데 필요한 수식 및 로직 함수
# Sigmoid 함수를 나타낸다. a=1로 한다
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 예측함수 h(x)를 나타낸다
def hx(w, x1_list, x2_list):
    z = np.array(w[0] + w[1] * np.array(x1_list) + w[2] * np.array(x2_list))
    return sigmoid(z)


# 비용함수를 나타낸다
def cost(w, x1_list, x2_list, y_list):
    predict_y = hx(w, x1_list, x2_list)
    return -1 * sum(y_list * np.log(predict_y) + (1 - y_list) * np.log(1 - predict_y))


# 비용함수의 기울기를 나타낸다
def grad(w, x1_list, x2_list, y_list):
    y_pred = hx(w, x1_list, x2_list)
    g = [0] * 3
    g[0] = -1 * sum(y_list * (1 - y_pred) - (1 - y_list) * y_pred)
    g[1] = -1 * sum(y_list * (1 - y_pred) * x1_list - (1 - y_list) * y_pred * x1_list)
    g[2] = -1 * sum(y_list * (1 - y_pred) * x2_list - (1 - y_list) * y_pred * x2_list)
    return g


# 경사하강법에 따라 비용을 최소화하는 함수이다 (경사하강법을 적용하는 함수)
# w_new = 갱신된 w(파라미터)값
# w_prev = 갱신되지 않은 w(파라미터)값
# learning_rate = 학습상수(알파값)
def descent(df, w_new, w_prev, learning_rate):
    print("Initiate gradient descent")
    print(w_prev)
    print(cost(w_prev, df["x1"], df["x2"], df["y"]))
    iter_cnt = 0
    while True:
        print("==================== Iter-" + str(iter_cnt) + " ===================")
        # 파라미터값을 갱신하여 경사하강 실행. (1루프 = 학습 1회차)
        w_prev = w_new
        w0 = w_prev[0] - learning_rate * grad(w_prev, df["x1"], df["x2"], df["y"])[0]
        w1 = w_prev[1] - learning_rate * grad(w_prev, df["x1"], df["x2"], df["y"])[1]
        w2 = w_prev[2] - learning_rate * grad(w_prev, df["x1"], df["x2"], df["y"])[2]
        w_new = [w0, w1, w2]
        print(w_new)
        print(cost(w_new, df["x1"], df["x2"], df["y"]))
        print("==================== End-" + str(iter_cnt) + " ===================")

        # 추가적인 경사하강을 하여도 w값들의 차이가 10^-6 이하로 변동이 매우 작다면 학습을 중지하고 파라미터값을 반환한다
        if (w_new[0] - w_prev[0]) ** 2 + (w_new[1] - w_prev[1]) ** 2 + (w_new[2] - w_prev[2]) ** 2 < pow(10, -6):
            return w_new

        # 반복횟수가 1000회를 초과할 경우 반환한다
        # if iter_cnt >= 1000:
        #     return w_new
        iter_cnt += 1


# Y값을 1 또는 0으로 변환시키는 함수이다 (특정 라벨에 해당하면 1, 해당하지 않으면 0)
def label_to_true_or_false(label, label_column):
    # 예를 들어 label에 Kuma를 넣어준다면, Kuma가 아닌지에 대해 One-vs-All 방식을 사용하여 y값을 0 or 1로 확정
    y_data_column = []
    for i in range(m):
        if float(label in label_column[i]):
            y_data_column.append(1.0)
        else:
            y_data_column.append(0.0)
    return y_data_column


# One-vs-All 방식을 적용하여 Kuma, Rosa, Canadian을 분류하는 문제를
# 3개의 하위 문제로 분할하여 로지스틱 회귀와 경사하강법을 이용한 학습 및 경계선 도출
def machine_learn(df, label):
    # 파라미터 초기값 설정
    initial_w = [0, 1, 1]

    # 학습 시작
    ideal_w = descent(initial_w, initial_w, 0.001)
    print("RESULT: Ideal_w of " + label + " is")
    print(ideal_w)

    # 학습을 통해 도출한 경계선을 그래프에 그린다
    x = np.array(range(-2, 2))
    y = (-ideal_w[0] - ideal_w[1] * x) / ideal_w[2]
    plt.plot(x, y)


# <메인 함수>
# 문제 1번에서 정규화된CSV 파일을 읽어 들인다
data = pd.read_csv("data_normalized.csv", encoding='utf-8')

# X1 열을 x1List 배열에 담는다
x1DataColumn = data["Normalized X1 kernel_area"].astype("float")
# X2 열을 x2List 배열에 담는다
x2DataColumn = data["Normalized X2 kernel_length"].astype("float")
# 분류 열을 y배열에 담는다
yLabelColumn = data["Wheat Varieties"]

# 데이터에 대한 기본정보
n = 2  # features of x(특징값 개수)
m = len(x1DataColumn)  # 학습 데이터 개수

# 데이터프레임으로 변환 (그래프에 데이터 표출용)
dfAll = pd.DataFrame(dict(x1=x1DataColumn, x2=x2DataColumn, label=yLabelColumn))

# 데이터프레임으로 변환 (각 Kuma인 것과 아닌것에 대한 문제, Rosa인 것과 아닌것에 대한 문제, Canadian인 것과 아닌것에 관한 문제 학습용)
dfKuma = pd.DataFrame(dict(x1=x1DataColumn, x2=x2DataColumn, y=label_to_true_or_false("Kuma", yLabelColumn)))
dfRosa = pd.DataFrame(dict(x1=x1DataColumn, x2=x2DataColumn, y=label_to_true_or_false("Rosa", yLabelColumn)))
dfCanadian = pd.DataFrame(dict(x1=x1DataColumn, x2=x2DataColumn, y=label_to_true_or_false("Canadian", yLabelColumn)))

# 데이터들을 좌표평면 위에 표시한다
colors = {"Kuma": 'red', "Rosa": 'green', "Canadian": 'blue'}
fig, ax = plt.subplots()
grouped = dfAll.groupby('label')

for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x1', y='x2', label=key, color=colors[key])

# 분할된 3개의 이진분류 문제를 학습시키고 각각의 그래프(분류경계선)을 한 좌표평면에 그린다
machine_learn(dfKuma, "Kuma")
machine_learn(dfRosa, "Rosa")
machine_learn(dfCanadian, "Canadian")

# 그래프 라벨을 표시하고 그래프 결과를 표출한다
plt.xlabel('x1: Normalized X1 kernel_area')
plt.ylabel('x2: Normalized X2 kernel_length')
plt.show()
