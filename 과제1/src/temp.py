import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

# CSV 파일을 읽어 들인다
data = pd.read_csv("data2.csv", encoding='utf-8')

# X1 열을 x1List 배열에 담는다
x1List = data["Normalized X1 kernel_area"].astype("float")
# X2 열을 x2List 배열에 담는다
x2List = data["Normalized X2 kernel_length"].astype("float")
# 분류 열을 y배열에 담는다
yList = data["Wheat Varieties"]

n = 2  # features of x
m = 210  # 학습 데이터 개수
half_m = int(m / 2)


# Sigmoid 함수와 그 도함수를 나타낸다. 범위는 -1~1이므로 a=1로 한다
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
def sigmoid(w0, w1, w2, x1, x2, a):
    return 1 / (1 + np.exp(-a * (w0 + w1 * x1 + w2 * x2)))


# def diff_sigmoid(z):
#     return sigmoid(z, 1) * (1 - sigmoid(z, 1))


def h(w0, w1, w2, x1, x2):
    return sigmoid(w0, w1, w2, x1, x2, 1)


def cost(hxi, yi):
    return -yi * np.log(hxi) - (1 - yi) * np.log(1 - hxi)


def j(m):
    sum_of_cost = 0
    for i in range(m):
        # 초기 w값은 w0 = 0, w1 = 1, w2 = 1로 한다
        sum_of_cost += cost(h(x1List[i], x2List[i], 0, 1, 1), yList[i] == "Kama")
    return 1 / m * sum_of_cost


wList = [0.0, 1.0, 1.0]


def update_w1(w, alpha):
    sum_of_cost = 0
    for i in range(m):
        # 초기 w값은 w0 = 0, w1 = 1, w2 = 1로 한다
        sum_of_cost += (h(w[0], w[1], w[2], x1List[i], x2List[i]) - float("Kama" in yList[i])) * x1List[i]
    return w[1] - alpha * sum_of_cost


def update_w2(w, alpha):
    sum_of_cost = 0
    for i in range(m):
        # 초기 w값은 w0 = 0, w1 = 1, w2 = 1로 한다
        sum_of_cost += (h(w[0], w[1], w[2], x1List[i], x2List[i]) - float("Kama" in yList[i])) * x2List[i]
    return w[2] - alpha * sum_of_cost


# TODO: 나중에는 w1뿐만아니라 w0, w1도 한꺼번에 업데이트가 이루어져야 한다

# W1값 학습은 잘 됐는데 중지할 타이밍을 못잡는다 - 이건 기울기 구하면 됨. 아래는 반복학습하는 코드
new_wList = [0, 0, 0]
for i in range(1000):
    new_wList[1] = update_w1(wList, 0.01)
    new_wList[2] = update_w2(wList, 0.01)
    wList[1] = new_wList[1]
    wList[2] = new_wList[2]
    print(wList[1])
    print(wList[2])

# 임시로 그래프 그려봄
for i in range(m):
    if float("Kama" in yList[i]):
        plt.plot(x1List[i], x2List[i], 'b.')
    else:
        plt.plot(x1List[i], x2List[i], 'r.')

tmpX = np.arange(-1, 1, 0.1)
tmpY = wList[0] + wList[1] * tmpX + wList[2] * tmpX
plt.plot(tmpX, tmpY)
plt.show()
exit(0)

# 참고자료

n = 2  # features of x
m = 210  # 학습 데이터 개수
half_m = int(m / 2)

x1_data = x1List
x2_data = x2List
y_data = data["Wheat Varieties"]

# 초기 파라미터값 설정 (아몰랑 랜덤으로 찍을래)
# updated_wi = np.random.normal(0.0, 0.5, n + 1).T
# print(updated_wi)
updated_wi = np.array([0.0, 1.0, 1.0])
print("Initial w0, w1, w2")
print(updated_wi)

iteration = 5000
learning_rate = 0.05

# 세타값이 업데이트된 w0, w1, w2 (각각)
# 감마값이 w0, w1, w2를 J(wi)에 넣었을 때 각각의 기울기이다.
for current_iter in range(iteration):
    gradient_of_wi = np.zeros(n + 1).T
    for i in range(m):
        x = np.array((1, x1_data[i], x2_data[i]))
        x = np.resize(x, n + 1).T
        gradient_of_wi += (sigmoid(updated_wi.T @ x) - float("Kama" in y_data[i])) * x
    updated_wi -= learning_rate * gradient_of_wi
    print("=================" + str(current_iter) + "===============")
    print("Gradient of w0, w1, w2")
    print(gradient_of_wi)
    print("Updated w0, w1, w2")
    print(updated_wi)
    print("=========================================================")

tmpX = np.arange(-1, 1, 0.1)
tmpY = (updated_wi[0] + updated_wi[1] * tmpX) / updated_wi[1]
#
print(tmpX)

for i in range(m):
    if float("Kama" in y_data[i]):
        plt.plot(x1_data[i], x2_data[i], 'b.')
    else:
        plt.plot(x1_data[i], x2_data[i], 'r.')
plt.plot(tmpX, tmpY)
plt.show()
