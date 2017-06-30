import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # マーカーとカラーマップの準備
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "grey", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # グリッドポイントの生成
    xx1, xx2, = np.meshgrid(np.arange(x1_min, x1_max, resolution),np.arange(x2_min, x2_max, resolution))

    # 各特徴量を１次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

    # テストサンプルを目立たせる（点を○で表示）
    if test_idx:
        X_test, y_test, = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c="", alpha=1.0, linewidths=1, marker="o", s=55, label="test set")
    
#ここからデータセット
iris = datasets.load_iris()

x = iris.data[:,[2,3]]

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# エントロピーを指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion="entropy", n_estimators=5, random_state=1, n_jobs=2)

# 決定木のモデルにトレーニングデータを適合させる
forest.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel("petal length [cm]")
plt.ylabel("petal width [cm]")
plt.legend(loc="upper left")
plt.show()

filename = "decitree.png"
plt.savefig(filename)
