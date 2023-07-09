import numpy as np
from scipy.signal import convolve2d
from skimage.color import rgb2gray


def canny(image: np.ndarray, blur_radius: int = 2) -> np.ndarray:
    """画像に対してCannyエッジ検出アルゴリズムを適用します。

    この関数は、まず入力画像をグレースケールに変換し、次にノイズを減らすためにブラー(ぼかし)フィルターを適用します。
    その後、エッジを検出するフィルターを使用し、エッジマップに閾値を適用します。最後に、結果のエッジマップを最大値で
    正規化します。

    この関数は、エッジ検出には簡易な3x3フィルター、ブラーフィルターには一様フィルターを使用します。エッジ検出フィルターは
    ラプラシアンフィルターで、急速な強度の変化を強調します。ブラーフィルターはボックスフィルターで、各要素が同じ値を持ちます。

    この関数は畳み込みに対して対称境界条件('symm')を使用します。これは、入力が最後のピクセルのエッジについてミラーリング
    されて拡張されることを意味します。

    Args:
        image (np.ndarray): 入力画像。グレースケール画像の場合は2D配列、
                            RGB画像の場合は色チャンネルを表す3次元を持つ3D配列を指定します。
        blur_radius (int, optional): 画像に適用するブラーフィルターの半径、デフォルトは2。
                                     ブラーフィルターの半径が大きいほど、
                                     エッジ検出前に画像がよりぼかされます。
                                     Defaults to 2.

    Returns:
        np.ndarray: 処理後の画像。それぞれの値は対応するピクセルでのエッジの強度を表す2D配列です。
                    値は範囲 [0, 1] に正規化されます。
    """
    edge_threshold = 3 / 256

    gray_img = rgb2gray(image)
    blur_filt = np.ones(shape=(2*blur_radius+1, 2*blur_radius+1)) / ((2*blur_radius+1) ** 2)
    blurred = convolve2d(gray_img, blur_filt, mode="same", boundary="symm")
    edge_filt = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    edge = convolve2d(blurred, edge_filt, mode="same", boundary="symm")
    for idx, val in np.ndenumerate(edge):
        if val < edge_threshold:
            edge[idx] = 0
    dense_filt = np.ones((3,3))
    dense = convolve2d(edge, dense_filt, mode="same", boundary="symm")
    dense /= np.amax(dense)
    return dense


def threshold_samples(n: int, weights: np.ndarray, threshold: float, seed: int) -> list:
    """与えられた `threshold` 以上の重みを持つ `weights` から `n` 個の要素をランダムにサンプリングします。

    この関数は、numpyの乱数生成器を使用して生成器を作成します。次に、重みが `threshold` 以上の `weights` から
    候補のインデックスのリストを生成します。 `n` を満たすための十分な候補がない場合、ValueErrorを発生させます。
    最後に、候補から置換なしで `n` 個のインデックスをランダムにサンプリングします。
    この関数はnumpyのデフォルトの乱数生成器を使用します。

    Args:
        n (int): 抽出するサンプルの数。
        weights (np.ndarray): 各候補の重みの配列。
        threshold (float): 候補が考慮されるための最小の重み。
        seed (int): シード値

    Raises:
        ValueError: `weights` に `threshold` を満たす十分な候補がない場合。

    Returns:
        list: `threshold` 以上の重みを持つ `weights` のインデックスをランダムに選んだ `n` 個のリスト。
              インデックスの順序はランダムです。
    """
    my_generator = np.random.default_rng(seed)
    candidates = np.array([idx for idx, weight in np.ndenumerate(weights) if weight >= threshold])
    if candidates.shape[0] < n:
        raise ValueError(f"Not enough candidate points for threshold {threshold}. "
                         f"Only {candidates.shape[0]} available.")
    return my_generator.choice(candidates, size=n, replace=False)
