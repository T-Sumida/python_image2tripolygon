import cv2
import numpy as np
from scipy.spatial import Delaunay
from skimage.draw import polygon

from polygonize.utils import canny, threshold_samples


class Polygonize:
    def __init__(self, seed: int = 1234) -> None:
        """Polygonizeクラスのインスタンスを初期化します。

        このクラスは画像をポリゴン化(多角形化)します。

        Args:
            seed (int, optional): 乱数生成器のシード値. Defaults to 1234.
        """
        self._seed = seed

    def polygonize(self, image: str | np.ndarray, max_point_num: int, threshold: float = 0.02) -> np.ndarray:
        """
        画像を多角形化します。

        この関数は、Cannyエッジ検出アルゴリズムを使用して重みを求め、指定された閾値以上の重みを持つサンプル点を取得します。
        サンプル点をDelaunay三角形分割により三角形に分割し、その三角形を返します。

        パラメータ
        ----------
        image : str | np.ndarray
            多角形化する画像。ファイルパス(str)またはnp.ndarray形式の画像を指定します。

        max_point_num : int
            サンプル点の最大数。

        threshold : float, optional
            サンプル点を取得するための重みの閾値、デフォルトは0.02。

        戻り値
        -------
        np.ndarray
            画像を三角形に分割した結果。各三角形は頂点の座標を持つ配列として表されます。
        """
        sample_points = self._get_sample_points(image, max_point_num, threshold=threshold)
        triangulation = Delaunay(sample_points)
        triangles = sample_points[triangulation.simplices]
        return triangles

    def _get_sample_points(self, image: np.ndarray, max_point_num: int, threshold: float) -> np.ndarray:
        """
        閾値以上の重みを持つサンプル点を取得します。

        この関数は、Cannyエッジ検出アルゴリズムを使用して重みを求め、指定された閾値以上の重みを持つサンプル点を取得します。
        画像の四隅をサンプル点に追加し、結果のサンプル点を返します。

        パラメータ
        ----------
        image : np.ndarray
            サンプル点を取得する画像。

        max_point_num : int
            サンプル点の最大数。

        threshold : float
            サンプル点を取得するための重みの閾値。

        戻り値
        -------
        np.ndarray
            サンプル点の座標を持つ配列。
        """
        height, width, _ = image.shape
        weights = canny(image)
        sample_points = threshold_samples(max_point_num, weights, threshold, self._seed)
        corners = np.array([[0, 0], [0, height-1], [width-1, 0], [width-1, height-1]])
        return np.append(sample_points, corners, axis=0)

    def render(self, image: np.ndarray, polygons: np.ndarray) -> np.ndarray:
        """
        三角形に分割された画像をレンダリングします。

        この関数は、指定された三角形を使用して低ポリゴンの画像を作成します。各三角形の色は、その三角形内のピクセルの色の平均値で設定されます。

        パラメータ
        ----------
        image : np.ndarray
            元の画像。

        polygons : np.ndarray
            画像を三角形に分割した結果。各三角形は頂点の座標を持つ配列として表されます。

        戻り値
        -------
        np.ndarray
            レンダリングされた低ポリゴンの画像。
        """

        low_poly = np.empty(shape=(2 * image.shape[0], 2 * image.shape[1], image.shape[2]), dtype=np.uint8)
        for triangle in polygons:
            rr, cc = polygon(2 * triangle[:,0], 2 * triangle[:,1], low_poly.shape)
            color = np.mean(image[polygon(triangle[:,0], triangle[:,1], image.shape)], axis=0)
            low_poly[rr,cc] = color

        low_poly = cv2.resize(low_poly, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        return low_poly
