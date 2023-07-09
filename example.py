import argparse

import cv2
from polygonize import Polygonize


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="画像のファイルパス")
    parser.add_argument("--max_point_num", type=int, default=1000, help="サンプル点の最大数")
    parser.add_argument("--threshold", type=float, default=0.02, help="サンプル点を取得するための重みの閾値")
    parser.add_argument("--output", type=str, default="result.jpg", help="出力画像のファイルパス")
    return parser.parse_args()


def main() -> None:
    args = parse_arg()
    polygonizer = Polygonize()

    img = cv2.imread(args.image)
    triangle_samples = polygonizer.polygonize(img, args.max_point_num, threshold=args.threshold)
    ret_img = polygonizer.render(img, triangle_samples)
    cv2.imwrite(args.output, ret_img)


if __name__ == "__main__":
    main()
