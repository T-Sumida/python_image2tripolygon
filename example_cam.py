import argparse

import cv2
from polygonize import Polygonize


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=int, default=0, help="カメラデバイスID")
    parser.add_argument("--max_point_num", type=int, default=1000, help="サンプル点の最大数")
    parser.add_argument("--threshold", type=float, default=0.02, help="サンプル点を取得するための重みの閾値")
    return parser.parse_args()


def main() -> None:
    args = parse_arg()
    polygonizer = Polygonize()

    cap = cv2.VideoCapture(args.video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        triangle_samples = polygonizer.polygonize(frame, args.max_point_num, threshold=args.threshold)
        ret_img = polygonizer.render(frame, triangle_samples)
        cv2.imshow("frame", ret_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
