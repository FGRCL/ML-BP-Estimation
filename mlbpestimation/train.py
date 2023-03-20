import argparse

from mlbpestimation.hypothesis import hypotheses_repository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hypothesis', choices=hypotheses_repository.keys(), nargs=1)
    args = parser.parse_args()
    h = hypotheses_repository[args.hypothesis[0]]
    h.train()


if __name__ == '__main__':
    main()
