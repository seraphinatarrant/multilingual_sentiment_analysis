import argparse


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-r', dest='', default='', help='')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
