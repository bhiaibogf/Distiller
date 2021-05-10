from urllib.parse import unquote


def decode(url):
    s = unquote(url)
    s = s.split('?tex=')[1]
    return s.replace('+', '')


def main():
    url = input()
    # s = 'https://www.zhihu.com/equation?tex=L+%3D+%5Cfrac%7Bd+%5CPhi+%7D%7Bd%5Comega+d+A%5E%7B%5Cbot+%7D+%7D+'
    print(decode(url))


if __name__ == '__main__':
    main()
