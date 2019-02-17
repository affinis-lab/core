def add(x, y):
    return x + y

def get_inc():
    y = 4
    return lambda x: add(x, y)

if __name__ == "__main__":
    inc = get_inc()
    print(inc(3))