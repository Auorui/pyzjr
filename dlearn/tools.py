from pyzjr.video import Timer

class Runcodes:
    """
    Comparing the running time of different algorithms.
    example:
        with Runcodes("inference time"):
            output = ...
    """
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.5f} sec')

class LoadingBar:
    def __init__(self, length: int = 40):
        """Simple dynamic display bar
        example:
            loading_bar = LoadingBar()
            for i in range(101):
                progress = i / 100.0
                bar_string = loading_bar(progress)
                print(f"\r[{bar_string}] {i}% ", end="")
                time.sleep(0.1)  # 模拟加载延迟
            print("\nLoading Complete!")
        """
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"

def show_config(**kwargs):
    """display configuration"""
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)