
class HexColors:
    def __init__(self):
        self.hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                     '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in self.hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    def get_hex_color(self, i):
        return self.hexs[int(i) % self.n]

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

if __name__ == "__main__":
    hex_color = '#FF0000'
    rgb_color = HexColors.hex2rgb(hex_color)
    print(f"RGB Color: {rgb_color}")
    colors = HexColors()
    index = 3
    rgb_color = colors(index)
    hex_color = colors.get_hex_color(index)
    print(f"RGB Color: {rgb_color}")
    print(f"Hex Color: {hex_color}")