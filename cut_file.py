def cut(fileName, num):
    output = 'cut-' + fileName

    fr = open(fileName,'br')
    fw = open(output, 'bw')

    for i in range(0, num):
        text = fr.read(1)
        fw.write(text)
    fr.close()
    fw.close()

def main():
    fileName = input('input file name, input nothing to quit: \n')

    while fileName != '':
        num = int(input('input character number: \n'))
        cut(fileName, num)
        fileName = input('input file name, input nothing to quit: \n')

if __name__ == '__main__':
    main()
