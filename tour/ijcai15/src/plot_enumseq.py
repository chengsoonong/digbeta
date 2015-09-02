import sys
import numpy as np
import matplotlib.pyplot as plt

#def doplot(listoflist, alist, blist):
    #print(listoflist)
    #plt.boxplot(listoflist)
    #xindex = list(range(1, len(alist) + 1))
    #plt.plot([], 'sg', xdata=xindex, ydata=alist)
    #plt.plot([], 'sr', xdata=xindex, ydata=blist)
    #plt.show()

def doplot(listoflist, alist, labels):
    """Plot data 10-by-10"""
    assert(len(listoflist) == len(alist))
    assert(len(alist) == len(labels))
    idx = 0
    while idx < len(alist):
        if idx + 10 > len(alist):
            N = len(alist[idx:])
            xindex = range(1, N+1)
            plt.boxplot(listoflist[idx:], labels=labels[idx:])
            plt.plot([], 'sg', xdata=xindex, ydata=alist[idx:], label='actual trajectory')
        else:
            xindex = range(1, 11)
            plt.boxplot(listoflist[idx:idx+10], labels=labels[idx:idx+10])
            plt.plot([], 'sg', xdata=xindex, ydata=alist[idx:idx+10], label='actual trajectory')
        plt.xlabel('trajectory length')
        plt.ylabel('score')
        plt.legend()
        plt.title('Scores of enumerated trajectories')
        plt.show()
        idx += 10
        a = input('Press any key to continue ...')


def readseqs(fname):
    """Load sequence info"""
    scores = []
    seqlen = ''
    with open(fname) as f:
        getlen = True
        for line in f:
            if getlen: 
                slen = len(line.split(']')[0].split(','))
                seqlen = str(slen)
                getlen = False
            s = line.strip().split(']')[1].split(' ')[1]
            scores.append(float(s))
    return scores, seqlen


def main(fnamelist):
    """Main Procedure"""
    actscores = []
    seqscores = []
    seqlens = []
    with open(fnamelist) as f:
        for fname in f:
            scores, seqlen = readseqs(fname.strip())
            assert(len(scores) > 1)
            actscores.append(scores[0])
            #maxscores.append(scores[1])
            seqscores.append(list(scores[1:]))
            seqlens.append(seqlen)

    assert(len(seqscores) == len(actscores))
    assert(len(actscores) == len(seqlens))
    doplot(seqscores, actscores, seqlens)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage:', sys.argv[0], 'FILE_LIST_NAME')
        sys.exit(0)

    fnamelist = sys.argv[1]
    main(fnamelist)
    #a = input('...')
    #print(a)

