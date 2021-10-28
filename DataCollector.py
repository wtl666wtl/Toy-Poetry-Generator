import os
import json
import re
import numpy as np

def _characterCollect(constrain=7, src='./chinese-poetry-master/json/', category="poet.tang"):
    def sentenceParse(para):
        result, number = re.subn(u"（.*）", "", para)
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r, number = re.subn(u"。。", u"。", r)
        return r

    def handleJson(file):
        rst = []
        data = json.loads(open(file,encoding='UTF-8').read())
        for poetry in data:
            pdata = ""
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    data = []
    for filename in os.listdir(src):
        if filename.startswith(category):
            data.extend(handleJson(src + filename))
    return data

def _parseRawData(constrain=7, src='./chinese-poetry-master/json/', category="poet.tang"):
    def sentenceParse(para):
        result, number = re.subn(u"（.*）", "", para)
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r, number = re.subn(u"。。", u"。", r)
        return r

    def getNum(file):
        data = json.loads(open(file, encoding='UTF-8').read())
        # 统计字频
        for poetry in data:
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                for x in sentence:
                    if x != '，' and x != '。' and x != '！' and x != '（' and x != '）' and x != '《' and x != '》'\
                            and x != '一' and x != '不' and x != '無' and x != '相' and x != '未' and x != '莫' and x != '何':
                        num[word2ix[x]] = num[word2ix[x]] + 1

    def handleJson(file):
        rst = []
        print(file)
        data = json.loads(open(file,encoding='UTF-8').read())
        for poetry in data:
            pdata = ""
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                maxnum = -1
                bestfit = None
                #重构诗句
                for x in sentence:
                    if  x != '，' and x != '。' and x != '！' and x != '（' and x != '）' and x != '《' and x != '》' and num[word2ix[x]] > maxnum \
                            and x != '一' and x != '不' and x != '無' and x != '相' and x != '未' and x != '莫' and x != '何':
                        maxnum = num[word2ix[x]]
                        bestfit = x
                if(bestfit!=None):
                    pdata += bestfit + '>' + sentence
            print(pdata)
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    num = [0 for x in range(0, len(word2ix))]
    for filename in os.listdir(src):
        if filename.startswith(category):
            getNum(src + filename)

    data = []
    for filename in os.listdir(src):
        if filename.startswith(category):
            data.extend(handleJson(src + filename))
    return data


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


if __name__ == '__main__':
    # 粗统计汉字
    data = _characterCollect(7, './chinese-poetry-master/json/', 'poet.tang')
    words = {_word for _sentence in data for _word in _sentence}
    word2ix = {_word: _ix for _ix, _word in enumerate(words)}
    word2ix['<EOP>'] = len(word2ix)  # 终止标识符
    word2ix['<START>'] = len(word2ix)  # 起始标识符
    word2ix['</s>'] = len(word2ix)  # 空格
    word2ix['>'] = len(word2ix)  # > 间隔号
    ix2word = {_ix: _word for _word, _ix in list(word2ix.items())}
    # 提取关键字并按要求拼好古诗
    data = _parseRawData(7, './chinese-poetry-master/json/', 'poet.tang')
    for i in range(len(data)):
        data[i] = ["<START>"] + list(data[i]) + ["<EOP>"]

    # 将每首诗歌保存的内容由字变成数
    new_data = [[word2ix[_word] for _word in _sentence] for _sentence in data]

    # 诗歌为maxlen，补足空格或删除多余，主要学习七言绝句80字符已经足够
    pad_data = pad_sequences(new_data, maxlen=80, padding='pre', truncating='post', value=len(word2ix) - 1)

    # 保存成二进制文件，省略路径名
    np.savez_compressed('.../new.npz', data=pad_data, word2ix=word2ix, ix2word=ix2word)