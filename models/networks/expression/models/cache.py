import math


def conflict(data, key1, key2):
    if data[key1] or data[key2]:
        if data[key2] == 0:
            data[key2] = -1
        else:
            data[key1] = -1
    return data


# def funx(x, v, max_value):
#     y = math.pow(max(0, x), max(0, v))
#     y = min(y, max_value)
#     return y

def fun_sensitive(x):
    x = min(max(0, x), 100)
    y = (100**2-(x-100)**2)**(1/2)
    return y

def fun_insensitive(x, thre):
    return 100/(thre-min(max(x, 0), thre-1))

def getValue(data):
    # 头部正负拆分
    if data['headup'] < 0:
        data['headup'] = 0
        data['headdown'] = abs(data['headdown'])
    else:
        data['headdown'] = 0

    if data['headleft'] < 0:
        data['headleft'] = 0
        data['headright'] = abs(data['headright'])
    else:
        data['headright'] = 0

    if data['headrollright'] < 0:
        data['headrollright'] = 0
        data['headrollleft'] = abs(data['headrollleft'])
    else:
        data['headrollleft'] = 0

    #下嘴唇调整
    data["mouthlowerdownleft"] = max(min((data["mouthlowerdownleft"] - 4.0) * 2.5, 100.0), 0.0)
    data["mouthlowerdownright"] = max(min((data["mouthlowerdownright"] - 4.0) * 2.5, 100.0), 0.0)

    # #  #  瞪眼调整
    data["eyewideleft"] = max(min(fun_sensitive((data["eyewideleft"] - 1.0) * 2), 100.0), 0.0)
    data["eyewideright"] = max(min((fun_sensitive(data["eyewideright"] - 1.0) * 2), 100.0), 0.0)


    #  皱眉毛
    data["browdownright"] = max(min((data["browdownright"] - 20.0) * 1.5, 100.0), 0.0)
    data["browdownleft"] = max(min((data["browdownleft"] - 20.0) * 1.5, 100.0), 0.0)


    #  抬眉毛调整
    data["browinnerup"] = max(min((data["browinnerup"] - 10.0) * 2.0, 100.0), 0.0)
    data["browouterupleft"] = max(min((data["browouterupleft"]) * 2.0, 100.0), 0.0)
    data["browouterupright"] = max(min((data["browouterupright"]) * 2.0, 100.0), 0.0)


    # #  嘴角调整
    data["mouthdimpleleft"] = max(min((data["mouthdimpleleft"] - 1.0) * 3.0, 100.0), 0.0)
    data["mouthdimpleright"] = max(min((data["mouthdimpleright"] - 1.0) * 3.0, 100.0), 0.0)

    # #  颧骨调整
    data["cheeksquintleft"] = max(min((data["cheeksquintleft"] - 6.0) * 2.0, 100.0), 0.0)
    data["cheeksquintright"] = max(min((data["cheeksquintright"] - 6.0) * 2.0, 100.0), 0.0)


    #  微笑调整
    data["mouthsmileleft"] = max(min((data["mouthsmileleft"]) * 1.5, 100.0), 0.0)
    data["mouthsmileright"] = max(min((data["mouthsmileright"]) * 1.5, 100.0), 0.0)

    if data["mouthsmileleft"] >= 30.0:
        data["eyesquintleft"] = data["mouthsmileleft"]
    

    if data["mouthsmileright"] >= 30.0:
        data["eyesquintright"] = data["mouthsmileright"]

    #  眯眼调整
    data["eyesquintleft"] = max(0.0, fun_sensitive(data["eyesquintleft"] - 40.0))
    data["eyesquintright"] = max(0.0, fun_sensitive(data["eyesquintright"] - 40.0))

    #  闭眼调整
    data["eyeblinkleft"] = max(0.0, fun_insensitive(data["eyeblinkleft"], 65) - data["cheeksquintleft"] / 4.0)
    data["eyeblinkright"] = max(0.0, fun_insensitive(data["eyeblinkright"], 65) - data["cheeksquintright"] / 4.0)

    #  舌头调整
    data["tongueout"] = fun_insensitive(data["tongueout"], 80.0)

    # #  抬嘴调整
    data["mouthupperupleft"] = max(min((data["mouthupperupleft"] - 2.0) * 1.5, 100.0), 0.0)
    data["mouthupperupright"] = max(min((data["mouthupperupright"] - 2.0) * 1.5, 100.0), 0.0)

    # #  张嘴调整
    data["jawopen"] = data["jawopen"] - 5.0 - (data["mouthupperupleft"] + data["mouthupperupright"]) / 30.0


    # #  左右下颚调整
    data["jawleft"] = max(min((data["jawleft"] - 3.0) * 2.5, 100.0), 0.0)
    data["jawright"] = max(min((data["jawright"] - 3.0) * 2.5, 100.0), 0.0)

    # #  鼓嘴调整
    data["cheekpuff"] = fun_insensitive(data["cheekpuff"], 70.0)


    #  左右同步
    sync(data, "eyelookinright", "eyelookoutleft", 'max')
    sync(data, "eyelookinleft", "eyelookoutright", 'max')
    sync(data, "eyelookdownleft", "eyelookdownright", 'max')
    sync(data, "eyelookupright", "eyelookupleft", 'max')
    sync(data, "nosesneerleft", "nosesneerright", 'max')
    sync(data, "browdownleft", "browdownright", 'max')
    sync(data, "browouterupleft", "browouterupright", 'max')

    return data


# 用于将连续输出转换至分级输出
def step(low_bound, up_bound, set_value, real_value):
    if len(low_bound) != len(up_bound) or len(low_bound) != len(set_value):
        raise ValueError('The length of the lists are not equal!')
    l = len(low_bound)
    for i in range(l):
        if low_bound[i] <= real_value <= up_bound[i]:
            return set_value[i]
    return -1


# 左右同步
def sync(res, key1, key2, max_min):
    if max_min == 'max':
        max_value = max(res[key2], res[key1])
    else:
        max_value = min(res[key2], res[key1])
    res[key1], res[key2] = max_value, max_value
    return res
