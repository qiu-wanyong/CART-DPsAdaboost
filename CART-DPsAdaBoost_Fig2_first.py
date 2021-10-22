

# Python3实现基于CART分类器的AdaBoost算法——CensusinCome分类
import numpy as np
from collections import OrderedDict # 集合

import copy
from math import log

from read_data_adult_fig2 import dt_data_fig2, predict_data_fig2



class CartAdaBoostAdult:
    def __init__(self, tree_length, modelcount, e, sign='a', train_dtdata=dt_data_fig2, pre_dtdata=predict_data_fig2):

        self.e = e  # 产生噪声的参数
        self.sign = sign  # 值为e 模型为DPsAdaBoost，值为f 模型为CART，其他值为CARTAdaBoost
        self.model_c = modelcount  # 弱分类器的个数
        self.tree_length = tree_length  # 分类器中树的个数

        # 训练数据
        self.train_dtdata = train_dtdata[0]['train']

        # 验证数据
        self.test_dtdata = train_dtdata[0]['test']

        # 预测数据
        self.pre_dtdata = pre_dtdata

        # 中间过程变量
        self.node_shujuji = {'0': self.train_dtdata}  # 存储每一个节点数据集的字典
        self.fenlei_shujuji = {'0': self.train_dtdata}  # 存储需要分类的数据集的字典

        # 叶子节点的集合
        self.leafnodes = []
        # 节点关系字典
        self.noderela = {}
        # 每一个节点的规则关系
        self.node_rule = {'0': []}

        # 样本权重,初始为1
        self.weight_list = np.array([1 for i in range(len(self.train_dtdata))])

    #  根据类别的数组计算基尼指数
    def jini_zhishu(self, exlist, weightlist):
        """

        :param exlist: 样本的标签序列
        :param weightlist: 对应的样本的权重序列
        :return:
        """
        exlist = np.array(exlist)
        weightlist = np.array(weightlist)
        dnum = 0
        leng = np.sum(weightlist)  # 样本权重的和
        for hh in list(set(exlist)):
            sum_f = np.sum(weightlist[exlist == hh])
            dnum += (sum_f / leng) ** 2
        return 1 - dnum

    #  计算基尼系数的函数
    def jini_xishu(self, tezheng, leibie, sample_sign):  # 输入特征数据，类别数据，返回最小基尼系数对应的值
        #  首先判断特征数据是连续、或者是分类的
        sign = 0
        try:
            tezheng[0] + 2
            # 证明是连续的
            sign = 1
        except TypeError:
            pass
        if sign:  # 连续变量
            # 去重、排序
            quzhong = np.array(sorted(list(set(tezheng))))
            # 判断是不是就一个值
            if len(quzhong) == 1:
                return False
            # 取中间值
            midd = (quzhong[:-1] + quzhong[1:]) / 2
            # 开始遍历每一个中间值，计算对应的基尼系数
            # 存储基尼系数的值
            save_ji, jini = np.inf, 0
            number = ''
            for mi in midd:
                #  计算基尼系数
                onelist = leibie[tezheng <= mi]
                twolist = leibie[tezheng > mi]
                # 获得对应样本的权重
                one_sign = sample_sign[tezheng <= mi]
                two_sign = sample_sign[tezheng > mi]
                one_weight = self.weight_list[[int(f) for f in one_sign]]
                two_weight = self.weight_list[[int(f) for f in two_sign]]


                # jini = (len(onelist) / length) * self.jini_zhishu(onelist) + (len(twolist) / length) * self.jini_zhishu(
                #     twolist)
                # 添加权重后
                jini = (np.sum(one_weight) / sum(list(one_weight) + list(two_weight)) *
                        self.jini_zhishu(onelist, one_weight) +
                       np.sum(two_weight) / sum(list(one_weight) + list(two_weight)) *
                        self.jini_zhishu(twolist, two_weight))

                if jini <= save_ji:
                    save_ji = jini
                    number = mi
            return number, save_ji
        else:  # 分类变量
            # 去重、排序
            quzhong = np.array(list(set(tezheng)))
            # 判断是不是就一个值
            if len(quzhong) == 1:
                return False
            # 开始遍历每一个值，计算对应的基尼系数
            # 存储基尼系数的值
            jini, save_ji = 0, np.inf
            number = ''
            for mi in quzhong:
                #  计算基尼系数
                onelist = leibie[tezheng == mi]
                twolist = leibie[tezheng != mi]
                # 获得对应样本的权重
                one_sign = sample_sign[tezheng == mi]
                two_sign = sample_sign[tezheng != mi]
                one_weight = self.weight_list[[int(f) for f in one_sign]]
                two_weight = self.weight_list[[int(f) for f in two_sign]]
                # jini = (len(onelist) / length) * self.jini_zhishu(onelist) + (len(twolist) / length) * self.jini_zhishu(
                #     twolist)
                # 添加权重后
                jini = (np.sum(one_weight) / sum(list(one_weight) + list(two_weight)) *
                        self.jini_zhishu(onelist, one_weight) +
                       np.sum(two_weight) / sum(list(one_weight) + list(two_weight)) *
                        self.jini_zhishu(twolist, two_weight))

                if jini <= save_ji:
                    save_ji = jini
                    number = mi
            return number, save_ji  # 该特征最好的分割值，以及该特征最小的基尼系数

    # 数据集确定分类特征以及属性的函数
    def feature_zhi(self, datadist):  # 输入的数据集字典，输出最优的特征编号，以及对应的值，还有基尼系数
        tezhengsign = ''
        number = np.inf
        jini = ''
        leib = datadist[:, -2].T  # 标签列
        sample_sign = datadist[:, -1].T  # 样本编号列
        for jil in range(1, len(datadist[0])-1):
            #  获取特征数据和类别数据
            tezhen = datadist[:, jil-1].T
            # 在其中选择最小的
            cresu = self.jini_xishu(tezhen, leib, sample_sign)
            # 判断这个特征可不可取
            if cresu:
                if cresu[1] <= number:
                    number = cresu[1]
                    tezhengsign = jil - 1
                    jini = cresu[0]
        if jini != '':
            return tezhengsign, jini, number  # 特征编号, 该特征最好的分割值，该数据集最小的基尼系数
        else:
            return False  # 这个数据集无法被分裂

    # 将数据集合分裂
    def devided_shujuji(self, datadis):  # 输入特征编号，对应的值，返回两个数据集
        # 运算的结果
        yuansuan = self.feature_zhi(datadis)
        if yuansuan:
            #  需要判断这个被选中的特征是连续还是离散的
            try:
                datadis[:, yuansuan[0]][0] + 2
                oneshujui = datadis[datadis[:, yuansuan[0]] <= yuansuan[1]]
                twoshujui = datadis[datadis[:, yuansuan[0]] > yuansuan[1]]
            except TypeError:
                oneshujui = datadis[datadis[:, yuansuan[0]] == yuansuan[1]]
                twoshujui = datadis[datadis[:, yuansuan[0]] != yuansuan[1]]
            return oneshujui, twoshujui, yuansuan
        else:
            return False

    # 决策树函数
    def grow_tree(self):
        while len(self.fenlei_shujuji) != 0:
            # 需要复制字典
            copy_dict = copy.deepcopy(self.fenlei_shujuji)
            # 开始遍历每一个需要分类的数据集
            for hd in self.fenlei_shujuji:
                #  在这里限制树的深度
                if len(hd) == self.tree_length + 1:
                    # 不需要在分裂
                    del copy_dict[hd]
                    # 添加到叶子节点的集合中
                    self.leafnodes.append(hd)
                else:
                    fenguo = self.devided_shujuji(copy_dict[hd])
                    if fenguo:
                        if len(set(fenguo[0][:, -2])) == 1:  # 数据集是一个类别就不再分裂
                            self.leafnodes.append('%sl' % hd)  # 成叶子节点
                        else:
                            copy_dict['%sl' % hd] = fenguo[0]  # 继续分裂

                        self.node_shujuji['%sl' % hd] = fenguo[0]  # 总的数据集

                        # 添加节点的规则
                        self.node_rule['%sl' % hd] = (self.node_rule[hd]).copy()
                        self.node_rule['%sl' % hd].append(fenguo[2])

                        if len(set(fenguo[1][:, -2])) == 1:
                            self.leafnodes.append('%sr' % hd)
                        else:
                            copy_dict['%sr' % hd] = fenguo[1]

                        self.node_shujuji['%sr' % hd] = fenguo[1]

                        # 添加节点的规则
                        self.node_rule['%sr' % hd] = (self.node_rule[hd]).copy()
                        self.node_rule['%sr' % hd].append(fenguo[2])

                        # 添加到节点关系字典
                        self.noderela[hd] = ['%sl' % hd, '%sr' % hd]

                    del copy_dict[hd]  # 需要在分裂数据中删除这一个

            self.fenlei_shujuji = copy.deepcopy(copy_dict)

            print('所有节点的个数：', len(self.fenlei_shujuji))
            print('需要分裂的数据集的个数：', len(self.node_shujuji))

        return 'done'

    # 根据树得出每一个节点数据集的结果
    def jieguo_tree(self):
        # 根据每一个数据得到每一个节点对应的结果
        shujuji_jieguo = {}
        for shuju in self.node_shujuji:
            zuihang = self.node_shujuji[shuju][:, -2]
            # 选择最多的
            duodict = {ik: list(zuihang).count(ik) for ik in set(list(zuihang))}
            # 在其中选择最多的
            shujuji_jieguo[shuju] = max(duodict.items(), key=lambda dw: dw[1])[0]

        return shujuji_jieguo

    # 要得到叶子节点的集合
    def leafnodes_tree(self):
        # 不在键值中的所有节点
        keynodes = list(self.noderela.keys())
        zhin = list(self.noderela.values())
        zhinodes = []
        for hhu in zhin:
            for fff in hhu:
                zhinodes.append(fff)
        leafnodes = [jj for jj in zhinodes if jj not in keynodes]
        return leafnodes

    # 寻找任何一个内部节点的叶子节点
    def iner_leaf(self, exnode):
        # 内部节点
        inernodes = list(self.noderela.keys())
        # 叶子节点
        llnodes = []
        # 全部的节点
        ghunodes = list(self.noderela.values())

        gugu = []

        for hhdd in ghunodes:
            for ghgh in hhdd:
                gugu.append(ghgh)

        for jj in gugu + ['0']:
            if jj not in inernodes:
                if len(jj) > len(exnode) and exnode in jj:
                    llnodes.append(jj)
        return llnodes

    # 寻找任何一个内部节点的下属的节点
    def xiashu_leaf(self, exnode):
        # 叶子节点
        xiashunodes = []
        # 全部的节点
        godes = list(self.noderela.values())

        gug = []

        for hhdd in godes:
            for ghgh in hhdd:
                gug.append(ghgh)

        for jj in gug + ['0']:
            if exnode in jj:
                xiashunodes.append(jj)
        return xiashunodes

    # 判读数据是否符合这个规矩的函数
    def judge_data(self, data, signstr, guize):
        # 首先判断数据连续或者是离散
        fign = 0
        try:
            data[guize[0]] + 2
            fign = 1
        except TypeError:
            pass
        if fign == 1:  # 连续
            if signstr == 'r':
                if data[guize[0]] > guize[1]:
                    return True
                return False
            elif signstr == 'l':
                if data[guize[0]] <= guize[1]:
                    return True
                return False
        elif fign == 0:  # 离散
            if signstr == 'r':
                if data[guize[0]] != guize[1]:
                    return True
                return False
            elif signstr == 'l':
                if data[guize[0]] == guize[1]:
                    return True
                return False

    # 预测函数, 根据节点的关系字典以及规则、每个节点的结果获得预测数据的结果
    def pre_tree(self, predata, sign='t'):
        # 每个数据集合的结果
        meire = self.jieguo_tree()
        # 存储结果
        savresu = []
        # 首先根据节点关系找到所有的叶子节点
        yezinodes = self.leafnodes_tree()
        # 开始判断数据
        for jj in predata:
            #  注意训练数据比测试、预测数据多个特征
            if sign == 't':
                shuju = jj[: -2]
            else:
                shuju = jj[: -1]
            # 开始判断
            for yy in yezinodes:
                gu = 1
                guide = self.node_rule[yy]
                for iu, ju in zip(yy[1:], guide):
                    if not self.judge_data(shuju, iu, ju):
                        gu = 0
                        break
                if gu == 1:
                    savresu.append(meire[yy])
        return savresu

    # 计算每一个节点的剪枝的基尼系数
    def jianzhi_iner(self, exnode):
        # 首先得到整体训练数据集的长度
        leng = len(self.train_dtdata)
        # # 在得到本节点数据集的长度,此项可以被消去
        # benleng = len(self.node_shujuji[exnode])

        # 计算被错误分类的数据的条数
        self.node_result = self.jieguo_tree()
        cuowu_leng = len(self.node_shujuji[exnode][self.node_shujuji[exnode][:, -2] != self.node_result[exnode]])
        # 计算
        jinum = cuowu_leng / leng
        return jinum

    # 计算每一个内部节点的下属叶子节点的基尼系数之和
    def iner_sum(self, ecnode):
        jnum = 0
        # 首先得到这个内部节点下属的所有叶子节点
        for hhh in self.iner_leaf(ecnode):
            jnum += self.jianzhi_iner(hhh)
        return jnum

    # 树的剪枝， 每一棵树都是一个字典形式（节点关系就代表一棵子树）
    def prue_tree(self):
        # 开始剪枝
        tree_set = {}
        # a值的字典
        adict = {}

        # 第一棵完全生长的树
        sign = 0
        tree_set[sign] = self.noderela.copy()
        # 开始剪枝
        while len(list(self.noderela.keys())) != 0:
            # 复制字典
            coppdict = self.noderela.copy()
            # 存储内部节点剪枝基尼系数的字典
            saveiner = {}
            for jiner in list(self.noderela.keys()):
                # 每一个内部节点计算
                saveiner[jiner] = (self.jianzhi_iner(jiner) - self.iner_sum(jiner)) / (len(self.iner_leaf(jiner)) - 1)
            # 选择其中最小的，如果有2个相同的选择最长的
            numm = np.inf
            dd = ''
            for hji in saveiner:
                if numm > saveiner[hji]:
                    dd = hji
                    numm = saveiner[hji]
                elif numm == saveiner[hji]:
                    if len(dd) < len(hji):
                        dd = hji
            # 添加到a值
            adict[sign] = numm
            # 需要删除hji这个内部节点
            # 首选得到这个内部节点所有的
            for hco in self.xiashu_leaf(dd):
                if hco in coppdict:
                    del coppdict[hco]
            # 树加1
            sign += 1
            self.noderela = coppdict.copy()
            tree_set[sign] = self.noderela.copy()
        return tree_set, adict

    # 计算正确率的函数
    def compuer_correct(self, exli_real, exli_pre):
        if len(exli_pre) == 0:
            return 0
        else:
            corr = np.array(exli_pre)[np.array(exli_pre) == np.array(exli_real)]
            return len(corr) / len(exli_pre)

    # 交叉验证函数
    def jiaocha_tree(self, treeset):  # 输出最终的树
        # 正确率的字典
        correct = {}

        # 遍历树的集合
        for jj in treeset:
            self.noderela = treeset[jj]
            yuce = self.pre_tree(self.test_dtdata, 'e')
            # 真实的预测值
            real = self.test_dtdata[:, -1]
            # 计算正确率
            correct[jj] = self.compuer_correct(real, yuce)

        # 获得最大的，如果有相同的，获取数目最小的键
        num = 0
        leys = ''
        for jj in correct:
            if correct[jj] > num:
                num = correct[jj]
                leys = jj
            elif num == correct[jj]:
                try:
                    if jj < leys:
                        leys = jj
                except TypeError:
                    leys = jj

        if self.sign == 'f':
            return treeset[leys], num
        else:
            # 获取最终的树后，更改样本的权重以及获取该模型的权重
            self.noderela = treeset[leys]
            train_yuce = self.pre_tree(self.train_dtdata)
            # 真实的预测值
            train_real = self.train_dtdata[:, -2]
            # 计算错误率
            error = 1 - self.compuer_correct(train_real, train_yuce)

            if 0 < error < 1:
                # 计算权重
                model_weight = 0.5 * log((1 - error) / error, 10)
            elif error == 1:
                model_weight = 0.5 * log((1 - 0.999) / 0.999, 10)
            else:
                model_weight = 0.5 * log((1 - 0.001) / 0.001, 10)
            # 更改样本的权重
            change_weight = self.weight_list * np.exp(-model_weight *
                                                      np.array(np.multiply(train_real, train_yuce), dtype=np.float))
            self.weight_list = change_weight / np.sum(change_weight)

            return treeset[leys], num, model_weight

    def noisyCount(self, sensitivety=1):
        beta = sensitivety / self.e
        n_values = []
        for i in range(100):
            u1 = np.random.random()
            u2 = np.random.random()
            if u1 <= 0.5:
                n_value = -beta * np.log(1. - u2)
            else:
                n_value = beta * np.log(u2)
            n_values.append(n_value)
        return np.mean(n_values)

    def laplace_mech(self, data):
        for i in range(len(data)):
            data[i] += self.noisyCount()
        return data

    def get_tp_fp_fn_tn(self, pre_type, real_type):
        # 真正
        tp = list(real_type[pre_type == real_type]).count(1)
        # 真负
        tn = list(real_type[pre_type == real_type]).count(-1)
        # 假正
        all_d = pre_type + real_type
        fp = list(all_d[real_type == -1]).count(0)
        # 假负
        fn = list(all_d[real_type == 1]).count(0)

        return tp, tn, fp, fn

    def get_f1score(self, pre_type, real_type):
        tp, tn, fp, fn = self.get_tp_fp_fn_tn(pre_type, real_type)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1 = 2 * precision * recall / (precision + recall)

        return f1, precision, recall

    def get_fpr_tpr(self, value_count, type_real, type_prediict):
        fpr_tpr = []
        for i in np.linspace(-1, 1, value_count):
            bu = np.where(type_prediict < i, -type_prediict / type_prediict, type_prediict)
            last_bu = np.where(bu >= i, bu / bu, bu)

            tp, tn, fp, fn = self.get_tp_fp_fn_tn(last_bu, type_real)

            fpr = fp / (fp + tn)

            tpr = tp / (tp + fn)

            fpr_tpr.append([fpr, tpr])

        # 将FPR从的值小到大排列，组成X轴，对应的TPR为Y轴，构成ROC曲线
        sort_gu = sorted(fpr_tpr, key=lambda x: x[0] * 10 + x[1])  # 保证FPR相同的，TPR大的在后面

        # 返回绘图需要的点的xy。
        plot_x = [h[0] for h in sort_gu]
        plot_y = [h[1] for h in sort_gu]

        if plot_x[0] != 0:
            plot_x.insert(0, 0)
            plot_y.insert(0, plot_y[0])

        if plot_y[1] != 1:
            plot_x.insert(-1, 1)
            plot_y.insert(-1, plot_y[-1])

        # 计算面积
        area = 0
        # 梯形法计算面积
        for index, value in enumerate(plot_x[1:]):
            area += (plot_x[index + 1] - plot_x[index]) * (plot_y[index + 1] + plot_y[index]) / 2

        return [plot_x, plot_y], area

    def AdaBoost_adult(self):
        # 记录每一个模型的结果
        model_result_dict = {}
        model_result_dict2 = {}
        for i in range(self.model_c):
            model_result_dict[i] = {}
            model_result_dict2[i] = {}
            # 完全成长的树
            self.grow_tree()
            # 剪枝形成的树的集
            gu = self.prue_tree()
            # 交叉验证形成的最好的树
            cc = self.jiaocha_tree(gu[0])
            # 根据最好的树预测新的数据集的结果
            self.noderela = cc[0]

            prenum = self.pre_tree(self.pre_dtdata, 'e')


            if self.sign == 'e':  # 添加噪声
                prenum = self.laplace_mech(prenum)


            elif self.sign == 'f':  # CART，直接返回结果
                # 计算正确率
                pre_correct = self.compuer_correct(prenum, self.pre_dtdata[:, -1])

                return pre_correct
            # Ada保存模型权重以及模型的预测结果
            model_result_dict[i] = {'m_weight': cc[2], 'predict_labe': np.array(prenum, dtype=np.float)}

            if cc[1] == 1:
                break

            # 初始化树
            # 中间过程变量
            self.node_shujuji = {'0': self.train_dtdata}  # 存储每一个节点数据集的字典
            self.fenlei_shujuji = {'0': self.train_dtdata}  # 存储需要分类的数据集的字典

            # 叶子节点的集合
            self.leafnodes = []
            # 节点关系字典
            self.noderela = {}
            # 每一个节点的规则关系
            self.node_rule = {'0': []}

        # 最终结果
        last_result = np.array([0 for i in range(len(self.pre_dtdata))], dtype=np.float)
        for m in model_result_dict:
            last_result += model_result_dict[m]['m_weight'] * model_result_dict[m]['predict_labe']

        # 最终的预测结果
        last_result = np.where(last_result == 0, last_result / last_result, last_result)
        last_predict_result = np.sign(last_result)

        # 计算预测样本的正确率
        # pre_correct = self.compuer_correct(last_predict_result, self.pre_dtdata[:, -1])

        # # 记录每个模型的ROC曲线的数据
        # plot_xy, area = self.get_fpr_tpr(40, self.pre_dtdata[:, -1], last_result)

        # 计算F1分数
        f1, pre, sca = self.get_f1score(last_predict_result, self.pre_dtdata[:, -1])

        return f1, pre, sca


# 树的深度列表
tree_depth = [2, 3, 4, 5, 6]
T = 10
e = 1
# 记录测试样本的f1,精确率，召回率
f1_data_dict = OrderedDict()
pre_data_dict = OrderedDict()
sca_data_dict = OrderedDict()

Tree_depth_list = [2, 3, 4, 5, 6]

for e in [0.10, 0.25, 0.50, 0.75, 1.00]:
    print('当前e=', e)
    f1_data_dict[e] = []
    pre_data_dict[e] = []
    sca_data_dict[e] = []
    for t_d in Tree_depth_list:
        print('当前树的深度', t_d)
        model = CartAdaBoostAdult(t_d, 10, e, sign='e')

        f1, pre, sca = model.AdaBoost_adult()

        f1_data_dict[e].append(f1)
        pre_data_dict[e].append(pre)
        sca_data_dict[e].append(sca)

        print(f1, pre, sca)

print(f1_data_dict)
print(pre_data_dict)
print(sca_data_dict)

import matplotlib.pyplot as plt
from pylab import mpl  # 作图显示中文
mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


def plot_dict(data_dict, title):
    plt.figure(figsize=(7, 4.5), dpi=80)
    for k, m, l in zip(data_dict, ['s', 'o', '^', 'v', 'D'], ['dotted', 'dashed',
                                                              'dashdot', (0, (3, 1, 1, 1)), 'solid']):

        print(k, m, l)

        plt.plot(Tree_depth_list, data_dict[k], marker=m, linestyle=l, label=r'$\epsilon=%.2f$' % k)

    plt.xlabel('树深度 d')
    plt.ylabel('%s' % title)

    plt.xticks(Tree_depth_list)

    plt.yticks(np.linspace(0, 1, 6))

    plt.legend(loc='upper left')

    plt.savefig('%s.png' % title)


# F1
plot_dict(f1_data_dict, 'f1')
# 精确率
plot_dict(pre_data_dict, 'Presicion')
# 召回率
plot_dict(sca_data_dict, 'Recall')























