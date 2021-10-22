from CART_AdaBoost_Adult import correct_dict, tree_depth_list, CartAdaBoostAdult, roc_correct_dict, area_correct_dict, \
    data_dict

for i in range(5):
    # DPsAdaBoost模型
    epsilon = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
    for e in epsilon:
        correct_dict['DPsAdaBoost(e=%s)' % str(e)] = []

        for tree_depth in tree_depth_list:

            DPsAdaBoost_model = CartAdaBoostAdult(tree_depth, 10, e, 'e')
            print('DPsAdaBoost：树的深度%d,基分类器个数%d,噪声参数%s'
                  % (DPsAdaBoost_model.tree_length, DPsAdaBoost_model.model_c, DPsAdaBoost_model.e))
            c, plotxt, area = DPsAdaBoost_model.AdaBoost_adult()
            if e in [0.1, 0.25, 0.5, 0.75, 1] and tree_depth == 5: # Census income最优树深度d=5.
                roc_correct_dict['DPsAdaBoost(e=%s)' % str(e)] = plotxt
                area_correct_dict['DPsAdaBoost(e=%s)' % str(e)] = area
            correct_dict['DPsAdaBoost(e=%s)' % str(e)].append(c)

            print('准确率: ', c)

    data_dict[i] = correct_dict
    data_dict['roc_%d' % i] = roc_correct_dict
    data_dict['area_%d' % i] = area_correct_dict

print(data_dict)