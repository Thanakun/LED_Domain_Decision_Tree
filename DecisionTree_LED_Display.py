import numpy as np
from collections import Counter
import pandas as pd
import numpy as np
from numpy import log2
from sklearn.model_selection import train_test_split
from sklearn import tree


data = pd.read_csv("output.csv") # load data จากไฟล์ output.csv โดยใช้ pandas
class_list_data = data['target'].unique() # ดูว่า class ของเรามีเลขใดบ้างโดยใช้ unique() เพื่อมาหา class
features = data.columns[:-1] #เอา features ในแต่ละหัวข้อมา

print(len(data['one']))

def calc_entropy(data, target_column): 
    """
    หา entropy จาก column ที่เราอยากดูในที่นี้คือ
        [one,two,three,four,five,six,seven]
    เพื่อที่จะดูค่าของ entropy เพื่อมาหา root node หลังจาก ฟังก์ชัน gain
    input: dataset(data pandas) , target_column(column ที่สนใจ ex. one,two,three etc.)
    output: float of entropy ของ attriubute ที่อยากได้
    """
    entropy = 0 # กำหนด entropy
    values = data[target_column].unique() # getting the attribue [0,1] หากเป็น target [0,1,2,3,4,5,6,7,8,9]
    for value in values:
        """
        fraction คือ ตัวแปรที่จะเอาค่าของจำนวนของเลขนั้น หารด้วย ขนาดของ target_column
        data['one'] เอา column one
        data['one'].value_counts() จะเข้าไปนับว่า 0 และ 1 ใน column นั้นมีกี่ตัว
        [value] = คือเอาค่าที่เราอยากได้ [0] คือเอาจำนวนของ 0 , [1] คือเอาจำนวนของ 1 
        หารด้วย ขนาดของ column ที่เราสนใจ
        entropy += (จำนวนเลข 0 หารด้วย จำนวนทั้งหมดของ column คูณด้วย log2(จำนวนเลข 0 หารด้วย จำนวนทั้งหมดของ column ))
        """
        fraction = data[target_column].value_counts()[value]/len(data[target_column]) 
        entropy += -fraction * np.log2(fraction)
    return entropy




def calc_information_gain(data, feature, target_column):
    """
    คำนวณค่า information_gain ที่ได้จาก entropy โดยที่เราจะคำนวณดูว่า entropy 
    ของ feature ที่เราสนใจมีอะไรได้บ้าง โดยจะดูจาก column ของ target class ที่เราต้องการ
    input : data (data frame pandas ) , (column ของ attribute) , target column (column ของ class)
    output : infomation_gain คือค่าของ gain ในแต่ละ attribute ของ column
    """
    total_entropy = calc_entropy(data, target_column) # ค่า entropy ทั้งหมด หาได้จาก entropy ของ class
    values = data[feature].unique() # เอาค่าของ values ที่เป็นไปได้ คือ [0,1]
    feature_entropy = 0 # กำหนดค่า feature_entropy = 0
    for value in values:
        """
        data[feature] == value คือ หาค่า feature ที่เราสนใจเช่น
        data['one'] ที่ช่องนั้นมีค่าเท่ากับ value ที่เป็นไปได้คือ 0 หรือ 1
        len(data[data[feature] == value]) คือ เมื่อเราได้ช่องนั้นแล้วและค่าที่เป็นของเราแล้ว เราก็ดูจำนวนของข้อมูลนั้น  หารด้วย จำนวนข้อมูล
        """
        fraction = len(data[data[feature] == value])/len(data)
        sub_data = data[data[feature] == value]
        feature_entropy += fraction * calc_entropy(sub_data, target_column) # ใช้สูตรได้ info(attribute)
    information_gain = total_entropy - feature_entropy #(หา gain)
    return information_gain



def find_most_information_feature(data, label, class_list):
    """
    เป็น ฟังก์ชัน information_gain เพื่อหา attribute ที่จะได้ค่า information gain จาก attribute ที่มีค่า information gain มากที่สุด
    """
    feature_information_gain_dict = {}
    for feature in list(data.columns[:-1]): # วิ่งดูทุกๆ feature ที่มีอยู่
        feature_information_gain = calc_information_gain(data, feature, label) # calculate information gain of the feature
        feature_information_gain_dict[feature] = feature_information_gain
        # print("feature_information_gain_dict : ",feature_information_gain_dict)
    max_info_feature = max(feature_information_gain_dict, key = feature_information_gain_dict.get) # get the feature with the maximum information gain
    return max_info_feature


def ID3(data, features, target, parent_node_class = None):
    """
    algorithm ID3 เอาไว้สร้าง tree ของ root node และ child node และ วนรูปแบบ recursive จนเป็นต้นไม้ไปเรื่อยๆ
    ในการหยุด recursive นีจะมีกรณีที่ต้นไม้จะหยุดสร้าง 3 อย่าง
    Base case: 1.) ขนาดของ unique ของค่าใน class (target) มีจำนวนน้อยกว่าเท่ากับ 1
                    return ค่าของ class นั้นออกมา
               2.) ถ้าขนาดของ data มีเท่ากับ 0
                    return None
               3.) ถ้าขนาดของ feature มีค่าเท่ากับ 0
                    return parent_node_class 
    Case ที่เราจะได้ต้นไม้ :
    parent_node_class = ค่า list unique ของ class
    [โดยที่ตำแหน่งของค่าที่มากที่สุด(ของค่า เฉพาะที่เป็นเอกลักษณ์ใน target,ค่าของแต่ละตัวโผล่ออกมา)[เอาค่าที่มากที่สุดออกมา]]
    item_values = [คำนวณหา information_gain ใน feature แต่ละตัว]
    best_feature_index = นำตำแหน่ง gain สูงสุดออกมาจาก item_values
    best_feature = นำค่า gain ออกมาจาก item_values
    tree = โดยจะสร้าง tree จาก key:values mapping จาก best features
    features = เอา best_features ออกจาก features ของเรา

    วนรูปค่า value เอาค่า attribute(best feature) คือ (0,1) 
        sub_data = สร้าง subdata จาก data ที่ฝั่งที่เป็น 0 หรือ 1 จากต้นไม้ของเรา
        subtree = เรียก ฟังก์ชันอีกรอบหนึ่งเพื่อสร้าง node ไปเรื่อยๆ
        กำหนดให้ tree มีลูกอยู่ตำแหน่งเลขใดเลขใดบ้าง 
    return tree


    input : (data(dataframe pandas ), features (attribute) , target (class attribute), parent_node_class(default=None) )
    output : tree (ต้นไม้)  
    """
    if len(np.unique(data[target])) <= 1:
        return np.unique(data[target])[0]
    elif len(data) == 0:
        return np.unique(parent_node_class)[0]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target])[np.argmax(np.unique(data[target], return_counts=True)[1])]
        item_values = [calc_information_gain(data, feature, target) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, features, target, parent_node_class)
            tree[best_feature][value] = subtree
        return tree




def predict(tree, data, label):

    """
    ทำนายค่าที่เราได้จาก tree และ data ที่เราได้มาว่าได้ค่าเท่าใด
    input : tree , data(data frame pandas) , (target label)
    output predictions
    """
    predictions = [] # array ของ predicitons
    for i in range(len(data)): # วนรูปของ data ที่นำเข้ามา
        row = data.iloc[i] # วิ่งที่ row แต่ละ row ของ data 
        current_node = tree # อยู่บน current_node
        while type(current_node) == dict: #ถ้า tree เป็น รูปแบบเป็น mapping key to value
            feature = list(current_node.keys())[0] # feature คือ list ของ key to value ที่ตำแหน่งที่ 0
            feature_value = row[feature] #นำค่า row ของ feature_value
            if feature_value not in current_node[feature]: #ถ้ามันไม่ได้อยู่ใน tree 
                current_node = "?"
                break
            else:
                current_node = current_node[feature][feature_value] #เข้าไปดูใน current node ว่ามีค่าที่วิ่งไปที่ node อีกหรือไม่
        predictions.append(current_node) #จะนำเข้าไปอยู่ใน prediction
    return predictions

def calculate_accuracy(predictions, actual):
    """
    คำนวณความแม่นยำ จาก predictions ที่เราได้มา
    input: predictions(dictionary)
    output: ค่าที่จะได้จาก data target (float)
    """
    correct_predictions = 0 # จำที่ predictions ถูก
    for i in range(len(predictions)): # วนรูปวิ่งตาม prediction
        if predictions[i] == actual[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(predictions)
    print("prediction_size : ",len(predictions))
    print("correct_predictions : ", correct_predictions)
    return accuracy


def classification_rule(tree, feature_names, rule):
    """
    return rules ของต้นไม้ที่จะได้ออกมา
    input : tree , attribute ทั้งหมด ยกเว้น target , rule => default
    output : print rules ที่เราได้
    """
    if isinstance(tree, dict):
        for key in tree.keys():
            if isinstance(key, int):
                feature = feature_names[key]
                threshold = tree[key][0]
                sub_tree = tree[key][1]
                sub_rule = f"{rule} AND {feature} <= {threshold}"
                classification_rule(sub_tree, feature_names, sub_rule)
            else:
                sub_tree = tree[key]
                sub_rule = f"{rule} AND {key}"
                classification_rule(sub_tree, feature_names, sub_rule)
    else:
        print(f"{rule} THEN class = {tree}")

def rule_input(tree, input_data, feature_names):
    if not isinstance(tree, dict): # if the node is a leaf node
        return " THEN class = " + str(tree)
    else:
        for feature_value in tree:
            if isinstance(feature_value, int): # check if the feature value is an integer
                feature_index = feature_names.index(str(feature_value))
                if input_data[feature_index] == 1: # check if the feature value in the input data is 1
                    return " AND " + feature_names[feature_index] + " = 1" + rule_input(tree[feature_value], input_data, feature_names)
            elif "<=" in str(feature_value):
                feature_index = int(feature_value.split("<=")[0])
                threshold = float(feature_value.split("<=")[1])
                if input_data[feature_index] <= threshold:
                    return " AND " + feature_names[feature_index] + " <= " + str(threshold) + rule_input(tree[feature_value], input_data, feature_names)
    return ""



# test_data_new = [1,1,0,1,1,1,1]
print("entropy Segment 1 : ",calc_entropy(data,'one'))
print("entropy Segment 2 : ",calc_entropy(data,'two'))
print("entropy Segment 3 : ",calc_entropy(data,'three'))
print("entropy Segment 4 : ",calc_entropy(data,'four'))
print("entropy Segment 5 : ",calc_entropy(data,'five'))
print("entropy Segment 6 : ",calc_entropy(data,'six'))
print("entropy Segment 7 : ",calc_entropy(data,'seven'))
print("entropy Segment target ",calc_entropy(data,'target'))

print("finding the information gain each attribute : \n")
print("information gain of one :",calc_information_gain(data,'one','target'))
print("information gain of two :",calc_information_gain(data,'two','target'))
print("information gain of three :",calc_information_gain(data,'three','target'))
print("information gain of four :",calc_information_gain(data,'four','target'))
print("information gain of five :",calc_information_gain(data,'five','target'))
print("information gain of six :",calc_information_gain(data,'six','target'))
print("information gain of seven :",calc_information_gain(data,'seven','target'))


print("the most information gain is :" , find_most_information_feature(data,'target',class_list_data))
#split train and test_data
train_data, test_data = train_test_split(data, test_size=0.1)

# most = find_most_informative_feature(data,'target',class_list_data)
# print(most)
#getting feature name of
features = data.columns[:-1]
print("Features of data : ",features)
target = data.columns[-1]
print("Target column : ", target)
tree = ID3(data, features, target)

print("Train Data: \n",train_data)
print("Test Data: \n",test_data)
# tree = make_tree(most,None,train_data,'target',class_list_data)


print(tree)

print("calc_information_gain(data,'one','target')",calc_information_gain(data,'one','target'))


A = data.columns.values.tolist()
print("A = ",A)
predictions = predict(tree,test_data,target)
print("Prediction_Values: ", predictions)
accuracy = calculate_accuracy(predictions, test_data['target'].values)
print("Accuracy: ", accuracy)
# print(tree)
feature_names = list(train_data.columns[:-1])
new_rule = ""
rules = classification_rule(tree,A,"IF")
# print(rules)

input_data = [0,0,1,0,1,0,0,1]
rule = "IF" + rule_input(tree, input_data, A)
print("Rule: ", rule)

