import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error  #均方误差
from sklearn.metrics import r2_score  #R方
from sklearn.metrics import explained_variance_score  #可解释方差值
from sklearn.metrics import mean_absolute_error  #平均绝对误差
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import warnings 
warnings.filterwarnings("ignore")




def init():
    global _data
    if os.path.exists("data.csv"):
        _data=pd.read_csv("data.csv",encoding="utf-8",header=None)
        
def get_0():
    init()
    return tuple(_data[0])
        
def _save_data():
    global _data
    with open("data.csv","w",encoding="utf-8") as f:
        _data.to_csv("data.csv",encoding="utf_8_sig",index=False,header=False)
    f.close()
    
def get_all_data():
    #获取所有数据组和数据
    global _data
    return _data

def get_data(data_name):
    #获取某个数据组的数据
    global _data
    datalist=np.array(_data[_data[0]==data_name]).tolist()[0][1:]
    return [a for a in datalist if a==a] #去除nan值

def add_data(data_name,data_data):  #data_data为逗号分割的str
    #添加数据组
    global _data
    data_data=[float(i) for i in data_data.split(" ")]
    data_data.insert(0,data_name)
    df=pd.DataFrame(data_data).T
    _data=pd.concat([_data,df],axis=0,ignore_index=True)
    _save_data()

def remove_data(data_name):
    #删除数据组
    global _data
    _data.drop(_data.index[(_data[0]==data_name)],inplace=True)
    _save_data()

def add_number(data_name,number):
    #添加最新一次实验的数据
    global _data
    new_datalist=np.array(_data[_data[0]==data_name]).tolist()[0]
    new_datalist.append(float(number))  
    new_df=pd.DataFrame(new_datalist).T
    remove_data(data_name)
    _data=pd.concat([_data,new_df],axis=0,ignore_index=True)
    _save_data()

def remove_number(data_name,c):
    #删除第c次的实验数据
    global _data
    for i in _data[_data[0]==data_name].T.isna().sum():
        _data.loc[(_data[0]==data_name),float(c)+i]=np.nan
    _save_data()
    
def change_number(data_name,c,number):
    #将第c次的实验数据更改为number
    global _data
    _data.loc[(_data[0]==data_name),int(c)]=float(number)
    _save_data()


def cal_formula(c,formula,new_result): #通过内部input获取变量
    #通过公式，代入实验数据，保存计算结果
    c=c.split(sep=" ")
    #formula = input('请输入公式（支持加减乘除、幂、括号运算）：')
    #new_result=input("请输入结果变量名")
    ##准备基本函数
    def mul_div(formula):
        #计算乘除法
        if '*' in formula:
            a, b = formula.split('*')
            return float(a)*float(b)
        if '/' in formula:
            a, b = formula.split('/')
            return float(a) / float(b)
    def cf_power(formula):
        #计算乘方
        if '^' in formula:
            a, b = formula.split('^')
            return pow(float(a),float(b))
    def pack_formula(formula):
        #解决加减号和正负号冲突
        while re.search('[+-]{2,}',formula):
            formula = formula.replace('--','+').replace('+-','-').replace('-+','-').replace('++','+')
        return formula
    def remove_addsub(formula):
        #计算所有加减法
        res_formula = re.findall('[-+]?\d+(?:\.\d+)?',formula)
        if len(res_formula) <= 0:
            return 0
        else:
            _sum = float(res_formula[0])
            for i in res_formula[1:]:
                _sum += float(i)
            return _sum
    def remove_muldiv(formula):
        #计算所有乘除法
        while True:
            res_formula = re.search('\d+(\.\d+)?[*/]-?\d+(\.\d+)?',formula)
            if res_formula:
                md_formula = res_formula.group()
                result_formula = mul_div(md_formula)
                formula = formula.replace(md_formula,str(result_formula))
            else:
                return formula
    def remove_power(formula):
        #计算所有乘方
        while True:
            res_formula = re.search('\d+(\.\d+)?\^-?\d+(\.\d+)?',formula)
            if res_formula:
                cf_formula = res_formula.group()
                result_formula = cf_power(cf_formula)
                formula = formula.replace(cf_formula,str(result_formula))
            else:
                return formula
    def cal(formula):
        #无括号计算
        res_formula = remove_power(formula)
        res_formula = pack_formula(res_formula)
        res_formula = remove_muldiv(res_formula)
        res_formula = pack_formula(res_formula)
        res_formula = remove_addsub(res_formula)
        return res_formula
    def main(formula):
        #代入数字后的公式计算
        formula = formula.replace(' ','')
        formula = formula.replace('（',"(").replace('）',")")
        while True:
            res_formula = re.search('\([^()]+\)', formula)       #计算括号内的式子
            if res_formula:
                result_formula = cal(res_formula.group())
                formula = formula.replace(res_formula.group(),str(result_formula))
            else:
                return cal(formula)
    ##cal_formula主代码
    l=[]
    for j in range(len(list(map(float,get_data(c[0]))))):
        formula_r=formula
        for i in c:
            formula_r=formula_r.replace(i,str(get_data(i)[j]))
        l.append(round(main(formula_r)))
  
    l_str = ""
    for i in l:
        
        l_str +=str(i)+" "
        
    add_data(new_result,l_str[0:-1])
    output=new_result+":"
    for i in l:
        output+=str(i)+","
    out=''
    for i in range(len(c)):
        LL = list(map(float,get_data(c[i])))
        out += str(c[i])+':'
        for j in LL:
            out+=str(j)+","
        out+="\n"
    out+=output
    return out#打印到弹窗


def draw_scatter(a,b):
    #绘制实验数据散点图，描述两个变量之间的关系
    global _data
    a_data = np.array(list(map(float,get_data(a))))
    
    
    b_data = np.array(list(map(float,get_data(b))))
    
    plt.figure(dpi=100,figsize=(5,3))
    plt.scatter(a_data,b_data,marker='o')
    plt.xlabel(a)
    plt.ylabel(b)
    for p,q in zip(a_data,b_data):
        plt.text(p,q,(p, q), ha='left', va='bottom', fontsize=9)
    plt.title('{}与{}的变化关系'.format(a,b),fontsize='xx-large',fontweight='heavy')
    plt.savefig('变化关系.png',bbox_inches = 'tight')  #图片名称
    


'''
    #pyecharts绘制
    from pyecharts.charts import Scatter
    from pyecharts import options as opt
    figsise = opt.InitOpts(animation_opts=opt.AnimationOpts(animation_delay=1000, animation_easing="elasticOut"),
                           width="800px", height="600px")
    scatter = Scatter(init_opts=figsise)
    x_data = np.array(list(map(float,get_data(a))))
    y_data = np.array(list(map(float,get_data(b))))
    scatter.add_xaxis(xaxis_data=x_data)
    scatter.add_yaxis(series_name=b,  #名称
                          y_axis=y_data,  # 数据
                          label_opts=opt.LabelOpts(is_show=False),  # 数据不显示
                          symbol_size=15,  # 设置散点的大小
                          symbol="circle"  # 设置散点的形状
                          )
    scatter.set_global_opts(
    xaxis_opts=opt.AxisOpts(name=a,type_='value'),
    yaxis_opts=opt.AxisOpts(name=b,type_='value'),
    title_opts=opt.TitleOpts(title='{}与{}之间的变化关系'.format(a,b)))
    scatter.render('{}与{}之间的变化关系.html'.format(a,b))
'''

    
def special(s,data_data): #s下拉框选择，data_data用文本框,提示空格分隔
    #特征值计算
    data_data = data_data.split(sep=' ')
    df = _data[_data[0]==data_data[0]].dropna(axis=1,how='any')
    for i in data_data[1:]:
        x_data = _data.loc[_data[0].isin([i])].dropna(axis=1,how='any')
        df = pd.concat([df,x_data])
    all_index = list(df[0])
    df.index = all_index
    df = df.drop([0],axis=1).T
    df = df.applymap(float)
    data_data = pd.DataFrame(data_data)
    if s == '平均值':
        data_spec = pd.DataFrame(df.mean(axis=0).values)
        result_s = pd.concat([data_data,data_spec],axis=1)
        result_s = '\n'.join(result_s.to_string(index = False).split('\n')[1:])
    if s == '方差':
        data_spec = pd.DataFrame(df.var(axis=0).values)
        result_s = pd.concat([data_data,data_spec],axis=1)
        result_s = '\n'.join(result_s.to_string(index = False).split('\n')[1:])
    if s == '标准差':
        data_spec = pd.DataFrame(df.std(axis=0).values)
        result_s = pd.concat([data_data,data_spec],axis=1)
        result_s = '\n'.join(result_s.to_string(index = False).split('\n')[1:])
    if s == '样本偏度（三阶矩）':
        data_spec = pd.DataFrame(df.skew(axis=0).values)
        result_s = pd.concat([data_data,data_spec],axis=1)
        result_s = '\n'.join(result_s.to_string(index = False).split('\n')[1:])
    if s == '样本峰度（四阶矩）':
        data_spec = pd.DataFrame(df.kurt(axis=0).values)
        result_s = pd.concat([data_data,data_spec],axis=1)
        result_s = '\n'.join(result_s.to_string(index = False).split('\n')[1:])
    if s == '协方差矩阵':
        result_s = df.cov()
    if s == '相关系数矩阵':
        result_s = df.corr()
    return result_s
    
def predict(i,x,y):    #模型拟合,i用下拉框，x用文本框输入,y用下拉框
    global _data
    
    #选择模型：单变量用多项式拟合，多变量用ols多元线性回归
    #多项式拟合
    if i=="多项式拟合":
        x_data = np.array(list(map(float,get_data(x))))
        y_data = np.array(list(map(float,get_data(y))))
        s = ""
        plt.figure(dpi=100,figsize=(4,2))
        for i in range(2,7):
            z = np.polyfit(x_data,y_data,i)
            r_pred = np.poly1d(z)
            y_pred = r_pred(x_data)
            plt.plot(x_data,y_pred,label=str(i)+'次多项式')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            a = mean_squared_error(y_data,y_pred)
            b = r2_score(y_data,y_pred)
            c = explained_variance_score(y_data,y_pred)
            d = mean_absolute_error(y_data,y_pred)
            s += '{}项式拟合模型的均方误差值为{}\n'.format(i,a) + '{}项式拟合模型的R2值为{}\n'.format(i,b)\
               +'{}项式拟合模型的可解释误差值为{}\n'.format(i,c) + '{}项式拟合模型的平均绝对误差值为{}\n'.format(i,d)
        
        plt.plot(x_data,y_data,"*",label='原始数据')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('{}与{}的原始曲线和拟合曲线比较'.format(x,y))

        plt.savefig('多项式拟合.png',bbox_inches = 'tight')
        return s
        
        
        
        
    
    if i=='插值拟合':
        #插值拟合
        x_data = np.array(list(map(float,get_data(x))))
        y_data = np.array(list(map(float,get_data(y))))
        plt.figure(dpi=100,figsize=(4,2))
        f1 = interp1d(x_data, y_data, kind='linear')      #分段线性
        f2 = interp1d(x_data, y_data, kind='nearest')     #最近邻点
        plt.plot(x_data,y_data,x_data,f1(y_data),x_data,f2(y_data))
        plt.legend(['origin','linear','nearest'], loc='best')
        plt.savefig('插值拟合.png',bbox_inches = 'tight')
        
        s=""
        a1 = mean_squared_error(y_data,f1(y_data))
        b1 = r2_score(y_data,f1(y_data))
        c1 = explained_variance_score(y_data,f1(y_data))
        d1 = mean_absolute_error(y_data,f1(y_data))
        a2 = mean_squared_error(y_data,f2(y_data))
        b2 = r2_score(y_data,f2(y_data))
        c2 = explained_variance_score(y_data,f2(y_data))
        d2 = mean_absolute_error(y_data,f2(y_data))
        s+= '分段线性拟合模型的均方误差值为{}\n'.format(a1) + '分段线性拟合模型的R2值为{}\n'.format(b1) +\
            '分段线性拟合模型的可解释误差值为{}\n'.format(c1) + '分段线性拟合模型的平均绝对误差值为{}\n'.format(d1)
        s+= '最近邻点拟合模型的均方误差值为{}\n'.format(a2) + '最近邻点拟合模型的R2值为{}\n'.format(b2) +\
            '最近邻点拟合模型的可解释误差值为{}\n'.format(c2) + '最近邻点拟合模型的平均绝对误差值为{}\n'.format(d2)
        return s
    
    
    #多维多项式回归
    if i == "多维多项式回归":
        x_=x.split(sep=' ')
        x_name=x.split(sep=' ')
        all_xy = x_
        all_xy.append(y)
        x_data = _data.loc[_data[0].isin(x_name)].dropna(axis=1,how='any').drop([0],axis=1)
        y_data = _data.loc[_data[0].isin([y])].dropna(axis=1,how='any').drop([0],axis=1)
        data = pd.concat([x_data,y_data])
        data.index = all_xy
        data = data.T
        data = data.applymap(float)
        x=data[x_name]
        y=data[y]
        x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state = 20)
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)
        Input = [('polynomial',PolynomialFeatures(degree=6)),('mode',LinearRegression())]
        pipe = Pipeline(Input)
        pipe.fit(x_train,y_train)
        y_pred= pipe.predict(x_test)
        a = mean_squared_error(y_test,y_pred)
        b = r2_score(y_test,y_pred)
        c = explained_variance_score(y_test,y_pred)
        d = mean_absolute_error(y_test,y_pred)
        s = '多维多项式回归的均方误差值为{}\n'.format(a) + '多维多项式回归的R2值为{}\n'.format(b) +\
            '多维多项式回归的可解释误差值为{}\n'.format(c) + '多维多项式回归的平均绝对误差值为{}\n'.format(d)
        return s
    
    
    #多元线性回归
    if i=='多元线性回归':
        x_=x.split(sep=' ')
        x_name=x.split(sep=' ')
        all_xy = x_
        all_xy.append(y)
        x_data = _data.loc[_data[0].isin(x_name)].dropna(axis=1,how='any').drop([0],axis=1)
        y_data = _data.loc[_data[0].isin([y])].dropna(axis=1,how='any').drop([0],axis=1)
        data = pd.concat([x_data,y_data])
        data.index = all_xy
        data = data.T
        data = data.applymap(float)
        x=data[x_name]
        y=data[y]
        x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state = 20)
        model = LinearRegression()
        model.fit(x_train, y_train)
        s = '多元线性回归模型的系数为{}\n'.format(model.coef_) + '多元线性回归模型的截距为{}\n'.format(model.intercept_)
        y_pred=model.predict(x_test)
        a = mean_squared_error(y_test,y_pred)
        b = r2_score(y_test,y_pred)
        c = explained_variance_score(y_test,y_pred)
        d = mean_absolute_error(y_test,y_pred)
        s += '多元线性回归模型的均方误差值为{}\n'.format(a) + '多元线性回归模型的R2值为{}\n'.format(b) +\
             '多元线性回归模型的可解释误差值为{}\n'.format(c) + '多元线性回归模型的平均绝对误差值为{}\n'.format(d)
        return s

    if i == 'OLS多元线性回归模型评估':
        #OLS多元线性回归
        x_name=x
        x_name=x_name.replace(" ","+")
        x_=x.split(sep=' ')
        all_xy = x.split(sep=' ')
        all_xy.append(y)
        x_data = _data.loc[_data[0].isin(x_)].dropna(axis=1,how='any').drop([0],axis=1)
        y_data = _data.loc[_data[0].isin([y])].dropna(axis=1,how='any').drop([0],axis=1)
        data = pd.concat([x_data,y_data])
        data.index = all_xy
        data = data.T
        data = data.applymap(float)
        train,test= train_test_split(data,test_size=0.2,random_state = 20)
        ols = sm.formula.ols('{}~{}'.format(y,x_name),data=data).fit()
        return ols.summary()   #模型总结
    
    if i == 'OLS多元线性回归模型异常值检测':
        # 异常值检测
        x_name=x
        x_name=x_name.replace(" ","+")
        x_=x.split(sep=' ')
        all_xy = x.split(sep=' ')
        all_xy.append(y)
        x_data = _data.loc[_data[0].isin(x_)].dropna(axis=1,how='any').drop([0],axis=1)
        y_data = _data.loc[_data[0].isin([y])].dropna(axis=1,how='any').drop([0],axis=1)
        data = pd.concat([x_data,y_data])
        data.index = all_xy
        data = data.T
        data = data.applymap(float)
        train,test= train_test_split(data,test_size=0.2,random_state = 20)
        ols = sm.formula.ols('{}~{}'.format(y,x_name),data=data).fit()
        outliers = ols.get_influence()   
        leverage = outliers.hat_matrix_diag    # 帽子矩阵
        dffits= outliers.dffits[0]    # dffits值
        resid_stu = outliers.resid_studentized_external   # 学生化残差
        cook = outliers.cooks_distance[0]   # cook距离
        # 合并各种异常值检验的统计量值
        contatl = pd.concat([pd.Series(leverage, name = 'leverage'),
                             pd.Series(dffits, name = 'dffits'),
                             pd.Series(resid_stu, name = 'resid_stu'),
                             pd.Series(cook, name = 'cook')
                            ],axis =1 )
        train.index = range(train.shape[0])
        profit_outliers = pd.concat([train,contatl],axis =1)
        # 计算异常值比例
        outliers_ratio = np.sum(np.where((np.abs(profit_outliers.resid_stu)>2),1,0))/profit_outliers.shape[0]
        return '异常值比例：'+str(outliers_ratio)

        
        '''
        none_outliers = profit_outliers.ix[np.abs(profit_outliers.resid_stu)<=2,]
        model = sm.formula.ols('{}~{}'.format(y,x.replace(" ","+")),data = none_outliers).fit()
        y_pred = model.predict(_data[x_data].dropna(axis=1,how='any'))
        print(f'OLS多元线性回归模型的均方误差值为{mean_squared_error(y_data,y_pred)}')
        print(f'OLS多元线性回归模型的R2值为{r2_score(y_data,y_pred)}')
        print(f'OLS多元线性回归模型的可解释误差值为{explained_variance_score(y_data,y_pred)}')
        print(f'OLS多元线性回归模型的平均绝对误差值为{mean_absolute_error(y_data,y_pred)}')
        plt.scatter(x_data,y_data,label="原始数据")
        plt.plot(x_data,y_pred,label='OLS线性回归')
        plt.legend()
        plt.show()
    '''
    
    
    

    
