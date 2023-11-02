#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


get_ipython().system('pip install folium')

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import json
from folium.features import CustomIcon

path=os.getcwd()
case=pd.read_csv('data/Case.csv')
patinfo=pd.read_csv('data/PatientInfo.csv')
region=pd.read_csv('data/region.csv')

get_ipython().system('pip install pyvis')

get_ipython().system('pip install networkx')

import networkx as nx
from pyvis.network import Network

#집단 감염 케이스
infect_case=list(set(case['infection_case'])-set(['etc','contact with patient','overseas inflow']))
#결측치 채우기
patinfo.fillna('nan',inplace=True)
# 지역 확진번호 리스트
pat_id=list(patinfo['patient_id'])
patient_case=list(patinfo['infection_case'])
#누구로부터 감염됐는가
who=list(patinfo['infected_by'])
#결측치값을 제외, 중복제외
from_who=set(list(patinfo['infected_by']))-set(['nan'])
# 확진자 id를 정수로 변환
from_who=[int(i) for i in from_who]
#전체 확진자들 중 who가 결측 아닌 사람들 인덱스
who_index=[int(i) for i in range(len(pat_id)) if list(patinfo['infected_by'])[i]!='nan' and who[i] in pat_id]
#감염시킨 사람 -> 감염된 확진자
new_edges=[(int(who[i]),int(pat_id[i]))for i in who_index]

#집단 감염간에는 링크가 연결되지 않는다
#감염장소 ->확진자 연결
num_case=[(patient_case[num],int(pat_id[num])) for num in range(len(pat_id)) if patient_case[num] in infect_case]
#집단감염된 확진자 리스트
patient_list=[i[1] for i in num_case]
case_list=list(set([i[0] for i in num_case]))

def get_date(pati_id):
    index=pat_id.index(pati_id)
    str_day=list(patinfo['confirmed_date'])[index]
    str_day=str_day.split('-')
    date=''.join(str_day)
    date=int(date)
    return date

new_nodes_1=[i[0] for i in new_edges if i[0] not in patient_list]
new_nodes_2=[i[1] for i in new_edges if i[1] not in patient_list]
patient_list.extend(new_nodes_1)
patient_list.extend(new_nodes_2)
patient_list=list(set(patient_list))

edge_list=list(num_case)
edge_list.extend(new_edges)

G=nx.Graph()
G.add_nodes_from(patient_list,bipartite=1)
G.add_nodes_from(case_list,bipartite=0)
G.add_edges_from(edge_list)
components_covid=[x for x in sorted(nx.connected_components(G),key=len,reverse=True)]
num_node_compo=[len(x) for x in sorted(nx.connected_components(G),key=len,reverse=True)]
first=num_node_compo.index(5)
smaller=components_covid[first:]
trash=set()
for i in smaller:
    trash=trash|i
patient_list=list(set(patient_list)-trash)   
case_list=list(set(case_list)-trash)
edge_list=[i for i in edge_list if i[0] not in list(trash) and i[1] not in list(trash)]
#확진자들의 인덱스
list_ind=[pat_id.index(i) for i in patient_list]

#networkx와 연계해서 딕셔너리 형태 구하기
dict_51={}
for i in case_list:
    dict_51[i]=list(set(G[i])-set(case_list))
    
for i in case_list:
    date=[get_date(j) for j in dict_51[i]]
    first_infected=[dict_51[i][k] for k in range(len(date)) if date[k]==min(date)]
    for k in first_infected:
        edge_list=list(set(edge_list)-set([(i,k)]))
        edge_list.append((k,i))

g=Network(height=800,width=1600,directed=True,notebook=True)
g.set_options("""
var options = {
  "nodes": {
    "font": {
      "size": 100,
      "strokeColor": "rgba(165,215,255,1)"}}}
""")

for i in case_list:
    g.add_node(i,title=i,color='gray',label=i,shape='star')

for i in list_ind:
    id_pat=list(patinfo['patient_id'])[i]
    id_pat_str=str(list(patinfo['patient_id'])[i])
    age=list(patinfo['age'])[i]
    sex=list(patinfo['sex'])[i]
    
    if age=='0s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='purple',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='purple',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='purple',size=12,title=id_pat_str)
    elif age=='10s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='indigo',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='indigo',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='indigo',size=12,title=id_pat_str)
    elif age=='20s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='blue',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='blue',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='blue',size=12,title=id_pat_str)
    elif age=='30s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='skyblue',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='skyblue',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='skyblue',size=12,title=id_pat_str)
    elif age=='40s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='green',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='green',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='green',size=12,title=id_pat_str)
    elif age=='50s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='lawngreen',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='lawngreen',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='lawngreen',size=12,title=id_pat_str)
    elif age=='60s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='yellow',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='yellow',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='yellow',size=12,title=id_pat_str)
    elif age=='70s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='orange',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='orange',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='orange',size=12,title=id_pat_str)
    elif age=='80s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='red',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='red',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='red',size=12,title=id_pat_str)
    elif age=='90s':
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='brown',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='brown',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='brown',size=12,title=id_pat_str)
    else:
        if sex=='male':
            g.add_node(id_pat,label=[' '],shape='square',color='black',size=12,title=id_pat_str)
        elif sex=='female':
            g.add_node(id_pat,label=[' '],shape='dot',color='black',size=12,title=id_pat_str)
        else:
            g.add_node(id_pat,label=[' '],shape='triangle',color='black',size=12,title=id_pat_str)

for i in edge_list:
    g.add_edge(source=i[0],to=i[1])

g.show('contact_age_sex.html') # 직접 네트워크를 출력하시려면, 맨 앞에 있는 #을 제거하시면 됩니다.

patinfo.loc[patinfo["infection_case"]=="gym facility in Cheonan","sex"].value_counts()

# 신천지 교인들 중에서 2차감염한 리스트
lst_2nd=patinfo.loc[patinfo["infected_by"].isin(patinfo.loc[(patinfo["infection_case"]=="Shincheonji Church"),"patient_id"].tolist()),"infected_by"].unique().tolist()

all_shin=patinfo.loc[(patinfo["infection_case"]=="Shincheonji Church"),"age" ].value_counts() # 신천지 케이스의 나이별 환자수
inf_shin=patinfo.loc[patinfo["patient_id"].isin(lst_2nd),"age"].value_counts() # 신천지 환자 중에서 2차감염을 일으킨 환자의 나이별 수

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.bar(np.arange(all_shin.shape[0]),all_shin,color="g")
plt.xticks(np.arange(all_shin.shape[0]),all_shin.index)
plt.xlabel("age")
plt.ylabel("count")
plt.title("the number of patient by age(shincheonji)")
plt.subplot(1,2,2)
plt.bar(np.arange(inf_shin.shape[0]),inf_shin)
plt.xticks(np.arange(inf_shin.shape[0]),inf_shin.index)
plt.xlabel("age")
plt.ylabel("count")
plt.title("the number of spreader by age(shincheonji)")
plt.show()

col_patients=[]
for i in list_ind:
    region=int(str(list(patinfo['patient_id'])[i])[:2])
    if region==10:
        col_patients.append('red')
    elif region==11:
        col_patients.append('aqua')
    elif region==12:
        col_patients.append('green')
    elif region==13:
        col_patients.append('purple')
    elif region==14:
        col_patients.append('orange')
    elif region==15:
        col_patients.append('lawngreen')
    elif region==16:
        col_patients.append('blue')
    elif region==17:
        col_patients.append('yellow')
    elif region==20:
        col_patients.append('crimson')
    elif region==30:
        col_patients.append('coral')
    elif region==40:
        col_patients.append('yellowgreen')
    elif region==41:
        col_patients.append('limegreen')
    elif region==50:
        col_patients.append('plum')
    elif region==51:
        col_patients.append('darkmagenta')
    elif region==60:
        col_patients.append('springgreen')
    else:
        col_patients.append('deepskyblue')

label_patients=[]
for i in list_ind:
    region=int(str(list(patinfo['patient_id'])[i])[:2])
    number=int(str(list(patinfo['patient_id'])[i])[4:])
    if region==10:
        label_patients.append('Seoul_%d' %number)
    elif region==11:
        label_patients.append('Busan_%d' %number)
    elif region==12:
        label_patients.append('Daegu_%d' %number)
    elif region==13:
        label_patients.append('Gwangju_%d' %number)
    elif region==14:
        label_patients.append('Incheon_%d' %number)
    elif region==15:
        label_patients.append('Daejeon_%d' %number)
    elif region==16:
        label_patients.append('Ulsan_%d' %number)
    elif region==17:
        label_patients.append('Sejong_%d' %number)
    elif region==20:
        label_patients.append('Gyeonggi_%d' %number)
    elif region==30:
        label_patients.append('Gangwon_%d' %number)
    elif region==40:
        label_patients.append('Chungbuk_%d' %number)
    elif region==41:
        label_patients.append('Chungnam_%d' %number)
    elif region==50:
        label_patients.append('Jeonbuk_%d' %number)
    elif region==51:
        label_patients.append('Jeonnam_%d' %number)
    elif region==60:
        label_patients.append('Gyeongbuk_%d' %number)
    else:
        label_patients.append('Gyeongnam_%d' %number)

size_case=[int(np.log(case.loc[case["infection_case"]==i, "confirmed"].values.sum()))*7 for i in case_list]

g=Network(height=1050,width=1680,directed=True,notebook=True)
g.add_nodes(patient_list,color=col_patients,label=[' ']*len(patient_list),title=label_patients,size=[10]*len(patient_list))
g.add_nodes(case_list,color=['gray']*len(case_list),title=case_list,size=size_case)
outcase=list(set(infect_case)-set(case_list))
g.add_nodes(outcase,size=[10]*len(outcase),color=['gray']*len(outcase))
g.set_options("""
var options = {
  "nodes": {
    "font": {
      "size": 100,
      "strokeColor": "rgba(165,215,255,1)"}}}
""")

for i in edge_list:
        g.add_edge(source=i[0],to=i[1])
g.show('contact_group_modi.html') #네트워크를 직접 만드시려면 맨 앞의 #을 지우세요.

#필터 없앤 버전의 네트워크가 필요하기 때문
infect_case=list(set(case['infection_case'])-set(['etc','contact with patient','overseas inflow']))
patinfo.fillna('nan',inplace=True)
pat_id=list(patinfo['patient_id'])
patient_case=list(patinfo['infection_case'])
who=list(patinfo['infected_by'])
from_who=set(list(patinfo['infected_by']))-set(['nan'])
from_who=[int(i) for i in from_who]
who_index=[int(i) for i in range(len(pat_id)) if list(patinfo['infected_by'])[i]!='nan' and who[i] in pat_id]
new_edges=[(int(who[i]),int(pat_id[i]))for i in who_index]
num_case=[(patient_case[num],int(pat_id[num])) for num in range(len(pat_id)) if patient_case[num] in infect_case]
patient_list=[i[1] for i in num_case]
new_nodes_1=[i[0] for i in new_edges if i[0] not in patient_list]
new_nodes_2=[i[1] for i in new_edges if i[1] not in patient_list]
patient_list.extend(new_nodes_1)
patient_list.extend(new_nodes_2)
patient_list=list(set(patient_list))
case_list=list(set([i[0] for i in num_case]))
edge_list=list(num_case)
edge_list.extend(new_edges)

G=nx.DiGraph()
G.add_nodes_from(patient_list,bipartite=1)
G.add_nodes_from(case_list,bipartite=0)
G.add_edges_from(edge_list)

k=dict(G.degree())
deg_nodes=[k[i] for i in G.nodes()]
ele=list(set(deg_nodes))
distribution=list(np.array([deg_nodes.count(i) for i in ele])/len(G.nodes()))
plt.xscale('log')
plt.yscale('log')
plt.xlabel('degree')
plt.ylabel('distribution')
plt.title('degree distribution of all nodes')
plt.plot(ele,distribution,'.')

get_ipython().system('pip install powerlaw')

import powerlaw

fit=powerlaw.Fit(ele,distribution)
alpha=fit.power_law.alpha
print(alpha)

k=dict(G.out_degree())
out_deg=sorted(k.items(),key=lambda x:x[1],reverse=True)
top_5=[i for i in out_deg if i[0] in patient_list][:5]
top_5_nodes=[i[0] for i in top_5]
top_5_count=[i[1] for i in top_5]
print("가장 많이 감염 시킨 확진자 5명:",top_5_nodes)
print("각각 몇명을 감염 시켰는가?",top_5_count)

b=nx.betweenness_centrality(G)
top_5=sorted(b.items(),key=lambda x: x[1],reverse=True)[:5]
N=len(G.nodes())
denominator=(N-2)*(N-1)
top_5_nodes=[i[0] for i in top_5]
top_5_count=[round(i[1]*denominator) for i in top_5]
print(top_5_nodes)
print(top_5_count)

print("2000000205번 확진자가 감염시킨 사람의 수: ",patinfo.loc[patinfo["infected_by"]==2000000205].shape[0])
patinfo.loc[patinfo["infected_by"]==2000000205,"infection_case"].unique()#.shape[0] #51명 감염
region=pd.read_csv(path+r'\data\region.csv') # 지역정보가 있는 데이터 프레임
pinfo=pd.read_csv(path+r'.\data\PatientInfo.csv') # 감염환자의 개인정보가 있는 데이터 프레임

pinfo.loc[(pinfo["city"]=="etc")|(pinfo["city"].isna()),"city"].shape # city가 결측치 혹은 etc인 것이 135건.
drop_index=pinfo.loc[(pinfo["city"]=="etc")|(pinfo["city"].isna()),"city"].index.tolist() # city
pinfo=pinfo.drop(drop_index).reset_index(drop=True)

case_address=case.loc[(~case["latitude"].isna())&(case["latitude"]!="-"),["infection_case","latitude","longitude"]].copy().reset_index(drop=True)
# case데이터에서 집단(group)감염 케이스만 위도 경도 정보가 있기 때문에 
#latitude 정보가 있는 것만 가져와도 집단 감염케이스의 위도, 경도 정보를 모두 가져올 수 있습니다.

#환자 정보(patientinfo)의 시(province)+구(city)로 묶어서 컬럼 만듭니다.
pinfo["address"]=pinfo.apply(lambda x: "|".join([str(x["province"]),str(x["city"])]),axis=1) 

#지역(region)의 시(province)+구(city)를 묶어서 컬럼을 만듭니다. 
#address컬럼은 나중에 df의 address와 연결되어서 필요한 위도 경도를 가져오게 합니다.
region["address"]=region.apply(lambda x: "|".join([str(x["province"]),str(x["city"])]),axis=1)

# 각 케이스 별 주소지(시+구)에 몇명의 환자가 있는지 df에 입력합니다. 
df=pinfo[["infection_case", "address","patient_id"]].groupby(["infection_case","address"]).count().reset_index()

c,not_c=[],[]
for i in case_address["infection_case"]:
    ad=pinfo.loc[pinfo["infection_case"]==i,"address"].nunique()
    if ad!=0:
        c.append(i)
    else: not_c.append(i)
        
df.head(2)

from folium.features import CustomIcon
def make_feature_group(case_name,u,group_n,edges=True): 
    # 환자 개인정보가 있는 집단감염 case의 네트워크 만들기 
    # case_name: 무슨 집단 케이스, u: color_리스트의 인덱스, folium의 feature 그룹
    icon_image="https://raw.githubusercontent.com/minsu1234/Statistics/master/KakaoTalk_20200503_221404958.png" #아이콘 이미지
    icon=CustomIcon(icon_image,icon_size=(25,25))  # 아이콘 이미지와 아이콘 사이즈를 입력합니다. 
    case_lat=float(case_address.loc[case_address["infection_case"]==case_name,"latitude"])  #집단 케이스의 위도
    case_long=float(case_address.loc[case_address["infection_case"]==case_name, "longitude"]) #집단 케이스의 경도

    df2=df.loc[df["infection_case"]==case_name] #case_name를 입력하면 df의 감염케이스 이름, 감염 주소, 감염환자수를 불러옵니다. 
    df3=pd.merge(df2,region[["address","latitude","longitude"]], how="left",on="address") #df의 address에 맞는 위도 경도를 매칭 
    region_count=df3.patient_id.count() # 해당 집단감염 케이스로 인해 감염자가 발생한 지역(시군-city)의 수

    #집단감염케이스의 마커 
    folium.Marker(location=[case_lat,case_long],#집단 케이스의 위도, 경도
                  popup="case: %s / the number of infected region:%d"%(case_name, region_count),#집단케이스이름과 감염시킨 지역의 수
                  icon=icon).add_to(group_n) #group_n에 마커 추가
    
    #집단감염케이스로 인해 감염된 환자들의 거주 지역 edge(네트워크의 선) 그리기
    if edges:
        for i in df3.index:
            patient_lat,patient_long=float(df3["latitude"][i]),float(df3["longitude"][i]) # 감염환자의 주거지(시군-city)의 위도,경도
            patient_num=int(df3["patient_id"][i]) # 해당지역의 환자수
            folium.PolyLine(weight=patient_num/2,# 감염된 환자 수에 따라서 선의 굵기가 굵고 얇아지도록 만들었다.
                            locations=[(case_lat,case_long),(patient_lat, patient_long)], #edge는 한쪽은 집단감염지, 다른 쪽은 감염된 환자의 지역
                            color=color_[u]).add_to(group_n) # 케이스마다 색을 달리해서 group_n에 edge를 추가한다.
    else: pass

def make_icon(case_name,group_n): # 환자 개인정보가 없는 집단감염 case 아이콘 만들기
                               # case_name: 무슨 집단 케이스, , map_: 아이콘은 추가하려는 map의 이름
    icon_image="https://raw.githubusercontent.com/minsu1234/Statistics/master/KakaoTalk_20200503_221404958.png"
    icon=CustomIcon(icon_image,icon_size=(25,25))
    c_index=case.loc[(case["infection_case"]==case_name)&(case["city"]!="from other city")].index
    for i in c_index:
        case_lat, case_long=case.loc[i,"latitude"], case.loc[i,"longitude"]
        confirmed=case.loc[i,"confirmed"]
        folium.Marker(location=[case_lat,case_long],popup="case: %s / the number of infected patients:%d"%(case_name, confirmed),icon=icon).add_to(group_n)    

ko_states=json.load(open("data/skorea-municipalities-2018-topo-simple.json", encoding="utf-8")) # 지도 json을 불러온다

lst_all,lst_in,lst_out=[],[],[]
for i in range(len(ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"])):
    u=ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"][i]["properties"]["name_eng"]
    lst_all.append(u)
    if u in region["city"].tolist():
        lst_in.append(u)
    else: 
        lst_out.append(u)

print("region의 행정명과 일치하는 경우:",len(lst_in), "/ region의 행정명과 일치하지 않는 경우:", len(lst_out))

city_={"Sejongsi":"Sejong", "Suwonsijangangu" :"Suwon-si","Suwonsigwonseongu":"Suwon-si","Suwonsipaldalgu":"Suwon-si","Suwonsiyeongtonggu" : "Suwon-si",
       "Seongnamsisujeonggu":"Seongnam-si","Seongnamsijungwongu":"Seongnam-si","Seongnamsibundanggu":"Seongnam-si",
       "Anyangsimanangu":"Anyang-si","Anyangsidongangu": "Anyang-si","Ansansisangnokgu":"Ansan-si","Ansansidanwongu":"Ansan-si",
       "Goyangsideogyanggu":"Goyang-si","Goyangsiilsandonggu":"Goyang-si" ,"Goyangsiilsanseogu":"Goyang-si",
       "Yonginsicheoingu":"Yongin-si","Yonginsigiheunggu":"Yongin-si","Yonginsisujigu":"Yongin-si", "Hwaseongsi": "Hwaseong-si", 
       "Yangjusi":"Yangju-si","Pocheonsi":"Pocheon-si","Cheongjusisangdanggu":"Cheongju-si","Cheongjusiseowongu":"Cheongju-si",
       "Cheongjusiheungdeokgu":"Cheongju-si","Cheongjusicheongwongu":"Cheongju-si","Jeungpyeonggun":"Jeungpyeong-gun",
       "Cheonansidongnamgu":"Cheonan-si","Cheonansiseobukgu":"Cheonan-si","Gyeryongsi":"Gyeryong-si","Dangjinsi":"Dangjin-si",
       "Jeonjusiwansangu":"Jeonju-si","Jeonjusideokjingu":"Jeonju-si","Uichanggu":"Changwon-si","Seongsangu":"Changwon-si",
       "Masanhappogu":"Changwon-si","Masanhoewongu":"Changwon-si", "Jinhaegu":"Changwon-si","Jeju-si": "Jeju-do","Seogwipo-si":"Jeju-do"}
city_keys=list(city_.keys())

# 지도의 지명(name_eng)과 region의 지명(city)를 일치 시키는 작업
for i in range(len(ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"])):
    u=ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"][i]["properties"]["name_eng"]
    if u in city_keys:
        #print(ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"][i]["properties"])
        ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"][i]["properties"]["name_eng"]=city_[u]
        #print(ko_states["objects"]["skorea_municipalities_2018_geo"]["geometries"][i]["properties"])
        #print("---"*30)

    else:pass

color_=["#FE0606","#FE6B06","#FBD900","#3BB900","#0022B9","#000A6D","#7F02C8","#C802C5","#95FF00","#00F9FF","#298B8B","#6F5A0B","#939B98","#546F9C","#000000","#78DFC2","#FF6E00","#C0FF00","#415600"]

map_case=folium.Map(location=[36,129],zoom_start=7,tiles="Cartodb Positron")
group19=folium.FeatureGroup(name='<span style = " font-size:0.7em;">%s</span> '% ('city'))
folium.Choropleth(
    geo_data=ko_states,
    key_on="properties.name_eng",topojson="objects.skorea_municipalities_2018_geo",fill_opacity=0.0,
    line_opacity=0.2).add_to(group19)
group19.add_to(map_case)

group0 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[0],c[0]))
make_feature_group(c[0],0,group0)
group0.add_to(map_case)

group1 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[1],c[1]))
make_feature_group(c[1],1,group1)
group1.add_to(map_case)

group2 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[2],c[2]))
make_feature_group(c[2],2,group2)
group2.add_to(map_case)

group3 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[3],c[3]))
make_feature_group(c[3],3,group3)
group3.add_to(map_case)

group4 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[4],c[4]))
make_feature_group(c[4],4,group4)
group4.add_to(map_case)

group5 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[5],c[5]))
make_feature_group(c[5],5,group5)
group5.add_to(map_case)

group6 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[6],c[6]))
make_feature_group(c[6],6,group6)
group6.add_to(map_case)

group7 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[7],c[7]))
make_feature_group(c[7],7,group7)
group7.add_to(map_case)

group8 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[8],c[8]))
make_feature_group(c[8],8,group8)
group8.add_to(map_case)

group9 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[9],c[9]))
make_feature_group(c[9],9,group9)
group9.add_to(map_case)

group10 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[10],c[10]))
make_feature_group(c[10],10,group10)
group10.add_to(map_case)

group11 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[11],c[11]))
make_feature_group(c[11],11,group11)
group11.add_to(map_case)

group12 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[12],c[12]))
make_feature_group(c[12],12,group12)
group12.add_to(map_case)

group13 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[13],c[13]))
make_feature_group(c[13],13,group13)
group13.add_to(map_case)

group14 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[14],c[14]))
make_feature_group(c[14],14,group14)
group14.add_to(map_case)

group15 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[15],c[15]))
make_feature_group(c[15],15,group15)
group15.add_to(map_case)

group16 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[16],c[16]))
make_feature_group(c[16],16,group16)
group16.add_to(map_case)

group17 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[17],c[17]))
make_feature_group(c[17],17,group17)
group17.add_to(map_case)

group18 = folium.FeatureGroup(name='<span style = " font-size:0.7em;  color: %s;">%s</span>'%(color_[18],c[18]))
make_feature_group(c[18],18,group18)
group18.add_to(map_case)


dummy=pd.pivot_table(df.loc[df["infection_case"].isin(c)], index=["address","infection_case"]).reset_index()
add=dummy["address"].unique()

df_dum=pd.merge(dummy,region[["address","latitude","longitude"]], how="left",on="address")

for i in add:
    pop_up=df_dum.loc[df_dum["address"]==i,["address","infection_case","patient_id"]]
    pop_up.columns=["address","infection_case","confirmed"]
    html = pop_up.to_html(classes='table table-striped table-hover table-condensed table-responsive')
    patient_marker_lat=df_dum.loc[df_dum["address"]==i,"latitude"].unique()[0]
    patient_marker_lng=df_dum.loc[df_dum["address"]==i,"longitude"].unique()[0]
    folium.CircleMarker([patient_marker_lat,patient_marker_lng],radius=3,popup=html,color="blue").add_to(map_case)



    
folium.map.LayerControl('topright', collapsed=False).add_to(map_case)


map_case.save("layercontrol_map.html")
map_case



group=case.loc[(case["group"]==True), "infection_case"].unique().tolist()
non_group=case.loc[(case["group"]==False)&(case["infection_case"]!='overseas inflow'), "infection_case"].unique().tolist() 
group_df=df.loc[df["infection_case"].isin(group)]

ngroup_df=df.loc[df["infection_case"].isin(non_group)]
ng=ngroup_df[["address","patient_id"]].groupby("address").sum().reset_index()
ng.columns=["address","non_group_patient"]

gr=group_df[["address","patient_id"]].groupby("address").sum().reset_index()
gr.columns=["address","group_patient"]

df_region=pd.merge(ng,gr, how="outer", on="address").fillna(0)
df_region["province"]=df_region["address"].map(lambda x: x.split("|")[0])
df_region["city"]=df_region["address"].map(lambda x: x.split("|")[1])

df_re=region.loc[~region["city"].isin(df_region["city"]),["province","city"]].copy() 
# region데이터에서 df_region의 city컬럼에 없는 데이터만 가져온다.
dummy_=np.repeat(0,df_re.shape[0]) 
non_p=pd.DataFrame(data={"address":dummy_, "non_group_patient":dummy_, "group_patient":dummy_, "province":df_re["province"],"city":df_re["city"]})
df_region=pd.concat([df_region, non_p])

pp=df_region[["province","non_group_patient","group_patient"]].groupby("province").sum()
pp.head()

print("집단감염자와 비집단감염자의 상관계수:",pp["non_group_patient"].corr(pp["group_patient"]))

map_=folium.Map(location=[35.8,127.6],zoom_start=7,tiles="Cartodb Positron")

for i in range(0,19):
    make_feature_group(c[i],i,map_,edges=False)
myscale=(df_region["non_group_patient"].quantile((0,0.6,0.7,0.8,0.9,0.95,1))).tolist()
folium.Choropleth(
    geo_data=ko_states,
    data=df_region, threshold_scale=myscale,
    columns=["city",'non_group_patient'],
    key_on="properties.name_eng",topojson="objects.skorea_municipalities_2018_geo",fill_color='PuRd',
    legend_name='non-group confirmed',
    fill_opacity=0.5,
    line_opacity=0.4,
).add_to(map_)
map_


# In[ ]:




