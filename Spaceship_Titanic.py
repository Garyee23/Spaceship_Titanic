import streamlit as st
import pandas as pd
import io
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

data_path = 'Spaceship_Titanic/'
train = pd.read_csv(data_path + 'train.csv', parse_dates=['PassengerId'])
test = pd.read_csv(data_path + 'test.csv', parse_dates=['PassengerId'])
submission = pd.read_csv(data_path + 'sample_submission.csv', parse_dates=['PassengerId'])

def feature_engineering():
    train[["Deck", "Cabin_num", "Side"]] = train["Cabin"].str.split("/", expand=True)

st.title('Spaceship_Titanic')

mnu = st.sidebar.selectbox('메뉴', options=['설명', 'EDA', '시각화', '모델링'])

if mnu == '설명':
    st.subheader('요구사항')
    st.write('''
    우주의 신비를 풀기 위해 여러분의 데이터 과학 기술이 필요한 2912년에 오신 것을 환영합니다. 우리는 4광년 떨어진 곳에서 데이터를 전송 받았고, 확인 결과 상황이 좋지 않은 것 같습니다.
    ''')
    st.write('''
    타이타닉 우주선은 한 달 전에 발사된 성간 여객선이었습니다. 거의 13,000명의 승객이 탑승한 이 우주선은 우리 태양계에서 이주자들을 근처 별 주위를 도는 새로운 거주 가능한 외계 행성 세 곳으로 실어 나르는 처녀 항해를 시작했습니다.
    ''')
    st.write('''
    첫 번째 목적지인 센타우루스자리 55E로 가는 길에서 센타우루스자리 알파를 선회하는 동안, 경계를 하지 않은 타이타닉 우주선은 그만 먼지 구름 안에 숨겨진 시공간 이상 현상과 충돌했습니다. 안타깝게도, 그것은 1000년 전 타이타닉호와 비슷한 운명을 맞이했습니다. 비록 타이타닉 우주선은 무사했지만, 거의 절반의 승객들이 다른 차원으로 옮겨졌습니다!
    ''')
    st.image('https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg')
    st.write('''
    구조대원들을 돕고 잃어버린 승객들을 되찾기 위해, 여러분은 우주선의 손상된 컴퓨터 시스템에서 복구된 기록을 사용하여 어떤 승객들이 이상 현상에 의해 이송되었는지 예측해야 합니다.
    ''')
    st.write('''
    그들을 구하고 역사를 바꾸도록 도와주세요!
    ''')
    st.markdown('#### 데이터 필드')
    st.markdown('**PassengerId** - 각 승객의 고유 ID입니다. 각 ID는 gggg_pp 형식을 취합니다. 여기서 gggg는 승객이 함께 여행하는 그룹을 나타내고 pp는 그룹 내에서 승객의 번호입니다. 한 그룹의 사람들은 대부분 가족 구성원이지만 항상 그런것은 아닙니다.')
    st.markdown('**HomePlanet** - 승객이 떠난 행성, 일반적으로 그들의 거주하던 행성.')
    st.markdown('**CryoSleep** - 승객이 항해 중에 냉동 수면 상태로 전환하였는지 여부를 나타냅니다. 냉동 수면 중인 승객들은 객실 밖을 나가지 못합니다.')
    st.markdown('**Cabin** - 승객이 머무는 객실 번호. deck/num/side 형식을 취합니다. 여기서 side는 좌현의 경우 P 또는 우현의 경우 S입니다.')
    st.markdown('**Destination** - 승객이 내릴 행성.')
    st.markdown('**Age** - 승객의 나이')
    st.markdown('**VIP** - 승객이 항해 중에 특별 VIP 서비스 비용을 지불했는지 여부.')
    st.markdown('**RoomService, FoodCourt, ShoppingMall, Spa, VRDeck** - 승객이 타이타닉 우주선의 다양한 편의 시설에 지불한 금액입니다.')
    st.markdown('**Name** - 승객의 이름과 성.')
    st.markdown('**Transported** - 승객이 다른 차원으로 이송되었는지 여부. 우리가 예측해야하는 데이터 값입니다.')

elif mnu == 'EDA':

    st.subheader('EDA')

    st.markdown('- (훈련 데이터 shape, 테스트 데이터 shape)')
    st.text(f'({train.shape}), ({test.shape})')

    st.markdown('- 훈련 데이터')
    st.dataframe(train.head())

    st.markdown('- 테스트 데이터')
    st.dataframe(test.head())

    st.markdown('- 제출 데이터')
    st.dataframe(submission.head())
    
    st.markdown('- 항목별 데이터 세부분석')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write(train['FoodCourt'].describe())

    with col2:
        st.write(train['ShoppingMall'].describe())

    with col3:
        st.write(train['Spa'].describe())

    with col4:
        st.write(train['VRDeck'].describe())

    st.markdown('- 거주 행성(HomePlanet), 냉동수면여부(CryoSleep), 특별 서비스 유무(VIP), 승객의 나이(Age)에 따른 차원이동비율 비교')
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.text('거주 행성(HomePlanet)')
        st.dataframe(train[['HomePlanet', 'Transported']].groupby(['HomePlanet'], as_index=True).mean().sort_values(by='Transported', ascending=False))

    with col6:
        st.text('냉동수면(CryoSleep)')
        st.dataframe(train[['CryoSleep', 'Transported']].groupby(['CryoSleep'], as_index=True).mean().sort_values(by='Transported', ascending=False))

    with col7:
        st.text('특별 서비스(VIP)')
        st.dataframe(train[['VIP', 'Transported']].groupby(['VIP'], as_index=True).mean().sort_values(by='Transported', ascending=False))

    with col8:
        st.text('승객의 나이(Age)')
        st.dataframe(train[['Age', 'Transported']].groupby(['Age'], as_index=True).mean().sort_values(by='Transported', ascending=True))

    st.markdown('- train.info()')
    buffer = io.StringIO()
    train.info(buf=buffer)
    st.text(buffer.getvalue())

    buffer.truncate(0)  # 버퍼 비우기
    st.markdown('- test.info()')
    test.info(buf=buffer)
    st.text(buffer.getvalue())



elif mnu == '시각화':

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('시각화')

    st.markdown('**:blue[그래프를 해석하세요.]**')

    mpl.rc('font', size=5)

    st.markdown('- Transported의 분포도')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    sns.histplot(train['Transported'])
    plt.xticks(np.arange(0,2,1))
    fig.set_size_inches(1,1)
    st.pyplot()

    st.markdown('- Transported의 분포도(y축 수정)')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    sns.histplot(train['Transported'])
    plt.xticks(np.arange(0, 2, 1))
    plt.ylim(4200,4400)
    fig.set_size_inches(1, 1)
    st.pyplot()
    st.write('원본 Transported값의 차이를 구별하기 어려워서 y축 제한을 두어 더 직관적으로 나타냄.')

    st.markdown('- 차원 이동 여부(Transported)에 따른 연령(Age) 분포')
    g = sns.FacetGrid(train, col='Transported')
    g.map(plt.hist, 'Age', bins=20)
    st.pyplot()


    st.markdown('- Age, FoodCourt, ShoppingMall, Spa, VRDeck의 분포도')
    mpl.rc('font', size=15)
    fig, ax = plt.subplots(5, 1, figsize=(10, 10))
    plt.subplots_adjust(top=2)
    sns.histplot(train['Age'], color='b', bins=50, ax=ax[0]);
    ax1 = sns.histplot(train['FoodCourt'], color='b', bins=50, ax=ax[1]);
    ax1.set_xlim(0,30000)
    ax2 = sns.histplot(train['ShoppingMall'], color='b', bins=50, ax=ax[2]);
    ax2.set_xlim(0,25000)
    ax3 = sns.histplot(train['Spa'], color='b', bins=50, ax=ax[3]);
    ax3.set_xlim(0,23000)
    ax4 = sns.histplot(train['VRDeck'], color='b', bins=50, ax=ax[4]);
    ax4.set_xlim(0,25000)
    st.pyplot()

    st.markdown('- Age, FoodCourt, ShoppingMall, Spa, VRDeck의 분포도(수정)')
    mpl.rc('font', size=15)
    fig, ax = plt.subplots(5, 1, figsize=(10, 10))
    plt.subplots_adjust(top=2)
    sns.histplot(train['Age'], color='b', bins=50, ax=ax[0]);
    sns.histplot(np.log(train['FoodCourt']), color='b', bins=50, ax=ax[1]);
    sns.histplot(np.log(train['ShoppingMall']), color='b', bins=50, ax=ax[2]);
    sns.histplot(np.log(train['Spa']), color='b', bins=50, ax=ax[3]);
    sns.histplot(np.log(train['VRDeck']), color='b', bins=50, ax=ax[4]);
    st.pyplot()
    st.markdown('Foodcourt, ShoppingMall, Spa, VRDeck 값들의 분포가 왼쪽으로 많이 편향되어 있어서 로그변환을 통해 정규분포에 가깝게 나타냄.')

elif mnu == '모델링':

    st.markdown('#### 피처 엔지니어링')
    st.markdown('**필요 없는 피처 제거**')
    st.write('PassengerId 와 Name 열은 모델 훈련에 필요하지 않기 때문에 drop을 진행.')
    st.code('''
    dataset_df = train.drop(['PassengerId', 'Name'], axis=1)
    dataset_df.head(5)
    ''')
    train = train.drop(['PassengerId', 'Name'], axis=1)
    train.head(5)

    st.markdown('**결측치 제거**')
    st.write('다음의 코드를 통해 결측된 값을 확인.')
    st.code('''
    train.isnull().sum().sort_values(ascending=False)
    ''')
    train.isnull().sum().sort_values(ascending=False)
    
    st.write('데이터에는 숫자, 범주형 및 누락된 피처가 혼합되어 있다. TF-DF는 이러한 모든 피처 유형을 기본적으로 지원하므로 사전 처리가 필요하지 않다.')
    st.write('그러나 데이터에는 결측값이 있는 boolean 필드도 있다. TF-DF는 아직 boolean 필드를 지원하지 않으므로 boolean필드들을 int로 변환해야한다. boolean 필드의 결측값을 설명하기 위해 0으로 대체해보자.')
    st.write('여기서는 숫자 열의 null 값 항목도 0으로 바꾸고, 범주형 열의 결측값만 TF-DF가 처리하도록 할 것이다.')
    st.markdown('**참고 : 필요한 경우 TF-DF가 숫자 열의 결측값을 처리하도록 선택할 수 있다.**')
    st.code('''
    train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    train.isnull().sum().sort_values(ascending=False)
    ''')
    train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(value=0)
    train.isnull().sum().sort_values(ascending=False)

    st.write('TF-DF는 boolean 열을 처리할 수 없기 때문에 Transported 열의 레이블을 조정하여 TF-DF가 예상하는 정수 형식으로 변환해야 한다.')
    st.code('''
    label = "Transported"
    train[label] = train[label].astype(int)
    ''')
    label = "Transported"
    train[label] = train[label].astype(int)

    st.write('또한 boolean필드, CryoSleep 및 VIP를 int로 변환할 것이다.')
    st.code('''
    train['VIP'] = train['VIP'].astype(int)
    train['CryoSleep'] = train['CryoSleep'].astype(int)
    ''')
    train['VIP'] = train['VIP'].astype(int)
    train['CryoSleep'] = train['CryoSleep'].astype(int)

    st.write('Cabin 열 값은 Deck/Cabin_num,Side 형식의 문자열이다. 여기서는 Cabin 열을 분할하고 Deck, Cabin_num 및 Side 열 3개를 새로 만든다. 이러한 개별 데이터를 통해 모델을 훈련시키는 것이 더 쉽다.')
    st.write('따라서 다음 코드를 통해 Cabin 열을 Deck, Cabin_num 및 Side 열로 분할한다.')
    st.code('''
    train[["Deck", "Cabin_num", "Side"]] = train["Cabin"].str.split("/", expand=True)
    ''')
    train[["Deck", "Cabin_num", "Side"]] = train["Cabin"].str.split("/", expand=True)

    st.write('원래의 Cabin 열을 더이상 필요하지 않으므로 삭제한다.')
    st.code('''
    try:
        train = train.drop('Cabin', axis=1)
    except KeyError:
        print("Field does not exist")
    ''')
    try:
        train = train.drop('Cabin', axis=1)
    except KeyError:
        print("Field does not exist")
        
    st.write('피처 엔지니어링을 통해 준비된 데이터를 확인해보자')
    st.dataframe(train.head(5))

    st.markdown('**데이터 나누기**')

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = train.drop(['Transported'], axis=1)
    y = train['Transported']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    rfc = RandomForestClassifier(n_estimators=600, max_depth=18, random_state=42, min_samples_leaf=4)
    rfc.fit(X_train, y_train)

    pred = rfc.predict(X_val)
    a = accuracy_score(y_val, pred)
    st.write("Accuracy Random Forest Classifier : ", round(accuracy_score(y_val, pred), 4) * 100, '%')

    # Load the test dataset
    test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')
    submission_id = test.PassengerId

    # Replace NaN values with zero
    test[['VIP', 'CryoSleep']] = test[['VIP', 'CryoSleep']].fillna(value=0)

    # Creating New Features - Deck, Cabin_num and Side from the column Cabin and remove Cabin
    test[["Deck", "Cabin_num", "Side"]] = test["Cabin"].str.split("/", expand=True)
    test_df = test.drop('Cabin', axis=1)

    # Convert boolean to 1's and 0's
    test_df['VIP'] = test_df['VIP'].astype(int)
    test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

    pred_final = rfc.predict(test)

    a = test['PassengerId']
    x = pd.DataFrame(a)
    b = pred_final
    y = pd.DataFrame(b)

    final = pd.concat([x, y], axis=1)
    final.replace(0, False, inplace=True)
    final.replace(1, True, inplace=True)
    final.rename(columns={0: 'Transported'}, inplace=True)
    st.dataframe(final)
