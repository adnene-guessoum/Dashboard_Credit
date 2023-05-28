def cleaning(data):
    '''
    reorders the columns of the dataframe
    '''
    #sert à assurer que les ID et Targets
    #sont les deux dernières colonnes
    ID = data['ID']
    target = data['TARGET']
    df = data.drop(['TARGET', 'ID'], axis = 1)
    df['ID'] = ID
    df['TARGET'] = target
    return df
